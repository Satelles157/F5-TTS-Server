"""Parent-side manager for the F5-TTS inference worker subprocess."""

import asyncio
import logging
import multiprocessing as mp
import queue
import threading
import time

from api.inference_worker import worker_main

logger = logging.getLogger(__name__)


class InferenceManager:
    """Owns a single F5-TTS worker subprocess.

    The worker is spawned lazily on the first inference request and
    fully terminated after `idle_timeout` seconds of inactivity, which
    releases all GPU/ROCm allocations the model held.
    """

    WORKER_STARTUP_TIMEOUT = 300

    def __init__(self, idle_timeout: int):
        self._idle_timeout = idle_timeout
        self._mp_ctx = mp.get_context("spawn")
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._active = 0
        self._last_activity = time.time()
        self._lifecycle_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._idle_task: asyncio.Task | None = None

    def start_monitor(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._idle_timeout > 0:
            logger.info(
                "Worker idle auto-stop enabled: timeout=%ds", self._idle_timeout
            )
            self._idle_task = asyncio.create_task(self._idle_monitor())
        else:
            logger.info("Worker idle auto-stop disabled (MODEL_IDLE_TIMEOUT=0)")

    async def infer(self, **kwargs) -> dict:
        await self._ensure_worker()

        # No awaits between this check and the future registration below:
        # the death-handler callback can only interleave at await points,
        # so capturing the live worker state here synchronously guarantees
        # the future we register will be visible to the death handler if
        # the worker dies later.
        proc = self._proc
        req_q = self._req_q
        if proc is None or not proc.is_alive() or req_q is None:
            raise RuntimeError("Worker process is not available")

        self._active += 1
        self._last_activity = time.time()
        try:
            fut: asyncio.Future = self._loop.create_future()
            req_id = self._next_id
            self._next_id += 1
            self._pending[req_id] = fut
            try:
                req_q.put({"id": req_id, "type": "infer", "args": kwargs})
            except Exception:
                self._pending.pop(req_id, None)
                raise
            return await fut
        finally:
            self._active -= 1
            self._last_activity = time.time()

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, Exception):
                pass
            self._idle_task = None
        async with self._lifecycle_lock:
            await self._stop_worker_locked()

    async def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        async with self._lifecycle_lock:
            if self._proc is not None and not self._proc.is_alive():
                await self._stop_worker_locked()
            if self._proc is None:
                await self._spawn_worker_locked()

    async def _spawn_worker_locked(self) -> None:
        loop = asyncio.get_running_loop()
        logger.info("Starting F5-TTS worker process...")
        self._req_q = self._mp_ctx.Queue()
        self._resp_q = self._mp_ctx.Queue()
        self._proc = self._mp_ctx.Process(
            target=worker_main,
            args=(self._req_q, self._resp_q),
            daemon=True,
            name="f5tts-worker",
        )
        self._proc.start()

        try:
            msg = await loop.run_in_executor(
                None,
                lambda: self._resp_q.get(timeout=self.WORKER_STARTUP_TIMEOUT),
            )
        except queue.Empty:
            await self._force_stop_worker()
            raise RuntimeError(
                f"Worker did not become ready within {self.WORKER_STARTUP_TIMEOUT}s"
            )

        if msg.get("type") == "fatal":
            await self._force_stop_worker()
            raise RuntimeError(f"Worker failed to start: {msg.get('error')}")
        if msg.get("type") != "ready":
            await self._force_stop_worker()
            raise RuntimeError(f"Unexpected handshake message from worker: {msg}")

        logger.info("Worker process ready (pid=%d)", self._proc.pid)
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._read_responses,
            daemon=True,
            name="f5tts-reader",
        )
        self._reader_thread.start()
        self._last_activity = time.time()

    def _read_responses(self) -> None:
        while not self._reader_stop.is_set():
            try:
                msg = self._resp_q.get(timeout=0.5)
            except queue.Empty:
                if self._proc is None or not self._proc.is_alive():
                    if not self._reader_stop.is_set():
                        self._loop.call_soon_threadsafe(self._handle_unexpected_death)
                    return
                continue
            except (EOFError, OSError, ValueError):
                if not self._reader_stop.is_set():
                    self._loop.call_soon_threadsafe(self._handle_unexpected_death)
                return
            self._loop.call_soon_threadsafe(self._dispatch_response, msg)

    def _dispatch_response(self, msg: dict) -> None:
        req_id = msg.get("id")
        fut = self._pending.pop(req_id, None)
        if fut is None or fut.done():
            return
        if msg.get("ok"):
            fut.set_result(msg.get("result"))
        else:
            fut.set_exception(RuntimeError(msg.get("error", "Worker reported error")))

    def _handle_unexpected_death(self) -> None:
        if self._reader_stop.is_set():
            return
        logger.error("Worker process died unexpectedly")
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(RuntimeError("Worker process died"))
        self._pending.clear()
        asyncio.create_task(self._async_cleanup_dead_worker())

    async def _async_cleanup_dead_worker(self) -> None:
        async with self._lifecycle_lock:
            if self._proc is not None and not self._proc.is_alive():
                await self._stop_worker_locked()

    async def _idle_monitor(self) -> None:
        check_interval = max(1, min(60, self._idle_timeout // 2 or self._idle_timeout))
        logger.info(
            "Idle monitor started (timeout=%ds, check_interval=%ds)",
            self._idle_timeout, check_interval,
        )
        while True:
            await asyncio.sleep(check_interval)
            if self._proc is None or not self._proc.is_alive():
                continue
            if self._active > 0:
                continue
            if time.time() - self._last_activity < self._idle_timeout:
                continue
            async with self._lifecycle_lock:
                if (
                    self._proc is not None
                    and self._proc.is_alive()
                    and self._active == 0
                    and time.time() - self._last_activity >= self._idle_timeout
                ):
                    logger.info(
                        "Worker idle for >%ds – stopping to free GPU memory",
                        self._idle_timeout,
                    )
                    await self._stop_worker_locked()

    async def _stop_worker_locked(self) -> None:
        proc = self._proc
        if proc is None:
            return
        loop = asyncio.get_running_loop()
        if proc.is_alive():
            try:
                self._req_q.put({"type": "shutdown"})
            except Exception:
                logger.exception("Failed to send shutdown to worker")
            await loop.run_in_executor(None, lambda: proc.join(10))
        if proc.is_alive():
            logger.warning("Worker did not shut down gracefully; terminating")
            proc.terminate()
            await loop.run_in_executor(None, lambda: proc.join(5))
        if proc.is_alive():
            logger.warning("Worker did not terminate; killing")
            proc.kill()
            await loop.run_in_executor(None, lambda: proc.join(5))

        self._reader_stop.set()
        reader = self._reader_thread
        if reader is not None and reader.is_alive():
            await loop.run_in_executor(None, lambda: reader.join(5))
        self._reader_thread = None

        for q in (self._req_q, self._resp_q):
            if q is None:
                continue
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
        self._req_q = None
        self._resp_q = None
        self._proc = None

        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(RuntimeError("Worker process stopped"))
        self._pending.clear()
        logger.info("Worker process stopped")

    async def _force_stop_worker(self) -> None:
        proc = self._proc
        loop = asyncio.get_running_loop()
        if proc is not None and proc.is_alive():
            proc.terminate()
            await loop.run_in_executor(None, lambda: proc.join(5))
            if proc.is_alive():
                proc.kill()
                await loop.run_in_executor(None, lambda: proc.join(5))
        self._reader_stop.set()
        for q in (self._req_q, self._resp_q):
            if q is None:
                continue
            try:
                q.close()
            except Exception:
                pass
        self._req_q = None
        self._resp_q = None
        self._proc = None
