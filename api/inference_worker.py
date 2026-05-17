"""Subprocess entrypoint that owns the F5-TTS model.

Run in its own process so the parent FastAPI server never imports torch
or holds GPU/ROCm resources. The parent starts the worker on demand and
terminates it after an idle period; killing the process is the only
reliable way to release ROCm allocations on some configurations.
"""

import logging
import os


def worker_main(req_q, resp_q):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [worker %(process)d] %(message)s',
    )
    logger = logging.getLogger("f5tts.worker")

    try:
        from f5_tts.api import F5TTS
        logger.info("Loading F5-TTS model (pid=%d)...", os.getpid())
        model = F5TTS(model="F5TTS_v1_Base")
        logger.info("F5-TTS model loaded")
    except Exception as exc:
        logger.exception("Failed to load F5-TTS model")
        try:
            resp_q.put({"id": -1, "type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            logger.exception("Failed to report fatal load error to parent")
        return

    try:
        resp_q.put({"id": -1, "type": "ready"})
    except Exception:
        logger.exception("Failed to send ready handshake")
        return

    while True:
        try:
            msg = req_q.get()
        except (EOFError, OSError):
            logger.info("Request queue closed; worker exiting")
            return

        if msg is None or msg.get("type") == "shutdown":
            logger.info("Worker received shutdown request")
            return

        if msg.get("type") != "infer":
            logger.warning("Worker received unknown message type: %s", msg.get("type"))
            continue

        req_id = msg["id"]
        args = msg.get("args", {})
        try:
            model.infer(show_info=logger.info, **args)
            resp_q.put({
                "id": req_id,
                "ok": True,
                "result": {"used_seed": model.seed},
            })
        except Exception as exc:
            logger.exception("Inference %s failed", req_id)
            resp_q.put({
                "id": req_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
