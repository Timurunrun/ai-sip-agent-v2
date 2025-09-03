import queue
from typing import Callable, Any, Dict
from utils.logging import get_logger


class CommandQueue:
    """Thread-safe queue to marshal PJSUA2 calls onto the main thread.

    Only execute() from the main thread that drives PJSUA2.
    """

    def __init__(self):
        self._q: "queue.Queue[tuple[Callable, tuple, Dict]]" = queue.Queue()
        self._log = get_logger("sip.cmdq")

    def put(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        self._q.put((func, args, kwargs))
        # Debug-level to avoid chatty logs by default
        self._log.debug("Enqueued", size=str(self._q.qsize()))

    def execute_pending(self) -> None:
        while True:
            try:
                func, args, kwargs = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                func(*args, **kwargs)
            finally:
                self._q.task_done()
