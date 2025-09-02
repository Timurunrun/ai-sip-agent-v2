import queue
from typing import Callable, Any, Dict


class CommandQueue:
    """Thread-safe queue to marshal PJSUA2 calls onto the main thread.

    Only execute() from the main thread that drives PJSUA2.
    """

    def __init__(self):
        self._q: "queue.Queue[tuple[Callable, tuple, Dict]]" = queue.Queue()

    def put(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        self._q.put((func, args, kwargs))

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