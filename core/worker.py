"""
Background Worker — xử lý AI trong thread riêng, không block UI.
"""
import threading
import traceback
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


class ProcessingWorker:
    """
    Chạy một hàm xử lý trong thread nền.
    Callbacks: on_progress(pct, msg), on_done(result), on_error(msg)
    """

    def __init__(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: dict = None,
        on_progress: Callable[[float, str], None] = None,
        on_done: Callable[[Any], None] = None,
        on_error: Callable[[str], None] = None,
    ):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}
        self.on_progress = on_progress or (lambda pct, msg: None)
        self.on_done = on_done or (lambda result: None)
        self.on_error = on_error or (lambda msg: None)

        self._thread: threading.Thread | None = None
        self._cancel_flag = threading.Event()

    def start(self):
        self._cancel_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel_flag.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_flag.is_set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        try:
            # Inject cancel_flag và progress_callback nếu fn accept
            kwargs = dict(self.kwargs)
            import inspect
            sig = inspect.signature(self.fn)
            if "cancel_flag" in sig.parameters:
                kwargs["cancel_flag"] = self._cancel_flag
            if "progress_cb" in sig.parameters:
                kwargs["progress_cb"] = self.on_progress

            result = self.fn(*self.args, **kwargs)

            if not self.is_cancelled:
                self.on_done(result)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(msg)
            self.on_error(str(e))


class BatchWorker:
    """
    Xử lý hàng loạt file với progress per-file và tổng thể.
    """

    def __init__(
        self,
        items: list,
        process_fn: Callable,
        on_item_done: Callable[[int, Any], None] = None,
        on_item_error: Callable[[int, str], None] = None,
        on_progress: Callable[[float, str], None] = None,
        on_all_done: Callable[[list], None] = None,
        max_workers: int = 1,
    ):
        self.items = items
        self.process_fn = process_fn
        self.on_item_done = on_item_done or (lambda i, r: None)
        self.on_item_error = on_item_error or (lambda i, e: None)
        self.on_progress = on_progress or (lambda pct, msg: None)
        self.on_all_done = on_all_done or (lambda results: None)
        self.max_workers = max_workers

        self._cancel_flag = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._cancel_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel_flag.set()

    def _run(self):
        results = []
        total = len(self.items)
        for i, item in enumerate(self.items):
            if self._cancel_flag.is_set():
                break
            try:
                pct = i / total * 100
                self.on_progress(pct, f"[{i+1}/{total}] {item}")
                result = self.process_fn(item, cancel_flag=self._cancel_flag)
                results.append(result)
                self.on_item_done(i, result)
            except Exception as e:
                logger.error(f"Batch error on item {i}: {e}")
                self.on_item_error(i, str(e))
                results.append(None)

        self.on_progress(100.0, "Hoàn thành!")
        self.on_all_done(results)
