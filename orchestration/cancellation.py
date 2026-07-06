# =================================================================
# MODULE: orchestration/cancellation.py
# VERSION: 5.6.1
# DESCRIPTION: Real thread cancellation for pipeline stages.
#
# FIXES: #1 - Timeout actually stops work
# =================================================================

import ctypes
import threading
import inspect
from typing import Optional, Any


class CancellableThread(threading.Thread):
    """Thread that can be truly cancelled (not just timeout wait)."""

    def __init__(self, target, args=None, kwargs=None):
        super().__init__(target=target, args=args or (), kwargs=kwargs or {})
        self._exc = None

    def cancel(self):
        """Raise KeyboardInterrupt in the target thread."""
        if not self.is_alive():
            return

        # Get thread ID
        tid = self.ident
        if tid is None:
            return

        # Raise exception in target thread
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid),
            ctypes.py_object(KeyboardInterrupt)
        )

    def run_with_timeout(self, timeout_seconds: float) -> tuple[bool, Optional[Any], Optional[Exception]]:
        """Run target with real cancellation on timeout."""
        self.start()
        self.join(timeout_seconds)

        if self.is_alive():
            # Thread still running - cancel it
            self.cancel()
            self.join(1.0)  # Give it time to clean up

            if self.is_alive():
                # Still alive - can't force harder, but at least we tried
                return False, None, TimeoutError(f"Stage exceeded {timeout_seconds}s and could not be cancelled")

            return False, None, TimeoutError(f"Stage exceeded {timeout_seconds}s and was cancelled")

        # Thread completed
        return True, None, None