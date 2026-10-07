"""The focused app on any OS (also available as ``Injector.focus()``)."""

from __future__ import annotations

import sys

from openwhisprflow.platform.base import FocusInfo


def get_focus() -> FocusInfo:
    """Never raises: an unknown focus is an empty FocusInfo."""
    try:
        if sys.platform == "win32":
            from openwhisprflow.platform.focus_windows import get_focus as impl
        elif sys.platform == "darwin":
            from openwhisprflow.platform.focus_macos import get_focus as impl
        else:
            from openwhisprflow.platform.focus_linux import get_focus as impl
        return impl()
    except Exception:
        return FocusInfo()
