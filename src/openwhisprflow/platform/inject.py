"""Injector factory (SPEC §6.2): picks the backend for this OS and session.

The app decides *when* to paste instead of type (``inject.method``, ``paste_over_chars``,
or after a typing InjectError); injectors type exactly the text they are given, so the
leading/trailing space comes from :func:`openwhisprflow.text.spacing.join_text`.
"""

from __future__ import annotations

import sys

from openwhisprflow.platform.base import Injector


def create(*, newline: str = "enter") -> Injector:
    """``newline``: "enter" (default) or "shift_enter", which inserts a line break instead of
    sending the message in chat apps (Slack, Discord, Teams, ChatGPT)."""
    if sys.platform == "win32":
        from openwhisprflow.platform.inject_windows import WindowsInjector

        return WindowsInjector(newline=newline)
    if sys.platform == "darwin":
        from openwhisprflow.platform.inject_macos import MacInjector

        return MacInjector(newline=newline)
    from openwhisprflow.platform.inject_linux import LinuxInjector

    return LinuxInjector(newline=newline)
