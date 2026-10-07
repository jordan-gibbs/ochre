"""Import first in every entry script. Python 3.12's platform module queries WMI on Windows
(numpy.testing / scipy call platform.machine()); on this box that WMI call intermittently dies
with 0x8007000e and kills the process without a traceback. Without `_wmi`, platform falls back
to the registry/environment."""

import sys

sys.modules.setdefault("_wmi", None)  # type: ignore[arg-type]
