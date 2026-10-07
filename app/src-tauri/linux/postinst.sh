#!/bin/sh
# Apply the uinput rule now, so typing works without a reboot (the logged-in user gets the ACL).
modprobe uinput 2>/dev/null || true
if command -v udevadm >/dev/null 2>&1; then
  udevadm control --reload-rules 2>/dev/null || true
  udevadm trigger --name-match=uinput 2>/dev/null || true
fi
exit 0
