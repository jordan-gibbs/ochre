"""The Voice key gesture state machine (ported from the earlier native tests, extended)."""

from __future__ import annotations

import pytest

from openwhisprflow.platform.base import Gesture as G
from openwhisprflow.platform.gesture import Decision, GestureMachine, Phase, event_time, safe_swallow


def m(**kw) -> GestureMachine:
    return GestureMachine(double_tap_ms=350, hold_min_ms=250, **kw)


def test_hold_records_while_held_and_release_finishes():
    g = m()
    d = g.key(True, 0)
    assert d.swallow and d.gestures == [G.PRESS]
    assert g.tick(249).gestures == [] and g.phase == Phase.PRESSED
    g.tick(250)
    assert g.phase == Phase.HELD
    d = g.key(False, 900)
    assert d.swallow and d.gestures == [G.RELEASE] and g.phase == Phase.IDLE


def test_release_past_threshold_without_tick_is_still_a_hold():
    g = m()
    g.key(True, 0)
    assert g.key(False, 260).gestures == [G.RELEASE]  # the ticker was late: timing decides


def test_auto_repeat_is_swallowed_and_silent():
    g = m()
    g.key(True, 0)
    for t in range(30, 600, 30):
        d = g.key(True, t)
        assert d.swallow and d.gestures == []
    assert g.key(False, 620).gestures == [G.RELEASE]


def test_double_tap_locks_and_single_tap_finishes():
    g = m()
    assert g.key(True, 0).gestures == [G.PRESS]
    d = g.key(False, 80)
    assert d.swallow and d.gestures == [] and g.phase == Phase.WAITING
    d = g.key(True, 300)
    assert d.swallow and d.gestures == [G.LOCK] and g.phase == Phase.LOCKED
    d = g.key(False, 360)  # the locking tap's own release is ignored
    assert d.swallow and d.gestures == []
    assert g.tick(5000).gestures == []  # locked recording runs hands-off
    d = g.key(True, 6000)
    assert d.swallow and d.gestures == []
    d = g.key(False, 6080)
    assert d.swallow and d.gestures == [G.FINISH] and g.phase == Phase.IDLE


def test_long_press_while_locked_also_finishes_on_release():
    g = m()
    g.key(True, 0), g.key(False, 50), g.key(True, 200), g.key(False, 250)
    g.key(True, 1000)
    g.tick(2000)
    assert g.key(False, 2500).gestures == [G.FINISH]


def test_lone_tap_cancels_after_window_and_replays_the_tap():
    g = m()
    g.key(True, 0)
    g.key(False, 50)
    assert g.tick(400).gestures == []  # 350 ms after the release, not yet past
    d = g.tick(401)
    assert d.gestures == [G.CANCEL] and d.replay == [True, False] and g.phase == Phase.IDLE


def test_replay_can_be_disabled():
    g = m(replay_taps=False)
    g.key(True, 0), g.key(False, 50)
    assert g.tick(1000).replay == []


def test_second_press_too_late_is_a_new_press():
    g = m()
    g.key(True, 0), g.key(False, 50)
    d = g.key(True, 500)  # the ticker never ran
    assert d.gestures == [G.CANCEL, G.PRESS] and d.replay == [True, False] and d.swallow
    assert g.phase == Phase.PRESSED


def test_late_hook_double_tap_measured_by_event_time():
    g = m()
    g.key(True, event_time(9000, 9000, 8500))
    g.key(False, event_time(9000, 9000, 8580))
    assert g.key(True, event_time(9000, 9000, 8700)).gestures == [G.LOCK]


def test_typing_after_a_tap_ends_the_wait_at_once():
    g = m()
    g.key(True, 0), g.key(False, 50)
    d = g.other(True, 100)
    assert d.gestures == [G.CANCEL] and d.replay == [True, False] and not d.swallow


def test_chord_with_voice_key_cancels_and_forwards():
    g = m()
    g.key(True, 0)
    d = g.other(True, 40)  # e.g. AltGr+e or Right Ctrl+C
    assert d.gestures == [G.CANCEL] and d.replay == [True] and not d.swallow
    assert g.other(False, 60) == Decision()
    d = g.key(False, 80)  # the OS saw the (replayed) press, so it must see the release
    assert not d.swallow and d.gestures == []
    assert g.key(True, 1000).gestures == [G.PRESS]  # back to normal


def test_chord_during_hold_cancels_too():
    g = m()
    g.key(True, 0)
    g.tick(300)
    d = g.other(True, 1000)
    assert d.gestures == [G.CANCEL] and d.replay == [True]
    assert not g.key(False, 1100).swallow


def test_chord_while_locked_keeps_the_lock():
    g = m()
    g.key(True, 0), g.key(False, 50), g.key(True, 200), g.key(False, 250)
    g.key(True, 1000)
    d = g.other(True, 1050)
    assert d.gestures == [] and d.replay == [True]
    assert not g.key(False, 1100).swallow and g.phase == Phase.LOCKED
    g.key(True, 2000)
    assert g.key(False, 2050).gestures == [G.FINISH]


def test_other_keys_while_idle_or_locked_hands_off_are_ignored():
    g = m()
    assert g.other(True, 0) == Decision()
    g.key(True, 0), g.key(False, 50), g.key(True, 200), g.key(False, 250)
    assert g.other(True, 300) == Decision() and g.phase == Phase.LOCKED


def test_ineligible_press_passes_through_with_its_release():
    g = m()
    d = g.key(True, 0, eligible=False)  # Win+RightAlt, Ctrl+Insert, ...
    assert not d.swallow and d.gestures == []
    for t in (30, 60):
        assert not g.key(True, t, eligible=False).swallow
    assert not g.key(False, 100).swallow
    assert g.key(True, 1000).gestures == [G.PRESS]


def test_ineligible_second_tap_does_not_lock():
    g = m()
    g.key(True, 0), g.key(False, 50)
    d = g.key(True, 200, eligible=False)
    assert d.gestures == [G.CANCEL] and not d.swallow and d.replay == [True, False]


@pytest.mark.parametrize("phase_setup", ["pressed", "held", "waiting", "locked"])
def test_escape_cancels_any_running_take(phase_setup):
    g = m()
    g.key(True, 0)
    if phase_setup == "held":
        g.tick(300)
    elif phase_setup == "waiting":
        g.key(False, 50)
    elif phase_setup == "locked":
        g.key(False, 50), g.key(True, 200), g.key(False, 250)
    d = g.other(True, 400, escape=True)
    assert d.swallow and d.gestures == [G.CANCEL] and g.phase == Phase.IDLE
    assert g.other(True, 430, escape=True).swallow  # Escape auto-repeat, no second cancel
    assert g.other(True, 460, escape=True).gestures == []
    assert g.other(False, 500, escape=True).swallow  # its release was ours too


def test_voice_key_release_after_escape_is_swallowed_silently():
    g = m()
    g.key(True, 0)
    g.tick(300)
    g.other(True, 400, escape=True), g.other(False, 450, escape=True)
    d = g.key(False, 600)
    assert d.swallow and d.gestures == []


def test_escape_when_idle_passes_through():
    g = m()
    assert g.other(True, 0, escape=True) == Decision()
    assert g.other(False, 50, escape=True) == Decision()


def test_escape_while_app_records_from_elsewhere():
    g = m()
    g.set_recording(True)
    d = g.other(True, 0, escape=True)
    assert d.swallow and d.gestures == [G.CANCEL]
    assert g.other(False, 40, escape=True).swallow
    g.set_recording(False)
    assert not g.other(True, 100, escape=True).swallow


def test_tap_finishes_a_take_started_from_the_ui():
    g = m()
    g.set_recording(True)  # tray/UI start: nothing from the key
    assert g.key(True, 0).gestures == []
    assert g.key(False, 60).gestures == [G.FINISH]
    assert g.key(True, 1000).gestures == [G.PRESS]  # external flag cleared by the finish


def test_late_recording_flag_from_a_key_take_is_not_external():
    g = m()
    g.key(True, 0)
    g.key(False, 400)  # RELEASE; the app is now transcribing
    g.set_recording(True)  # the app's (late) report for the take the key started
    assert g.key(True, 1000).gestures == [G.PRESS]  # a new take, not a "finish" tap


def test_recording_false_never_resets_a_waiting_first_tap():
    g = m()
    g.key(True, 0), g.key(False, 50)
    g.set_recording(False)  # e.g. the worker reported idle in between
    assert g.key(True, 200).gestures == [G.LOCK]


def test_recording_false_never_resets_the_take_being_held():
    g = m()
    g.key(True, 0), g.key(False, 400)  # take 1
    g.key(True, 600)                   # take 2 held
    g.set_recording(False)             # late report that take 1 is done
    assert g.key(False, 1200).gestures == [G.RELEASE]


@pytest.mark.parametrize("raw_at", ["hold", "lock"])
def test_raw_modifier_at_finish_replaces_the_finish(raw_at):
    g = m()
    if raw_at == "hold":
        g.key(True, 0)
        assert g.key(False, 500, raw=True).gestures == [G.RAW]
    else:
        g.key(True, 0), g.key(False, 50), g.key(True, 200), g.key(False, 250)
        g.key(True, 1000)
        assert g.key(False, 1050, raw=True).gestures == [G.RAW]


def test_raw_on_the_locking_tap_does_nothing():
    g = m()
    g.key(True, 0), g.key(False, 50), g.key(True, 200)
    assert g.key(False, 250, raw=True).gestures == []


def test_release_without_press_passes_through():
    g = m()
    d = g.key(False, 0)  # hook installed while the key was held
    assert not d.swallow and d.gestures == []


def test_missed_release_recovers_instead_of_repeating_forever():
    g = m()
    g.key(True, 0)
    g.tick(300)
    # The release was lost (hook dropped). The next press, long after, is a new press.
    d = g.key(True, 5000)
    assert d.gestures == [G.RELEASE, G.PRESS] and d.swallow
    assert g.key(False, 5600).gestures == [G.RELEASE]


def test_missed_release_while_locked():
    g = m()
    g.key(True, 0), g.key(False, 50), g.key(True, 200), g.key(False, 250)
    g.key(True, 1000)  # finishing tap... whose release is lost
    d = g.key(True, 9000)
    assert d.gestures == [G.FINISH, G.PRESS]


def test_next_deadline():
    g = m()
    assert g.next_deadline() is None
    g.key(True, 100)
    assert g.next_deadline() == 350
    g.key(False, 150)
    assert g.next_deadline() == 150 + 351
    g.key(True, 200)
    assert g.next_deadline() is None


def test_idle_and_involved():
    g = m()
    assert g.idle and not g.involved
    g.key(True, 0)
    assert not g.idle and g.involved
    g.key(False, 400)
    assert g.idle and not g.involved


def test_reset_keeps_settings():
    g = GestureMachine(500, 300, replay_taps=False)
    g.key(True, 0)
    g.reset()
    assert g.phase == Phase.IDLE and not g.key_down
    assert (g.double_tap_ms, g.hold_min_ms, g.replay_taps) == (500, 300, False)


def test_safe_swallow():
    assert safe_swallow(True, True, False)
    assert safe_swallow(True, True, True)       # a press may always stay hidden
    assert safe_swallow(True, False, False)
    assert not safe_swallow(True, False, True)  # the OS saw the press: never hide its release
    assert not safe_swallow(False, False, False)


def test_event_time():
    assert event_time(5000, 5000, 4700) == 4700
    assert event_time(5000, 3, 0xFFFFFFFF) == 4996  # 32-bit tick wrapped
    assert event_time(5000, 5000, 5000) == 5000
    assert event_time(100, 5000, 4000) == 100       # never before the clock began
    assert event_time(200000, 200000, 100) == 200000  # implausibly old: use now
