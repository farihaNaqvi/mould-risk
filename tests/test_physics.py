import numpy as np
import pytest

from mould_risk.physics import psat, rh_crit, surface_rh, surface_temperature, vapour_pressure, vtt_index


def test_psat_matches_reference_values():
    assert psat(20.0) == pytest.approx(23.37, abs=0.1)
    assert psat(0.0) == pytest.approx(6.11, abs=0.01)


def test_surface_rh_equals_air_rh_when_wall_is_at_room_temperature():
    vp = vapour_pressure(20.0, 60.0)
    assert surface_rh(20.0, vp, 5.0, f_rsi=1.0) == pytest.approx(60.0)


def test_cold_wall_raises_surface_rh():
    vp = vapour_pressure(18.0, 65.0)
    assert surface_temperature(18.0, 2.0, 0.75) == pytest.approx(14.0)
    assert surface_rh(18.0, vp, 2.0, 0.75) > 80.0


def test_rh_crit_is_80_above_20c_and_rises_when_cold():
    assert rh_crit(25.0) == 80.0
    assert rh_crit(5.0) > 85.0


def test_vtt_grows_in_wet_conditions_and_stays_zero_when_dry():
    hours = 24 * 60
    wet = vtt_index(np.full(hours, 20.0), np.full(hours, 97.0))
    dry = vtt_index(np.full(hours, 20.0), np.full(hours, 60.0))
    assert wet[-1] > 1.0
    assert np.all(np.diff(wet) >= 0)
    assert np.all(dry == 0)


def test_vtt_declines_after_wet_period_and_holds_through_gaps():
    t = np.full(24 * 90, 20.0)
    rh = np.r_[np.full(24 * 60, 97.0), np.full(24 * 30, 50.0)]
    m = vtt_index(t, rh)
    peak = m[24 * 60 - 1]
    assert m[-1] < peak
    gap = vtt_index(np.r_[np.full(240, 20.0), np.full(48, np.nan)], np.r_[np.full(240, 97.0), np.full(48, np.nan)])
    assert gap[-1] == gap[239]
