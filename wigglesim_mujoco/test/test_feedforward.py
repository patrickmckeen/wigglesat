"""Run from the repository root: pytest wigglesim_mujoco/test"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from wigglesim_mujoco.boom_profile import ARCSEC, rejection_feedforward, smoothstep_double_integral

XML_DIR = Path(__file__).resolve().parents[1]
TORQUE = np.array([0.10, -0.15, 0.08]) * 1e-6
T_END, DT = 60.0, 0.001


def bus_rotation_arcsec(xml, shaft_suffix, use_feedforward):
    """Bus rotation after T_END with a constant body torque, in plain MuJoCo (no orbit, no gravity)."""
    model = mujoco.MjModel.from_xml_path(str(XML_DIR / xml))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    shaft = np.array([model.joint(f"{a}_{shaft_suffix}_joint").dofadr[0] for a in "xyz"])
    accel, torque = rejection_feedforward(model, shaft, TORQUE)
    for _ in range(int(round(T_END / DT))):
        data.qfrc_applied[3:6] = TORQUE
        if use_feedforward:
            data.ctrl[:] = torque
        mujoco.mj_step(model, data)
    return 2.0 * np.linalg.norm(data.qpos[4:7]) * ARCSEC, accel, data.qpos[model.jnt_qposadr[model.dof_jntid[shaft]]]


@pytest.mark.parametrize("xml, suffix", [("sat_booms.xml", "arm"), ("sat_booms_compliant.xml", "motor")])
def test_feedforward_hinge_torques_hold_the_bus_against_a_body_torque(xml, suffix):
    free, _, _ = bus_rotation_arcsec(xml, suffix, use_feedforward=False)
    held, accel, shaft_angle = bus_rotation_arcsec(xml, suffix, use_feedforward=True)
    assert free > 100.0                 # the torque alone turns the bus by arcminutes
    assert held < 1e-3 * free           # the booms absorb it (what is left is the linearisation
                                        # and, with compliant mounts, their settling)
    # and the booms travel the commanded parabola
    np.testing.assert_allclose(shaft_angle, 0.5 * accel * T_END ** 2, rtol=2e-2)


def test_smoothstep_double_integral_matches_quadrature():
    from wigglesim_mujoco.boom_profile import smoothstep
    ramp, t = 5.0, np.linspace(0.0, 12.0, 120001)
    accel = np.array([smoothstep(x / ramp) for x in t])
    dt = t[1] - t[0]
    rate = np.concatenate([[0.0], np.cumsum(0.5 * (accel[1:] + accel[:-1]) * dt)])
    angle = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)])
    for tk in (2.0, 5.0, 12.0):
        k = int(round(tk / dt))
        assert smoothstep_double_integral(tk, ramp) == pytest.approx(angle[k], rel=1e-6)
