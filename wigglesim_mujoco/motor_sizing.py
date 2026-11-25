"""python -m wigglesim_mujoco.motor_sizing
run from wigglesat root

Back-of-envelope requirements on the boom actuator, and how each motor preset compares.
No simulation: mass properties come from sat_booms.xml, the rest is arithmetic.

The numbers to keep in mind (paper regime, Tracy et al. 2022):
  * the booms only have to produce the disturbance torque, ~0.1 uN*m, by accelerating at
    tau / J_boom. Anything in the drive train larger than that (cogging, friction) decides
    what the boom does instead of the command.
  * moving a boom by d_theta turns the bus by about -(J_boom / J_bus) * d_theta, however the
    move is made. So the smallest boom move the actuator can make sets the pointing resolution.
"""
from pathlib import Path

import mujoco
import numpy as np

from wigglesim_mujoco.actuators.motor_presets import MOTOR_PRESETS

ARCSEC = np.degrees(1.0) * 3600.0     # arcseconds per radian
DIST_TORQUE = 0.1e-6                  # N*m, per axis (paper Fig. 6: 0.05-0.17 uN*m)
ECLIPSE_S = 35.0 * 60.0               # magnetorquers are off for this long (paper Sec. 4)
POINTING_GOAL_ARCSEC = 0.1            # body pointing step we are willing to accept
ALT_KM = 400.0


def mass_properties():
    """(bus inertia about each hinge axis incl. the stowed booms, boom inertia about its hinge)."""
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("sat_booms.xml")))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    hinges = [model.joint(f"{a}_arm_joint").dofadr[0] for a in "xyz"]
    return np.diag(M)[3:6].copy(), np.diag(M)[hinges].copy(), model


def main():
    j_sys, j_boom, model = mass_properties()
    bus = model.body("sc_bus").id
    boom_mass = model.body_subtreemass[bus] - model.body_mass[bus]
    ratio = j_boom / j_sys
    print(f"bus {model.body_mass[bus]:g} kg, booms {boom_mass:.2f} kg total "
          f"({100 * boom_mass / model.body_subtreemass[bus]:.1f}% of the spacecraft)")
    print(f"system inertia about x/y/z (booms at zero): {np.round(j_sys, 3)} kg m^2")
    print(f"boom inertia about its hinge:               {np.round(j_boom, 3)} kg m^2")
    print(f"bus rotation per unit boom rotation (J_boom / J_sys): {np.round(ratio, 2)}\n")

    alpha = DIST_TORQUE / j_boom.mean()
    rate = alpha * ECLIPSE_S
    travel = 0.5 * alpha * ECLIPSE_S ** 2
    print(f"Absorbing a constant {DIST_TORQUE * 1e6:g} uN*m for a {ECLIPSE_S / 60:g} min eclipse:")
    print(f"  boom acceleration {alpha:.2e} rad/s^2, final rate {rate:.2e} rad/s "
          f"({np.degrees(rate) * 60:.2f} deg/min), travel {np.degrees(travel):.0f} deg")
    print("  (travel scales as 1 / J_boom: this is what sizes the boom length and tip mass)")
    n2 = 3.986004418e14 / ((6378.137 + ALT_KM) * 1e3) ** 3
    print(f"Gravity-gradient torque from the booms themselves, ~1.5 n^2 J_boom: "
          f"{1.5 * n2 * j_boom.mean() * 1e6:.2f} uN*m, changing with boom angle")
    step_goal = POINTING_GOAL_ARCSEC / ratio.max()
    print(f"For {POINTING_GOAL_ARCSEC:g} arcsec body steps the boom must be placed to "
          f"{step_goal:.3f} arcsec; friction and cogging must be well under {DIST_TORQUE * 1e6:g} uN*m.\n")

    cols = ("motor", "mass kg", "ustep \"", "bus step \"", "cog/tau_d", "cog Hz", "backlash->bus \"", "hold W")
    print("{:<26}{:>8}{:>10}{:>12}{:>11}{:>9}{:>17}{:>8}".format(*cols))
    for m in MOTOR_PRESETS.values():
        cog = m.cogging_output_amplitude / DIST_TORQUE
        print("{:<26}{:>8.2f}{:>10.3g}{:>12.3g}{:>11.2g}{:>9.2g}{:>17.3g}{:>8.2g}".format(
            m.name, 3 * m.total_mass, m.microstep_output_arcsec,
            m.microstep_output_arcsec * ratio.max(), cog, m.cogging_freq_hz(rate),
            m.backlash_arcmin * 60.0 * ratio.max(), 3 * m.phase_resistance * m.run_current ** 2))
    print("""
  mass kg          three motors + gearboxes
  ustep "          commanded output angle per microstep at the preset setting (no placement error)
  bus step "       resulting bus rotation, worst axis
  cog/tau_d        output cogging amplitude over the disturbance torque (0 = no detent data)
  cog Hz           cogging frequency at the end-of-eclipse boom rate
  backlash->bus "  bus rotation if the gearbox backlash is taken up (not modelled in the sims)
  hold W           winding loss of three motors in open-loop position mode""")


if __name__ == "__main__":
    main()
