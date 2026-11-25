# wigglesat

Simulation of a nanosatellite that points with three slow, tip-mass booms instead of reaction
wheels, after Tracy, Manchester & Douglas, *Ultra-Fine Pointing for Nanosatellite Telescopes With
Actuated Booms* (IEEE Aerospace 2022). The goal is to reproduce the paper's idea and then extend it
with realistic boom actuators.

There are two simulators:

| | `wigglesim/` | `wigglesim_mujoco/` |
|---|---|---|
| What | From-scratch rigid body + three hinged booms (NumPy/SciPy) | MuJoCo multibody model on [mjorbit](https://github.com/johnzhang3/mjorbit) |
| Environment | None (free rotation; magnetic field is a placeholder) | Propagated orbit; gravity gradient on, J2/drag/SRP/magnetic available |
| Actuators | Prescribed hinge acceleration, or a simple stepper torque model | Geared stepper models (`actuators/`), position or torque mode, optional compliant mount |
| Use it for | Checking the dynamics by hand; it has the unit tests | Actuator studies |

Neither has the paper's planner, estimator or feedback controller yet: boom commands are open loop.

## Setup

Python 3.12 (mjorbit does not support 3.13).

```bash
uv sync          # creates .venv, builds mjorbit from the pinned git revision (needs a C++ compiler)
```

To develop against a local mjorbit checkout instead, point `[tool.uv.sources]` in `pyproject.toml`
at it: `mjorbit = { path = "../mjorbit", editable = true }`.

## Running

```bash
# From-scratch simulator and its tests (run from inside wigglesim/)
cd wigglesim
uv run python simulation.py
uv run pytest test

# MuJoCo simulator (run from the repository root)
uv run python -m wigglesim_mujoco.boom_profile          # headless; writes plots/ and log.npz
uv run python -m wigglesim_mujoco.boom_profile_viewer   # browser viewer
uv run python wigglesim_mujoco/run_sat.py               # bare MuJoCo momentum check, no mjorbit
uv run python -m wigglesim_mujoco.motor_sizing          # actuator requirements vs. each motor preset
uv run pytest wigglesim_mujoco/test
```

Settings for the MuJoCo run (motor preset, backend, command profile, time step) are the constants at
the top of `wigglesim_mujoco/boom_profile.py`. Example outputs are in `plot_archive/`.

## The spacecraft and the actuator question

Both simulators model the same spacecraft: a 12 kg 6U bus with three 0.75 m booms, each a 30 g rod
with a 200 g tip mass (0.118 kg m^2 about its hinge). The paper gives no boom dimensions; these come
from its disturbance level (about 0.1 uN*m per axis) and the 35-minute eclipse the booms must cover
alone. `motor_sizing.py` prints the arithmetic.

The default MuJoCo run (`PROFILE = "reject"`) applies a constant disturbance torque to the bus and
commands the booms, open loop, to absorb it. It reports the bus pointing error and the jitter over
10 s windows. Results at 400 km, 300 s, rigid mounts (`plot_archive/6u_reject_*`):

| Actuator | Mode | Jitter (arcsec rms) | Winding loss, 3 motors |
|---|---|---|---|
| NMB PG15S, 102.5:1 | position, full step | 463 | 0.3 W |
| NMB PG15S, 102.5:1 | position, 256 microsteps | 4.6 | 0.3 W |
| Lin 5718L, 50:1 | position, 16 microsteps | 2.2 | 26 W |
| Zaber X-NMS17, direct | position, 64 microsteps | 171 | 24 W |
| Zaber X-NMS17, direct | torque | 0.0006 | ~0 |
| Zaber X-NMS17, direct | torque, 0.1 uN*m hinge friction | boom does not move | ~0 |
| Lin 5718L, 50:1 | torque (its detent torque is modelled) | boom does not move | ~0 |

What these say:

- An open-loop stepper sets the pointing resolution by its step size: a boom step of `d` turns the bus
  by about `d * J_boom / J_system` (0.23 to 0.39 here). None of the presets gets under an arcsecond,
  and the models are optimistic: no microstep placement error, no backlash, no friction.
- Torque control is smooth in the model only because the model has no friction and, for most presets,
  no detent data. Friction or cogging at the level of the disturbance torque (0.1 uN*m) stops the boom.
  That, not torque capacity, is the requirement to design the actuator around.
- The bus still drifts by tens of arcminutes in these runs. Most of that is gravity-gradient torque
  through the tip masses (about 0.2 uN*m, as large as the disturbance being rejected), which mjorbit
  always applies and the open-loop command does not cancel. It needs the feedback loop.

`plot_archive/large_bus_original/` holds runs of the earlier 3 m, 10 kg bus with 3 m booms.


## Model notes

- `wigglesim` state is `[q (4, scalar first), w (3), boom angles (3), boom rates (3)]`. It treats the
  chassis centre of mass as the reference point, so it ignores the shift of the system centre of mass
  as the booms swing. MuJoCo does not make that approximation.
- In `wigglesim`, the boom input is a hinge acceleration when `motor_dynamics` is false and a
  commanded angle when it is true.
- Motor presets in `wigglesim_mujoco/actuators/motor_presets.py` are from datasheets where a source
  is linked; values marked "Estimate" or "Assumed" are not.
