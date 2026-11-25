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
```

Settings for the MuJoCo run (motor preset, backend, command profile, time step) are the constants at
the top of `wigglesim_mujoco/boom_profile.py`. Example outputs are in `plot_archive/`.

## Model notes

- `wigglesim` state is `[q (4, scalar first), w (3), boom angles (3), boom rates (3)]`. It treats the
  chassis centre of mass as the reference point, so it ignores the shift of the system centre of mass
  as the booms swing. MuJoCo does not make that approximation.
- In `wigglesim`, the boom input is a hinge acceleration when `motor_dynamics` is false and a
  commanded angle when it is true.
- Motor presets in `wigglesim_mujoco/actuators/motor_presets.py` are from datasheets where a source
  is linked; values marked "Estimate" or "Assumed" are not.
