# Stickman Ragdoll AI

Verlet-integration ragdoll physics engine built for reinforcement learning training.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

python main.py         # sandbox / visual demo
python debug_viewer.py # physics microscope
```

## Scripts

```bash
python scripts/check_env.py    # Gymnasium interface validation
python scripts/smoke_test.py   # 200-step smoke test
pytest tests/                  # unit tests
```

## Training

```bash
python train.py   # trains SAC, saves to models/sac_final
python eval.py    # evaluate trained model
```

## Architecture

```
ragdoll_ai/
├── body.py         # ragdoll definition (points, sticks, angles)
├── physics.py      # verlet integration + constraint solver
├── env.py          # Gymnasium environment
├── observations.py # state vector (joint pos/vel)
├── actions.py     # torque application from policy output
├── rewards.py     # reward signal
├── termination.py # episode end conditions
├── renderer.py    # pygame drawing
└── constants.py  # all fixed values
```

## Physics

- Verlet integration (position-based, stable at 60 fps)
- 22 point masses with distance sticks + angle constraints
- 18 solver iterations/frame
- Floor friction (0.82), air damping (0.999)
- Joint angle limits enforced per-limb

## AI Interface

Observations: `(22 joints × 4) = 88 floats` — normalized x, y, vx, vy per joint

Actions: `(22 joints) = 22 floats` — bounded torques scaled to [-1, 1]

Algorithm: SAC with Stable-Baselines3
