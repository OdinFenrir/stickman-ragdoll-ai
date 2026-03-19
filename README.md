# Stickman Ragdoll AI

Verlet-integration ragdoll physics engine built for AI training. A stickman falls, tumbles, and collapses under gravity — joint states are exposed as a normalized input vector for reinforcement learning.

## Quick Start

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install pygame

python main.py
```

Or use the included launcher:

```bash
run_game.bat
```

## Project Structure

```
.
├── src/
│   ├── ragdoll.py      # Physics engine, constraints, rendering
│   └── __init__.py     # Public API
├── main.py             # Game entry point
├── requirements.txt
└── run_game.bat
```

## Physics

- **Integration**: Verlet (position-based, stable)
- **Constraints**: Distance sticks + angle limits, solved iteratively
- **Solver**: 18 iterations/frame for stable joint enforcement
- **Damping**: 0.999 air resistance
- **Floor friction**: 0.82 coefficient

## AI Integration

`Ragdoll.get_joint_input_vector()` exposes per-joint state:

```
[px, py, vx, vy] * num_joints
```

All values are normalized. Feed into a neural network to learn motor control from ragdoll physics.
