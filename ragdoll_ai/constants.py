import numpy as np

RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
FPS = 60
PHYSICS_DT = 1.0 / FPS
MAX_EPISODE_STEPS = 1000
FLOOR_Y = RENDER_HEIGHT - 80
SOLVER_ITERATIONS = 18
GRAVITY = 1800.0
AIR_DAMPING = 0.999
FLOOR_FRICTION = 0.82
WALL_BOUNCE = 0.5

BG = np.array([18, 18, 24], dtype=np.uint8)
GRID_COLOR = np.array([34, 34, 44], dtype=np.uint8)
TEXT_COLOR = np.array([230, 230, 235], dtype=np.uint8)
MUTED_COLOR = np.array([150, 150, 160], dtype=np.uint8)
BONE_COLOR = np.array([220, 220, 225], dtype=np.uint8)
JOINT_COLOR = np.array([255, 180, 90], dtype=np.uint8)
HEAD_COLOR = np.array([235, 235, 240], dtype=np.uint8)
FLOOR_COLOR = np.array([80, 90, 110], dtype=np.uint8)
PELVIS_COLOR = np.array([255, 110, 110], dtype=np.uint8)

JOINT_NAMES = [
    "head", "neck", "spine_upper", "spine_lower", "pelvis",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_hand", "r_hand",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
    "l_foot", "r_foot",
    "l_toe", "r_toe",
]

NUM_JOINTS = len(JOINT_NAMES)
OBS_DIM = NUM_JOINTS * 4
ACTION_DIM = NUM_JOINTS
