from dataclasses import dataclass

@dataclass
class State:
    fps: int = 0
    num_bodies: int = 0
    num_joints: int = 0
    num_proxies: int = 0
    num_contacts: int = 0

    cars = None
    car_pos: float = 0.0

    draw_aabb: bool = False
    draw_obb: bool = False
    draw_joints: bool = False
    draw_pairs: bool = False
    draw_coms: bool = False

    # Sim Controls
    paused: bool = False
    reset: bool = False

    # Genetic Algorithms
    generation = 0
