import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tvc_sim.rocket import Rocket, RocketInputs, RocketParams, RocketState

__all__ = ["Rocket", "RocketInputs", "RocketParams", "RocketState"]
