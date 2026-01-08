from __future__ import annotations
from typing import List
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Minigrid'))

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid_custom import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Box
from minigrid.manual_control import ManualControl
# from minigrid.minigrid_env import MiniGridEnv
from minigrid.minigrid_env_custom import MiniGridEnvCustom
import inspect

agents_start_pos: List[tuple[int, int]] = [
    (2, 2), # Agent 0
    (2, 4), # Agent 1
    # (2, 2), # Agent 2
    # (2, 2), # Agent 3
    # (2, 2), # Agent 4

]

agents_start_dir: List[int] = [
    0, # Agent 0
    0, # Agent 1
    # 0, # Agent 2
    # 0, # Agent 3
    # 0, # Agent 4
]

agents_colors: List[str] = [
    "red",
    "green",
    # "blue",
    # "purple",
    # "yellow"
]

main_agent_idx = 1


class SimpleEnv(MiniGridEnvCustom):
    def __init__(
        self,
        grid_size=10,
        agents_start_pos=agents_start_pos,
        agents_start_dir=agents_start_dir,
        main_agent_idx=main_agent_idx,
        agents_colors=agents_colors,
        max_steps: int | None = None,
        **kwargs,
    ):

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * grid_size**2
            
        super().__init__(
            # mission_space=mission_space,
            grid_size=grid_size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            agents_pos=agents_start_pos,
            agents_dir=agents_start_dir,
            main_agent_idx=main_agent_idx,
            agents_colors=agents_colors,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)
        self.place_obj(Box(color="green"), x = 3, y = 2)

        self.place_agents()


def main():
    env = SimpleEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    
if __name__ == "__main__":
    main()