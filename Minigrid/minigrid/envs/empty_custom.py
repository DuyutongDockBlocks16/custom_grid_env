from __future__ import annotations

from minigrid.core.grid_custom import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Box
from minigrid.minigrid_env_custom import MiniGridEnvCustom


class EmptyEnvCustom(MiniGridEnvCustom):
    """
    ## Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`

    """

    def __init__(
        self,
        grid_size=8,
        max_steps: int | None = None,
        **kwargs,
    ):
        agents_pos = kwargs.pop("agents_start_pos", {})
        agents_dir = kwargs.pop("agents_start_dir", {})
        main_agent_idx = kwargs.pop("main_agent_idx", 0)
        agents_colors = kwargs.pop("agents_colors", {})

        if max_steps is None:
            max_steps = 4 * grid_size**2

        super().__init__(
            # mission_space=mission_space,
            grid_size = grid_size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            agents_pos=agents_pos,
            agents_dir=agents_dir,
            main_agent_idx=main_agent_idx,
            agents_colors=agents_colors,
            # max_of_other_agents = max_of_other_agents,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)
        self.place_obj(Box(color="green"), x = 7, y = 7)

        self.place_agents()

