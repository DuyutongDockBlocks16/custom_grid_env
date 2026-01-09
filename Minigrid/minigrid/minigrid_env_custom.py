from __future__ import annotations

import hashlib
import math
from abc import abstractmethod
from typing import Any, Iterable, SupportsFloat, TypeVar, List

import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS
from minigrid.core.grid_custom import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Point, WorldObj, Agent, Box
import numpy


T = TypeVar("T")

class MiniGridEnvCustom(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        # mission_space: MissionSpace,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        agents_pos: List[tuple[numpy.int64, numpy.int64]] | None = None,
        agents_dir: List[int] | None = None,
        main_agent_idx: int = 0,
        agents_colors: List[str] | None = None,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        # Initialize mission

        # print(agents_pos)

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
            }
        )

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Environment configuration
        self.width = width
        self.height = height

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        self.see_through_walls = see_through_walls

        self.agents_initial_start_pos: List[tuple[numpy.int64, numpy.int64]] = agents_pos
        self.agents_initial_start_dir: List[int] = agents_dir
        self.agents_colors: List[str] = agents_colors
        self.agents_pos: List[tuple[numpy.int64, numpy.int64]] = []
        self.agents_dir: List[int] = []
        self.main_agent_idx: int = main_agent_idx
        # print("main_agent_idx:", self.main_agent_idx)
        self.number_of_agents: int = len(agents_pos)
        
        self.agents_observations: List[ObsType] = []

        # Current grid and mission and carrying
        self.grid = Grid(width, height)
        # self.under_grid = Grid(width, height)

        self.agent_carrying_list = [None for _ in range(len(agents_pos))]
        # self.carrying = None

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        
        # print("resetting env")
        
        super().reset(seed=seed)

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)
        
        # Item picked up, being carried, initially nothing
        self.agent_carrying_list = [None for _ in range(self.number_of_agents)]

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        self.agents_observations = self.gen_obs_list()

        obs = self.agents_observations[self.main_agent_idx]
        
        print("obs in reset:", obs)

        return obs, {}

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out: list[T] = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self) -> str:
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(
        self, x_low: int, x_high: int, y_low: int, y_high: int
    ) -> tuple[int, int]:
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )
    
    def move_obj(self, obj: WorldObj, i: int, j: int):

        self.grid.set(obj.cur_pos[0], obj.cur_pos[1], None)
        self.grid.set(i, j, obj)
        obj.cur_pos = (i, j)

    def place_agents(self):
        """
        Set the agent's starting point at an empty position in the grid
        """
        self.agents_pos: List[tuple[numpy.int64, numpy.int64]] = self.agents_initial_start_pos
        self.agents_dir: List[int] = self.agents_initial_start_dir
        # self.agents_colors: List[str] = self.agents_colors

        for i in range(self.number_of_agents):
            obj = Agent(color=self.agents_colors[i])
            x = self.agents_pos[i][0]
            y = self.agents_pos[i][1]
            obj.init_pos = (x, y)
            obj.cur_pos = (x, y)
            self.grid.set(x, y, obj)

    def dir_vec(self, agent_idx):
        """
        Get the vector pointing in the direction of the agent.
        """

        return DIR_TO_VEC[self.agents_dir[agent_idx]]

    def right_vec(self, agent_idx):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec(agent_idx)
        return np.array((-dy, dx))

    def fwd_pos(self, agent_idx):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agents_pos[agent_idx] + self.dir_vec(agent_idx)

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec(agent_idx=self.main_agent_idx)

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self, agent_idx, agent_view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        """

        agent_view_size = agent_view_size or self.agent_view_size

        # Facing right
        if self.agents_dir[agent_idx] == 0:
            topX = self.agents_pos[agent_idx][0]
            topY = self.agents_pos[agent_idx][1] - agent_view_size // 2
        # Facing down
        elif self.agents_dir[agent_idx] == 1:
            topX = self.agents_pos[agent_idx][0] - agent_view_size // 2
            topY = self.agents_pos[agent_idx][1]
        # Facing left
        elif self.agents_dir[agent_idx] == 2:
            topX = self.agents_pos[agent_idx][0] - agent_view_size + 1
            topY = self.agents_pos[agent_idx][1] - agent_view_size // 2
        # Facing up
        elif self.agents_dir[agent_idx] == 3:
            topX = self.agents_pos[agent_idx][0] - agent_view_size // 2
            topY = self.agents_pos[agent_idx][1] - agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs_list()[self.main_agent_idx]

        obs_grid, _ = Grid.decode(obs["image"])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        assert world_cell is not None

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        other_agents_actions:List[ActType] = []

        for agent_idx in range(self.number_of_agents):
            if agent_idx == self.main_agent_idx:
                self._command_main_agent(action, self.main_agent_idx)
            # else:
            #     self._command_other_agent(other_agents_actions, self.agents_observations[agent_idx], agent_idx)

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs_list()[self.main_agent_idx]
        
        print("obs in step:", obs)

        return obs, reward, terminated, truncated, {}

    def _command_main_agent(self, action, agent_idx):
        # Get the position in front of the agent
        fwd_pos = self.fwd_pos(agent_idx)
        agent_pos = self.agents_pos[agent_idx]
        # print(f"Agent {agent_idx} position: {agent_pos}, forward position: {fwd_pos}")

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        current_cell = self.grid.get(*agent_pos)

        # Rotate left
        if action == self.actions.left:
            self.agents_dir[agent_idx] -= 1
            if self.agents_dir[agent_idx] < 0:
                self.agents_dir[agent_idx] += 4

        # Rotate right
        elif action == self.actions.right:
            self.agents_dir[agent_idx] = (self.agents_dir[agent_idx] + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agents_pos[agent_idx] = tuple(fwd_pos)
                self.move_obj(current_cell, self.agents_pos[agent_idx][0], self.agents_pos[agent_idx][1])
                if self.agent_carrying_list[agent_idx] is not None:
                    self.move_obj(self.agent_carrying_list[agent_idx], self.agents_pos[agent_idx][0], self.agents_pos[agent_idx][1])
            if current_cell is not None and current_cell.type == "box":
                # terminated = True
                reward = self._reward()

        # Pick up an object
        elif action == self.actions.pickup:
            if current_cell and current_cell.can_pickup():
                if self.agent_carrying_list[agent_idx] is None:
                    self.agent_carrying_list[agent_idx] = current_cell
                    current_cell.is_picked_up = True

        # Drop an object
        elif action == self.actions.drop:
            if self.agent_carrying_list[agent_idx] is not None:
                self.grid.set(agent_pos[0], agent_pos[1], self.agent_carrying_list[agent_idx])
                self.agent_carrying_list[agent_idx] = None
                current_cell.is_picked_up = False

        # # Toggle/activate an object
        # elif action == self.actions.toggle:
        #     if current_cell:
        #         current_cell.toggle(self, agent_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")
        
        # print("agents_dir:", self.agents_dir[self.main_agent_idx])
        
    def _command_other_agent(self):
        pass

    def gen_obs_grid(self, agent_idx, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_idx, agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.agents_dir[agent_idx] + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        # agent_pos = grid.width // 2, grid.height - 1
        # if self.agent_carrying_list[agent_idx] is not None:
        #     grid.set(*agent_pos, self.agent_carrying_list[agent_idx])
        # else:
        #     grid.set(*agent_pos, Agent())
        # grid.set(*agent_pos, Agent(self.agents_colors[agent_idx]))

        return grid, vis_mask

    def gen_obs_list(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        
        self.agents_observations = []
        
        self.box_positions = self.grid.search_unpicked_up_box()
    
        for agent_idx in range(self.number_of_agents):
            grid, vis_mask = self.gen_obs_grid(agent_idx)
            image = grid.encode(vis_mask)

            # print("agents_dir:", self.agents_dir[agent_idx])

            nearest_uncarried_box_pos = None
            # check all elements in the box_positions list and find the nearest one to the agent
            if self.box_positions:
                min_distance = float("inf")
                for box_pos in self.box_positions:
                    distance = math.sqrt(
                        (box_pos[0] - self.agents_pos[agent_idx][0]) ** 2
                        + (box_pos[1] - self.agents_pos[agent_idx][1]) ** 2
                    )
                    if distance < min_distance:
                        min_distance = distance
                        nearest_uncarried_box_pos = box_pos
                
            obs = {
                    "image": image,
                    "direction": self.agents_dir[agent_idx],
                    "agent_pos": self.agents_pos[agent_idx],
                    "nearest_uncarried_box_pos": nearest_uncarried_box_pos
                }

            self.agents_observations.append(obs)

        return self.agents_observations

    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.gen_obs_grid(self.main_agent_idx)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
        )

        return img

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        
        agent_idx = self.main_agent_idx
        
        _, vis_mask = self.gen_obs_grid(agent_idx)

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec(agent_idx)
        r_vec = self.right_vec(agent_idx)
        top_left = (
            self.agents_pos[agent_idx]
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # print("Agents positions:", self.agents_pos)
        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agents_pos,
            self.agents_dir,
            self.agents_colors,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            # text = self.mission
            text = "1"
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            pygame.quit()

    def place_box(
        self,
        obj: WorldObj | None,
        x: int = None,
        y: int = None
    ):
        obj.init_pos = (x, y)
        obj.cur_pos = (x, y)
        self.grid.set(x, y, obj)

