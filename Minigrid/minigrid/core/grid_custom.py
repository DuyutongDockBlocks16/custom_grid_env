from __future__ import annotations

import math
from typing import Any, Callable, List

import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS
from minigrid.core.world_object import Wall, WorldObj
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(self, width: int, height: int):
        assert width >= 3
        assert height >= 3

        self.width: int = width
        self.height: int = height

        self.grid: list[WorldObj | None] = [None] * (width * height)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other: Grid) -> bool:
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other: Grid) -> bool:
        return not self == other

    def copy(self) -> Grid:
        from copy import deepcopy

        return deepcopy(self)

    def set(self, i: int, j: int, v: WorldObj | None):
        assert (
            0 <= i < self.width
        ), f"column index {i} outside of grid of width {self.width}"
        assert (
            0 <= j < self.height
        ), f"row index {j} outside of grid of height {self.height}"
        self.grid[j * self.width + i] = v

    def get(self, i: int, j: int) -> WorldObj | None:
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        assert self.grid is not None
        return self.grid[j * self.width + i]

    def horz_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x: int, y: int, w: int, h: int):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self) -> Grid:
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX: int, topY: int, width: int, height: int) -> Grid:
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if 0 <= x < self.width and 0 <= y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None,
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size: int,
        agents_pos: List[tuple[int, int]],
        agents_dir: List[int],
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        
        # Create different colors for agents
        agent_colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green  
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
            (255, 192, 203),  # Pink
            (165, 42, 42),    # Brown
            (128, 128, 128),  # Gray
        ]

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                
                agents_here = []
                for agent_idx, agent_pos in enumerate(agents_pos):
                    if np.array_equal(agent_pos, (i, j)):
                        agents_here.append((agent_idx, agents_dir[agent_idx]))

                if agents_here:
                    print(f"Rendering {len(agents_here)} agent(s) at position ({i}, {j}): {agents_here}")
            
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                if agents_here:
                    tile_img = self.render_tile_with_multiple_agents(
                        cell, 
                        agents_here, 
                        agent_colors,
                        highlight_mask[i, j], 
                        tile_size
                    )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask: np.ndarray | None = None) -> np.ndarray:
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                assert vis_mask is not None
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX["empty"]
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array
    
    def render_tile_with_multiple_agents(
    self,
    obj: WorldObj | None,
    agents_here: List[tuple[int, int]],
    agent_colors: List[tuple[int, int, int]],
    highlight: bool,
    tile_size: int,
    subdivs: int = 3,
    ) -> np.ndarray:

        
        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        for i, (agent_idx, agent_dir) in enumerate(agents_here):
            agent_color = agent_colors[agent_idx % len(agent_colors)]
            
            offset_x, offset_y = self.get_agent_offset(i, len(agents_here))
            
            tri_fn = point_in_triangle(
                (0.12 + offset_x, 0.19 + offset_y),
                (0.87 + offset_x, 0.50 + offset_y),
                (0.12 + offset_x, 0.81 + offset_y),
            )
            
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, agent_color)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        return img

    def get_agent_offset(self, agent_index: int, total_agents: int) -> tuple[float, float]:
        """
        Generate different position offsets for multiple agents to avoid overlap
        """
        if total_agents == 1:
            return (0.0, 0.0)
        elif total_agents == 2:
            # Two agents: left-right arrangement
            offsets = [(-0.15, 0.0), (0.15, 0.0)]
        elif total_agents == 3:
            # Three agents: triangle arrangement
            offsets = [(0.0, -0.1), (-0.15, 0.1), (0.15, 0.1)]
        elif total_agents == 4:
            # Four agents: corner arrangement
            offsets = [(-0.15, -0.15), (0.15, -0.15), (-0.15, 0.15), (0.15, 0.15)]
        else:
            # More agents: circular arrangement
            angle = 2 * math.pi * agent_index / total_agents
            radius = 0.15
            offset_x = radius * math.cos(angle)
            offset_y = radius * math.sin(angle)
            return (offset_x, offset_y)
        
        return offsets[agent_index] if agent_index < len(offsets) else (0.0, 0.0)

    @staticmethod
    def decode(array: np.ndarray) -> tuple[Grid, np.ndarray]:
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = type_idx != OBJECT_TO_IDX["unseen"]

        return grid, vis_mask

    def process_vis(self, agent_pos: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, self.height)):
            for i in range(0, self.width - 1):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, self.width)):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, self.height):
            for i in range(0, self.width):
                if not mask[i, j]:
                    self.set(i, j, None)

        return mask
