import base64
import json
import os
import random
import sys

from io import BytesIO
import copy

import numpy as np
import requests
import datetime
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel
from PIL import ImageFont
import matplotlib.font_manager as fm
import time
import re

font_size = 16
font_path = fm.findfont(fm.FontProperties(family="Arial"))
font = ImageFont.truetype(font_path, size=font_size)


def get_dataset(split="train"):
    base_path = os.path.dirname(os.path.abspath(__file__))
    if split == "train":
        tasks = base_path + "/arc2024/arc-agi_training_challenges.json"
        solutions = base_path + "/arc2024/arc-agi_training_solutions.json"
    else:
        tasks = base_path + "/arc2024/arc-agi_evaluation_challenges.json"
        solutions = base_path + "/arc2024/arc-agi_evaluation_solutions.json"

    with open(tasks, "r") as f:
        tasks = json.load(f)
    with open(solutions, "r") as f:
        solutions = json.load(f)

    tasks_keys = set(tasks.keys())
    solutions_keys = set(solutions.keys())
    assert (
        tasks_keys == solutions_keys
    ), "Error: The keys of tasks and solutions do not match."
    for key in solutions.keys():
        for i in range(len(solutions[key])):
            tasks[key]["test"][i]["output"] = solutions[key][i]
    result = []
    for key, task in tasks.items():
        # train_two = random.sample(task["train"], 2)
        # test_one = random.sample(task["test"], 1)
        train_two = task["train"][:3] if len(task["train"]) >= 3 else task["train"]
        # test_one = task["test"]
        test_one = task["test"][:1] if len(task["test"]) >= 1 else task["test"]
        sample = []
        sample.extend(train_two)
        sample.extend(test_one)

        x = [np.array(t["input"]) for t in sample]
        y = [np.array(t["output"]) for t in sample]

        max_shape = max(max(arr.shape) for arr in x + y)
        if max_shape > 30:
            print(
                f"Skipping task with key '{key}' because max_shape = {max_shape} > 30"
            )
            continue  # 跳过当前任务

        rotate_times = np.random.randint(0, 4)  # [0,3]
        flip = np.random.randint(0, 2)  # [0,1]
        permutation = np.random.permutation(10)
        permutation = np.append(permutation, 10)

        def aug(matrix):
            matrix = np.rot90(matrix, rotate_times)
            matrix = np.flipud(matrix) if flip else matrix
            return matrix

        def permutate(matrix):
            mapping = {i: permutation[i] for i in range(11)}
            vectorized_map = np.vectorize(mapping.get)
            matrix = vectorized_map(matrix)
            return matrix

        # x = [aug(matrix) for matrix in x]
        # y = [aug(matrix) for matrix in y]
        # x = [permutate(matrix) for matrix in x]
        # y = [permutate(matrix) for matrix in y]

        result.append({"key": key, "input": x, "output": y})
    return result


def pad_x(matrix, pad_to_size):
    if False:
        pad_index_0 = random.randint(0, pad_to_size - matrix.shape[0])
        pad_index_1 = random.randint(0, pad_to_size - matrix.shape[1])
    else:
        pad_index_0 = 0
        pad_index_1 = 0

    return np.pad(
        matrix,
        (
            (pad_index_0, pad_to_size - matrix.shape[0] - pad_index_0),
            (pad_index_1, pad_to_size - matrix.shape[1] - pad_index_1),
        ),
        mode="constant",
        constant_values=(10, 10),
    )


def pad_y(matrix, pad_to_size):
    pad_index_0 = 0
    pad_index_1 = 0
    return np.pad(
        matrix,
        (
            (pad_index_0, pad_to_size - matrix.shape[0] - pad_index_0),
            (pad_index_1, pad_to_size - matrix.shape[1] - pad_index_1),
        ),
        mode="constant",
        constant_values=(10, 10),
    )


class BoardCore:
    def __init__(self):
        self.dataset = get_dataset()
        self.pixel_per_element = 15
        self.max_grid_size = 30
        self.grid_separator_size = 1
        self.toolbar_height = 1
        self.dynamic = False
        if self.dynamic:
            self.super_grid_width = 96
            self.super_grid_height = 48
        else:
            self.super_grid_width = 128
            self.super_grid_height = 64
        self.width = self.super_grid_width * self.pixel_per_element
        self.height = self.super_grid_height * self.pixel_per_element
        self.actions_shortcuts_map = {
            "left": "Left",
            "right": "Right",
            "up": "Up",
            "down": "Down",
            "pill_foreground": "p",
            "pill_background": "P",
            "fill": "f",
            "flood_fill": "F",
            "select": "1",
            "region_select": "2",
            "diagonal_select": "3",
            "unselect": "u",
            "unselect_all": "U",
            "undo": "z",
            "move_left": "a",
            "move_right": "d",
            "move_up": "w",
            "move_down": "s",
            "rotate": "r",
            "flip": "v",
            "copy": "c",
            "insert_left": "A",
            "insert_right": "D",
            "insert_up": "W",
            "insert_down": "S",
        }

        self.color_map = {
            0: "white",
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "grey",
            6: "darkred",
            7: "lightblue",
            8: "darkblue",
            9: "brown",
            10: "lightgrey",
            11: "darkgrey",
        }
        self.reset(task_id=0)

    def reset(self, task_id=None):
        if task_id is None:
            self.task_id = random.randint(0, len(self.dataset) - 1)
        else:
            self.task_id = task_id
        print("task_id:" + str(self.task_id))
        self.task_key = self.dataset[self.task_id]["key"]
        self.history = [
            {"action": "init", "task_id": self.task_id, "task_key": self.task_key}
        ]
        self.static_image = Image.new("RGB", (self.width, self.height), "lightgrey")
        self.static_layer = ImageDraw.Draw(self.static_image)
        self.dynamic_image = Image.new(
            "RGBA", (self.width, self.height), (255, 255, 255, 0)
        )
        self.dynamic_layer = ImageDraw.Draw(self.dynamic_image)
        self.state = {
            # static
            "foreground_color_region": [0, 0, 1, 1],
            "background_color_region": [1, 0, 2, 1],
            "super_grid": None,
            "grid_regions": [],
            "result_region_base": None,
            # dynamic
            "foreground_color": "red",
            "background_color": "white",
            "position": [0, 0],
            "result_grid": None,
            "selected_objects": [],
        }
        self.pre_state = self.state.copy()
        self.task_sample_size = len(self.dataset[self.task_id]["input"])
        test_input_shape = self.dataset[self.task_id]["input"][
            self.task_sample_size - 1
        ].shape
        self.state["result_grid"] = np.zeros(test_input_shape, dtype=np.int8)

        self.init_super_grid()
        self.draw_static()
        self.draw_dynamic()

    def draw_single_grid(self, grid: np.array, base=[0, 0], static=True):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color = self.color_map[grid[i][j]]
                bbox = [
                    (base[0] + i) * self.pixel_per_element,
                    (base[1] + j) * self.pixel_per_element,
                    (base[0] + (i + 1)) * self.pixel_per_element,
                    (base[1] + (j + 1)) * self.pixel_per_element,
                ]
                #  50% 灰度 (128, 128, 128)
                if static:
                    self.static_layer.rectangle(
                        bbox, fill=color, outline=(128, 128, 128), width=1
                    )
                else:
                    self.dynamic_layer.rectangle(
                        bbox, fill=color, outline=(128, 128, 128), width=1
                    )

    def init_super_grid(self):
        if self.dynamic:
            super_grid_width = 0
            for task_sample_index in range(self.task_sample_size - 1):
                input_grid = self.dataset[self.task_id]["input"][task_sample_index]
                output_grid = self.dataset[self.task_id]["output"][task_sample_index]
                max_width = max(input_grid.shape[0], output_grid.shape[0])
                super_grid_width += max_width
            super_grid_width = (
                super_grid_width
                + self.max_grid_size
                + (self.task_sample_size - 1) * self.grid_separator_size
            )
            max_input_height = 0
            for input in self.dataset[self.task_id]["input"]:
                if input.shape[1] > max_input_height:
                    max_input_height = input.shape[1]
            super_grid_height = (
                max_input_height
                + self.max_grid_size
                + self.toolbar_height
                + 2 * self.grid_separator_size
            )
        else:
            super_grid_width = (
                self.max_grid_size * self.task_sample_size
                + (self.task_sample_size - 1) * self.grid_separator_size
            )
            super_grid_height = (
                self.max_grid_size * 2
                + self.toolbar_height
                + 2 * self.grid_separator_size
            )
        assert (
            super_grid_width <= self.super_grid_width
        ), f"{self.dataset[self.task_id]['key']} super_grid_width ({super_grid_width}) must be less than {self.super_grid_width}"
        assert (
            super_grid_height <= self.super_grid_width
        ), f"{self.dataset[self.task_id]['key']} super_grid_height ({super_grid_height}) must be less than {self.super_grid_width}"
        self.state["super_grid"] = np.full(
            (self.super_grid_width, self.super_grid_height), 11, dtype=np.int8
        )
        self.state["super_grid"][:, 0] = 10
        for task_sample_index in range(self.task_sample_size):
            input_grid = self.dataset[self.task_id]["input"][task_sample_index]
            output_grid = self.dataset[self.task_id]["output"][task_sample_index]
            x_start = task_sample_index * (
                self.max_grid_size + self.grid_separator_size
            )
            input_y_start = self.toolbar_height + self.grid_separator_size
            output_y_start = (
                self.toolbar_height
                + self.grid_separator_size
                + (self.max_grid_size + self.grid_separator_size)
            )
            self.state["grid_regions"].append(
                [
                    x_start,
                    input_y_start,
                    x_start + input_grid.shape[0],
                    input_y_start + input_grid.shape[1],
                ]
            )
            if task_sample_index < self.task_sample_size - 1:
                self.state["grid_regions"].append(
                    [
                        x_start,
                        output_y_start,
                        x_start + output_grid.shape[0],
                        output_y_start + output_grid.shape[1],
                    ]
                )
            else:
                self.state["result_region_base"] = [x_start, output_y_start]

            input_grid = pad_x(input_grid, self.max_grid_size)
            input_grid = pad_y(input_grid, self.max_grid_size)
            output_grid = pad_x(output_grid, self.max_grid_size)
            output_grid = pad_y(output_grid, self.max_grid_size)
            self.state["super_grid"][
                x_start : x_start + self.max_grid_size,
                input_y_start : input_y_start + self.max_grid_size,
            ] = input_grid
            if task_sample_index < self.task_sample_size - 1:
                self.state["super_grid"][
                    x_start : x_start + self.max_grid_size,
                    output_y_start : output_y_start + self.max_grid_size,
                ] = output_grid
            else:
                self.state["super_grid"][
                    x_start : x_start + self.max_grid_size,
                    output_y_start : output_y_start + self.max_grid_size,
                ] = np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.int8)

    def draw_static(self):
        self.draw_single_grid(self.state["super_grid"])

    def draw_dynamic(self):
        # Clear
        self.dynamic_layer.rectangle(
            [0, 0, self.width, self.height], fill=(255, 255, 255, 0)
        )
        # Draw result grid
        self.draw_single_grid(
            self.state["result_grid"],
            base=self.state["result_region_base"],
            static=False,
        )

        # Draw current foreground and background colors
        fg_bbox = [
            self.state["foreground_color_region"][0] * self.pixel_per_element,
            self.state["foreground_color_region"][1] * self.pixel_per_element,
            self.state["foreground_color_region"][2] * self.pixel_per_element,
            self.state["foreground_color_region"][3] * self.pixel_per_element,
        ]
        bg_bbox = [
            self.state["background_color_region"][0] * self.pixel_per_element,
            self.state["background_color_region"][1] * self.pixel_per_element,
            self.state["background_color_region"][2] * self.pixel_per_element,
            self.state["background_color_region"][3] * self.pixel_per_element,
        ]
        self.dynamic_layer.rectangle(fg_bbox, fill=self.state["foreground_color"])
        self.dynamic_layer.rectangle(bg_bbox, fill=self.state["background_color"])

        # Draw current position with thicker outline
        pos_bbox = [
            self.state["position"][0] * self.pixel_per_element,
            self.state["position"][1] * self.pixel_per_element,
            (self.state["position"][0] + 1) * self.pixel_per_element,
            (self.state["position"][1] + 1) * self.pixel_per_element,
        ]
        self.dynamic_layer.rectangle(pos_bbox, outline="black", width=3)

        # Draw selected objects with diagonal hatch pattern
        for region in self.state["selected_objects"]:
            x0 = region[0] * self.pixel_per_element
            y0 = region[1] * self.pixel_per_element
            x1 = (region[0] + 1) * self.pixel_per_element
            y1 = (region[1] + 1) * self.pixel_per_element

            self.dynamic_layer.line(
                [x0, y0, x1, y1],
                fill=(0, 0, 0, 128),
                width=2,
            )

            self.dynamic_layer.line(
                [
                    x0,
                    y0 + self.pixel_per_element / 2,
                    x1 - +self.pixel_per_element / 2,
                    y1,
                ],
                fill=(0, 0, 0, 128),
                width=2,
            )
            self.dynamic_layer.line(
                [
                    x0 + self.pixel_per_element / 2,
                    y0,
                    x1,
                    y1 - self.pixel_per_element / 2,
                ],
                fill=(0, 0, 0, 128),
                width=2,
            )

    def handle_mouse_action(self, press: bool, x: float, y: float):
        x = int(x)
        y = int(y)
        # Convert pixel coordinates to grid coordinates
        if press:
            self.history.append(
                {
                    "action": "mouse",
                    "press": press,
                    "x": x,
                    "y": y,
                }
            )
            self.pre_state = copy.deepcopy(self.state)
            grid_x = int(x // self.pixel_per_element)
            grid_y = int(y // self.pixel_per_element)
            if (
                0 <= grid_x < self.state["super_grid"].shape[0]
                and 0 <= grid_y < self.state["super_grid"].shape[1]
            ):
                self.state["position"] = [grid_x, grid_y]
        self.draw_dynamic()

    def is_in_grid_regions(self, x, y):
        for region in self.state["grid_regions"]:
            x1, y1, x2, y2 = region
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def is_in_result_region(self, x, y):
        result_x_base, result_y_base = self.state["result_region_base"]
        result_grid = self.state["result_grid"]
        rel_x = x - result_x_base
        rel_y = y - result_y_base
        if 0 <= rel_x < result_grid.shape[0] and 0 <= rel_y < result_grid.shape[1]:
            return True
        return False

    def handle_keyboard_action(self, key):
        action = next(
            (
                action
                for action, shortcut in self.actions_shortcuts_map.items()
                if shortcut == key
            ),
            None,
        )
        if action is None:
            return

        self.history.append({"action": action})
        self.pre_state = copy.deepcopy(self.state)

        current_x, current_y = self.state["position"]
        grid = self.state["super_grid"]

        if action == "left" and current_x > 0:
            self.state["position"][0] -= 1
        elif action == "right" and current_x < grid.shape[0] - 1:
            self.state["position"][0] += 1
        elif action == "up" and current_y > 0:
            self.state["position"][1] -= 1
        elif action == "down" and current_y < grid.shape[1] - 1:
            self.state["position"][1] += 1

        elif action == "insert_left":
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]
            rel_x = current_x - result_x_base
            rel_y = current_y - result_y_base

            if 0 <= rel_x < result_grid.shape[0] and 0 <= rel_y < result_grid.shape[1]:
                new_row = np.zeros((1, result_grid.shape[1]), dtype=np.int8)
                result_grid = np.vstack(
                    (result_grid[:rel_y, :], new_row, result_grid[rel_y:, :])
                )
                self.state["result_grid"] = result_grid
                self.state["position"][1] += 1  # Adjust y position for row insertion
        elif action == "insert_right":
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]
            rel_x = current_x - result_x_base
            rel_y = current_y - result_y_base

            if 0 <= rel_x < result_grid.shape[0] and 0 <= rel_y < result_grid.shape[1]:
                new_row = np.zeros((1, result_grid.shape[1]), dtype=np.int8)
                result_grid = np.vstack(
                    (result_grid[: rel_y + 1, :], new_row, result_grid[rel_y + 1 :, :])
                )
                self.state["result_grid"] = result_grid
        elif action == "insert_up":
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]
            rel_x = current_x - result_x_base
            rel_y = current_y - result_y_base

            if 0 <= rel_x < result_grid.shape[0] and 0 <= rel_y < result_grid.shape[1]:
                new_column = np.zeros((result_grid.shape[0], 1), dtype=np.int8)
                result_grid = np.hstack(
                    (result_grid[:, :rel_x], new_column, result_grid[:, rel_x:])
                )
                self.state["result_grid"] = result_grid
                self.state["position"][0] += 1  # Adjust x position for column insertion
        elif action == "insert_down":
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]
            rel_x = current_x - result_x_base
            rel_y = current_y - result_y_base

            if 0 <= rel_x < result_grid.shape[0] and 0 <= rel_y < result_grid.shape[1]:
                new_column = np.zeros((result_grid.shape[0], 1), dtype=np.int8)
                result_grid = np.hstack(
                    (
                        result_grid[:, : rel_x + 1],
                        new_column,
                        result_grid[:, rel_x + 1 :],
                    )
                )
                self.state["result_grid"] = result_grid

        elif action == "pill_foreground":
            current_x, current_y = self.state["position"]

            # Check if in grid regions (super_grid)
            if self.is_in_grid_regions(current_x, current_y):
                color_idx = self.state["super_grid"][current_x, current_y]
                if color_idx in self.color_map and color_idx not in [
                    10,
                    11,
                ]:  # Exclude lightgrey, darkgrey
                    self.state["foreground_color"] = self.color_map[color_idx]

            # Check if in result region (result_grid)
            elif self.is_in_result_region(current_x, current_y):
                result_x_base, result_y_base = self.state["result_region_base"]
                rel_x = current_x - result_x_base
                rel_y = current_y - result_y_base
                color_idx = self.state["result_grid"][rel_y, rel_x]
                if color_idx in self.color_map and color_idx not in [
                    10,
                    11,
                ]:  # Exclude lightgrey, darkgrey
                    self.state["foreground_color"] = self.color_map[color_idx]

        elif action == "pill_background":
            current_x, current_y = self.state["position"]

            # Check if in grid regions (super_grid)
            if self.is_in_grid_regions(current_x, current_y):
                color_idx = self.state["super_grid"][current_x, current_y]
                if color_idx in self.color_map and color_idx not in [
                    10,
                    11,
                ]:  # Exclude lightgrey, darkgrey
                    self.state["background_color"] = self.color_map[color_idx]

            # Check if in result region (result_grid)
            elif self.is_in_result_region(current_x, current_y):
                result_x_base, result_y_base = self.state["result_region_base"]
                rel_x = current_x - result_x_base
                rel_y = current_y - result_y_base
                color_idx = self.state["result_grid"][rel_y, rel_x]
                if color_idx in self.color_map and color_idx not in [
                    10,
                    11,
                ]:  # Exclude lightgrey, darkgrey
                    self.state["background_color"] = self.color_map[color_idx]

        elif action == "fill":
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]
            rel_x = current_x - result_x_base
            rel_y = current_y - result_y_base

            fg_color = next(
                key
                for key, value in self.color_map.items()
                if value == self.state["foreground_color"]
            )
            if len(self.state["selected_objects"]) > 0:
                for x, y in self.state["selected_objects"]:
                    object_rel_x = x - result_x_base
                    object_rel_y = y - result_y_base
                    if not (0 <= object_rel_x < result_grid.shape[0]):
                        return
                    if not (0 <= object_rel_y < result_grid.shape[1]):
                        return
                for x, y in self.state["selected_objects"]:
                    result_grid[x - result_x_base, y - result_y_base] = fg_color
            else:
                if (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    result_grid[rel_x, rel_y] = fg_color
        elif action == "flood_fill":
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]
            rel_x = current_x - result_x_base
            rel_y = current_y - result_y_base

            fg_color = next(
                key
                for key, value in self.color_map.items()
                if value == self.state["foreground_color"]
            )

            if 0 <= rel_x < result_grid.shape[0] and 0 <= rel_y < result_grid.shape[1]:
                target_color = result_grid[rel_x, rel_y]
                stack = [(rel_x, rel_y)]
                while stack:
                    x, y = stack.pop()
                    if result_grid[x, y] == target_color:
                        result_grid[x, y] = fg_color
                        for nx, ny in [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]:
                            if (
                                0 <= nx < result_grid.shape[0]
                                and 0 <= ny < result_grid.shape[1]
                                and result_grid[nx, ny] == target_color
                            ):
                                stack.append((nx, ny))

        elif action == "select":
            if self.is_in_grid_regions(
                current_x, current_y
            ) or self.is_in_result_region(current_x, current_y):
                if (current_x, current_y) not in self.state["selected_objects"]:
                    self.state["selected_objects"].append((current_x, current_y))
        elif action == "region_select":
            if len(self.state["selected_objects"]) == 2:
                p1, p2 = self.state["selected_objects"]
                min_x = min(p1[0], p2[0])
                max_x = max(p1[0], p2[0])
                min_y = min(p1[1], p2[1])
                max_y = max(p1[1], p2[1])
                selection = []
                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        selection.append((x, y))
                self.state["selected_objects"] = selection
        elif action == "diagonal_select":
            if len(self.state["selected_objects"]) == 2:
                p1, p2 = self.state["selected_objects"]
                x1, y1 = p1
                x2, y2 = p2
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) == abs(dy) and dx != 0 and dy != 0:
                    step_x = 1 if dx > 0 else -1
                    step_y = 1 if dy > 0 else -1

                    selection = []
                    x, y = x1, y1
                    steps = abs(dx)
                    for _ in range(steps + 1):
                        selection.append((x, y))
                        x += step_x
                        y += step_y
                    self.state["selected_objects"] = selection
        elif action == "unselect":
            if (current_x, current_y) in self.state["selected_objects"]:
                self.state["selected_objects"].remove((current_x, current_y))
        elif action == "unselect_all":
            self.state["selected_objects"] = []

        elif action == "move_left" and len(self.state["selected_objects"]) > 0:
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]

            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                if not (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    return

            # Sort selected_objects: top-to-bottom (increasing y), left-to-right (increasing x)
            sorted_objects = sorted(
                self.state["selected_objects"], key=lambda p: (p[1], p[0])
            )
            bg_color = next(
                key
                for key, value in self.color_map.items()
                if value == self.state["background_color"]
            )
            new_selected = []
            for x, y in sorted_objects:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                new_rel_x = rel_x - 1
                if 0 <= new_rel_x < result_grid.shape[0]:
                    result_grid[new_rel_x, rel_y] = result_grid[rel_x, rel_y]
                    result_grid[rel_x, rel_y] = bg_color
                    new_selected.append((x - 1, y))
            self.state["selected_objects"] = new_selected
        elif action == "move_right" and len(self.state["selected_objects"]) > 0:
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]

            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                if not (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    return

            # Sort selected_objects: top-to-bottom (increasing y), right-to-left (decreasing x)
            sorted_objects = sorted(
                self.state["selected_objects"], key=lambda p: (p[1], -p[0])
            )
            bg_color = next(
                key
                for key, value in self.color_map.items()
                if value == self.state["background_color"]
            )
            new_selected = []
            for x, y in sorted_objects:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                new_rel_x = rel_x + 1
                if 0 <= new_rel_x < result_grid.shape[0]:
                    result_grid[new_rel_x, rel_y] = result_grid[rel_x, rel_y]
                    result_grid[rel_x, rel_y] = bg_color
                    new_selected.append((x + 1, y))
            self.state["selected_objects"] = new_selected
        elif action == "move_up" and len(self.state["selected_objects"]) > 0:
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]
            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                if not (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    return

            # Sort selected_objects: left-to-right (increasing x), top-to-bottom (increasing y)
            sorted_objects = sorted(
                self.state["selected_objects"], key=lambda p: (p[0], p[1])
            )
            bg_color = next(
                key
                for key, value in self.color_map.items()
                if value == self.state["background_color"]
            )
            new_selected = []
            for x, y in sorted_objects:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                new_rel_y = rel_y - 1
                if 0 <= new_rel_y < result_grid.shape[1]:
                    result_grid[rel_x, new_rel_y] = result_grid[rel_x, rel_y]
                    result_grid[rel_x, rel_y] = bg_color
                    new_selected.append((x, y - 1))
            self.state["selected_objects"] = new_selected
        elif action == "move_down" and len(self.state["selected_objects"]) > 0:
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]

            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                if not (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    return

            # Sort selected_objects: left-to-right (increasing x), bottom-to-top (decreasing y)
            sorted_objects = sorted(
                self.state["selected_objects"], key=lambda p: (p[0], -p[1])
            )
            bg_color = next(
                key
                for key, value in self.color_map.items()
                if value == self.state["background_color"]
            )
            new_selected = []
            for x, y in sorted_objects:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                new_rel_y = rel_y + 1
                if 0 <= new_rel_y < result_grid.shape[1]:
                    result_grid[rel_x, new_rel_y] = result_grid[rel_x, rel_y]
                    result_grid[rel_x, rel_y] = bg_color
                    new_selected.append((x, y + 1))
            self.state["selected_objects"] = new_selected
        elif action == "rotate" and self.state["selected_objects"]:
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]

            # Check if all selected objects are within the result region
            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                if not (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    return

            # Find the bounding box and center of selected objects
            min_x = min(x for x, y in self.state["selected_objects"])
            max_x = max(x for x, y in self.state["selected_objects"])
            min_y = min(y for x, y in self.state["selected_objects"])
            max_y = max(y for x, y in self.state["selected_objects"])

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            # Store original colors
            bg_color = next(
                key
                for key, value in self.color_map.items()
                if value == self.state["background_color"]
            )
            color_map = {}
            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                color_map[(x, y)] = result_grid[rel_x, rel_y]

            # Calculate new positions after clockwise rotation
            new_selected = []
            for x, y in self.state["selected_objects"]:
                # Translate to origin relative to center
                rel_x = x - center_x
                rel_y = y - center_y
                # Rotate 90° clockwise: (x, y) -> (y, -x)
                new_rel_x = rel_y
                new_rel_y = -rel_x
                # Translate back
                new_x = int(center_x + new_rel_x)
                new_y = int(center_y + new_rel_y)
                new_selected.append((new_x, new_y))

            # Check if new positions are within bounds
            for x, y in new_selected:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                if not (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    return

            # Clear original positions
            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                result_grid[rel_x, rel_y] = bg_color

            # Apply new positions
            for (old_x, old_y), (new_x, new_y) in zip(
                self.state["selected_objects"], new_selected
            ):
                rel_x = new_x - result_x_base
                rel_y = new_y - result_y_base
                result_grid[rel_x, rel_y] = color_map[(old_x, old_y)]

            self.state["selected_objects"] = new_selected
        elif action == "flip" and self.state["selected_objects"]:
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]

            # Check if all selected objects are within the result region
            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                if not (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    return

            # Find the vertical center of selected objects
            min_y = min(y for x, y in self.state["selected_objects"])
            max_y = max(y for x, y in self.state["selected_objects"])
            center_y = (min_y + max_y) / 2

            # Store original colors
            bg_color = next(
                key
                for key, value in self.color_map.items()
                if value == self.state["background_color"]
            )
            color_map = {}
            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                color_map[(x, y)] = result_grid[rel_x, rel_y]

            # Calculate new positions after vertical flip
            new_selected = []
            for x, y in self.state["selected_objects"]:
                # Reflect over the horizontal center: y' = 2 * center_y - y
                new_y = int(2 * center_y - y)
                new_selected.append((x, new_y))

            # Check if new positions are within bounds
            for x, y in new_selected:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                if not (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    return

            # Clear original positions
            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                result_grid[rel_x, rel_y] = bg_color

            # Apply new positions
            for (old_x, old_y), (new_x, new_y) in zip(
                self.state["selected_objects"], new_selected
            ):
                rel_x = new_x - result_x_base
                rel_y = new_y - result_y_base
                result_grid[rel_x, rel_y] = color_map[(old_x, old_y)]

            self.state["selected_objects"] = new_selected

        elif action == "copy" and self.state["selected_objects"]:
            result_x_base, result_y_base = self.state["result_region_base"]
            result_grid = self.state["result_grid"]
            current_rel_x = current_x - result_x_base
            current_rel_y = current_y - result_y_base

            # Check if current position is in result region
            if not (
                0 <= current_rel_x < result_grid.shape[0]
                and 0 <= current_rel_y < result_grid.shape[1]
            ):
                return

            # Check if any selected objects are in result region (we'll assume we copy if none are)
            all_outside_result = True
            for x, y in self.state["selected_objects"]:
                rel_x = x - result_x_base
                rel_y = y - result_y_base
                if (
                    0 <= rel_x < result_grid.shape[0]
                    and 0 <= rel_y < result_grid.shape[1]
                ):
                    all_outside_result = False
                    break

            if all_outside_result:
                # Store the pattern relative to the minimum coordinates of selected objects
                min_x = min(x for x, y in self.state["selected_objects"])
                min_y = min(y for x, y in self.state["selected_objects"])
                pattern = {}
                for x, y in self.state["selected_objects"]:
                    rel_x = x - min_x
                    rel_y = y - min_y
                    grid_x = x  # Absolute coordinates in super_grid
                    grid_y = y
                    if (
                        0 <= grid_x < self.state["super_grid"].shape[0]
                        and 0 <= grid_y < self.state["super_grid"].shape[1]
                    ):
                        pattern[(rel_x, rel_y)] = self.state["super_grid"][
                            grid_x, grid_y
                        ]

                # Apply pattern at current position in result_grid
                new_selected = []
                for (rel_x, rel_y), color in pattern.items():
                    new_x = current_x + rel_x
                    new_y = current_y + rel_y
                    new_rel_x = new_x - result_x_base
                    new_rel_y = new_y - result_y_base
                    if (
                        0 <= new_rel_x < result_grid.shape[0]
                        and 0 <= new_rel_y < result_grid.shape[1]
                    ):
                        result_grid[new_rel_x, new_rel_y] = color
                        new_selected.append((new_x, new_y))

                # Update selected objects to new positions
                self.state["selected_objects"] = new_selected

        elif action == "undo":
            self.state["foreground_color"] = self.pre_state["foreground_color"]
            self.state["background_color"] = self.pre_state["background_color"]
            self.state["position"] = self.pre_state["position"].copy()
            self.state["result_grid"] = self.pre_state["result_grid"].copy()
            self.state["selected_objects"] = self.pre_state["selected_objects"].copy()

        self.draw_dynamic()

    def step(self, action):
        if action["action"] == "init":
            print(f"""Replaying action: {action["action"]} {action["task_id"]}""")
            self.reset(task_id=action["task_id"])
        elif action["action"] == "mouse":
            print(
                f"""Replaying action: {action["action"]} {action["press"]} {action["x"]} {action["y"]}"""
            )
            self.handle_mouse_action(action["press"], action["x"], action["y"])
        else:
            print(f"""Replaying action: {action["action"]}""")
            key = self.actions_shortcuts_map[action["action"]]
            self.handle_keyboard_action(key)

    def render(self):
        buffered = BytesIO()
        self.final_image = Image.new("RGB", (self.width, self.height), "white")
        self.final_image.paste(self.static_image, (0, 0))
        self.final_image.paste(
            self.dynamic_image,
            (0, 0),
            mask=self.dynamic_image.split()[3],
        )
        self.final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"static_image": img_str}

    def is_done(self):
        """
        Check if the current result_grid matches the ground truth (gt) for the test output.
        """
        result_grid = self.state["result_grid"]
        gt = self.dataset[self.task_id]["output"][self.task_sample_size - 1]

        if result_grid.shape != gt.shape:
            return False

        return np.array_equal(result_grid, gt)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

board_core = BoardCore()


class MouseAction(BaseModel):
    press: bool
    x: float
    y: float


class ChatMessage(BaseModel):
    message: str


class ResetToIdRequest(BaseModel):
    task_id: int


@app.get("/")
async def redirect_to_index():
    return RedirectResponse(url="/static/arc_board_prog.html")


@app.post("/mouse_action/")
async def mouse_action(action: MouseAction):
    board_core.handle_mouse_action(action.press, action.x, action.y)
    return {"status": "success"}


@app.get("/render/")
async def render():
    return board_core.render()


@app.get("/get_history/")
async def get_history():
    return {"status": "success", "history": board_core.history}


@app.post("/save/")
async def save_history(request: Request):
    body = await request.json()
    remark = body.get("remark", "")  # Get remark from request, default to empty string
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_actions = [
        action for action in board_core.history
    ]  # Copy the history actions

    # Format the log entry
    log_entry = f"Date: {current_date} | Actions: {json.dumps(history_actions)} | Remark: {remark}\n"

    # Append to history.log in the current directory
    with open("history.log", "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)

    return {"status": "success", "message": "History saved to history.log"}


@app.post("/keyboard_action/")
async def keyboard_action(request: Request):
    body = await request.json()
    key = body.get("key", "")
    board_core.handle_keyboard_action(key)
    return {"status": "success"}


@app.post("/reset/")
async def reset():
    # Reset to the initial state of the current task
    current_task_id = board_core.task_id
    board_core.reset(task_id=current_task_id)
    return {
        "status": "success",
        "task_id": board_core.task_id,
        "task_key": board_core.task_key,
    }


@app.post("/reset_to_id/")
async def reset_to_id(request: ResetToIdRequest):
    task_id = request.task_id
    if 0 <= task_id < len(board_core.dataset):
        board_core.reset(task_id=task_id)
        return {
            "status": "success",
            "task_id": board_core.task_id,
            "task_key": board_core.task_key,
        }
    else:
        return {
            "status": "error",
            "message": f"Invalid task_id. Must be between 0 and {len(board_core.dataset) - 1}",
        }


@app.post("/next/")
async def next_task():
    # Move to the next task, wrapping around if at the end
    current_task_id = board_core.task_id
    next_task_id = (current_task_id + 1) % len(board_core.dataset)
    board_core.reset(task_id=next_task_id)
    return {
        "status": "success",
        "task_id": next_task_id,
        "task_key": board_core.task_key,
    }


@app.post("/random/")
async def random_task():
    # Select a random task from the dataset
    random_task_id = random.randint(0, len(board_core.dataset) - 1)
    board_core.reset(task_id=random_task_id)
    return {
        "status": "success",
        "task_id": random_task_id,
        "task_key": board_core.task_key,
    }


@app.post("/chat/")
async def chat(chat_message: ChatMessage):
    user_message = chat_message.message
    bot_message = f"Echo: {user_message}"
    return {"response": bot_message}


@app.get("/load_options/")
async def load_options():
    try:
        with open("history.log", "r", encoding="utf-8") as log_file:
            lines = log_file.readlines()

        options = []
        for i, line in enumerate(lines):
            match = re.match(r"Date: (.*?) \| Actions: (.*?) \| Remark: (.*?)", line)
            if match:
                date, actions_json, remark = match.groups()
                try:
                    actions = json.loads(actions_json)
                    if (
                        actions
                        and isinstance(actions, list)
                        and actions[0].get("action") == "init"
                    ):
                        options.append(
                            {
                                "index": i,
                                "date": date,
                                "task_id": actions[0].get("task_id", "Unknown"),
                                "remark": remark,
                                "actions_count": len(actions),
                            }
                        )
                except json.JSONDecodeError:
                    continue
        print(options)
        return {"status": "success", "options": options}
    except FileNotFoundError:
        return {"status": "error", "message": "history.log not found"}
    except Exception as e:
        return {"status": "error", "message": f"Error reading history.log: {str(e)}"}


@app.post("/load/")
async def load_history(request: Request):
    body = await request.json()
    index = body.get("index")
    try:
        with open("history.log", "r", encoding="utf-8") as log_file:
            lines = log_file.readlines()

        if 0 <= index < len(lines):
            line = lines[index]
            match = re.match(r"Date: (.*?) \| Actions: (.*?) \| Remark: (.*)", line)
            if match:
                _, actions_json, _ = match.groups()
                actions = json.loads(actions_json)
                board_core.history = actions
                return {"status": "success", "history": actions}
        return {"status": "error", "message": "Invalid index or entry"}
    except Exception as e:
        return {"status": "error", "message": f"Error loading history: {str(e)}"}


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.board_core = BoardCore()
        self.canvas = tk.Canvas(
            root, width=self.board_core.width, height=self.board_core.height, bg="white"
        )
        self.canvas.pack()
        self.board_core.render()
        rendered = self.board_core.final_image.copy()
        self.tk_image = ImageTk.PhotoImage(rendered)
        self.canvas_image = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.tk_image
        )
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.focus_set()
        self.canvas.bind("<KeyPress>", self.on_key_press)

    def on_press(self, event):
        self.board_core.handle_mouse_action(True, event.x, event.y)
        self.update_canvas()

    def on_key_press(self, event):
        if event.keysym == "R":
            history = copy.deepcopy(self.board_core.history)

            def replay_step(index):
                if index < len(history):
                    action = history[index]
                    self.board_core.step(action)
                    self.update_canvas()
                    self.root.after(500, replay_step, index + 1)  # ms

            replay_step(0)
        else:
            self.board_core.handle_keyboard_action(event.keysym)
            self.update_canvas()

    def update_canvas(self):
        self.board_core.render()
        rendered = self.board_core.final_image.copy()
        self.tk_image = ImageTk.PhotoImage(rendered)
        self.canvas.itemconfig(self.canvas_image, image=self.tk_image)


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)


def run_gui():
    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    DrawingApp(root)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        run_gui()
    else:
        run_server()
