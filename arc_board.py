import base64
import json
import os
import random
import sys
import tkinter as tk
from io import BytesIO

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageTk
from pydantic import BaseModel


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
    assert tasks_keys == solutions_keys, (
        "Error: The keys of tasks and solutions do not match."
    )
    for key in solutions.keys():
        for i in range(len(solutions[key])):
            tasks[key]["test"][i]["output"] = solutions[key][i]
    result = []
    for key, task in tasks.items():
        # train_two = random.sample(task["train"], 2)
        # test_one = random.sample(task["test"], 1)
        train_two = task["train"][:3] if len(task["train"]) >= 3 else task["train"]
        test_one = task["test"]
        sample = []
        sample.extend(train_two)
        sample.extend(test_one)
        x = [np.array(t["input"]) for t in sample]
        y = [np.array(t["output"]) for t in sample]

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


class MouseAction(BaseModel):
    press: bool
    x: float
    y: float


class BoardCore:
    def __init__(self):
        self.dataset = get_dataset()
        self.task_id = 0
        self.width = 1200
        self.height = 800
        self.board_image = Image.new("RGB", (self.width, self.height), "white")
        self.board_layer = ImageDraw.Draw(self.board_image)
        self.tool_image = Image.new("RGBA", (self.width, self.height), "white")
        self.tool_layer = ImageDraw.Draw(self.tool_image)
        self.pen_image = Image.new("RGBA", (self.width, self.height), "white")
        self.pen_layer = ImageDraw.Draw(self.pen_image)
        self.text_image = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 0))
        self.text_layer = ImageDraw.Draw(self.text_image)

        # select 点击，拖动，松开，显示鼠标轨迹
        # 如果轨迹闭合，就选择完全在区域内部的方块
        # 如果轨迹不闭合，就选择与轨迹相交的方块
        # 选中的方块颜色不变，填充斜线
        # 选中区域如果在result区域，是可以选择填充颜色
        # 选中区域如果在input区域，是可以将区域拖动到output区域
        # 再次按住鼠标左键，拖动，可以移动选中的方块
        # 如何取消选中的物体？点击空白区域？

        # 选中物体，点击fill，选择颜色，填充颜色
        # 选中物体，如果物体在result区域，点击rotate，旋转物体
        # 选中物体，如果物体在result区域，点击flip，翻转物体
        # 选中物体，如果物体在input区域，点击move，移动物体，背景区域填充

        # 背景颜色？被移动的物体的原来不覆盖区域自动填充为背景颜色
        # 背景颜色怎么选择？颜色有两个前景和背景

        # result区域，是否有背景grid
        # 结果边框怎么定？

        self.state = {
            "tool": "pen",
            "color": "black",
            "pen_size": 4,
            "eraser_size": 20,
            "pre_x": 0,
            "pre_y": 0,
            "pre_press": False,
            "button": None,
            "selected_objects": [],
            "mouse_trajectory": [{"xy":[1,1],"bbox":[]},],
            "background_color": "white",
            "rendered_object_bbox":[{"name":"","bbox":""}]
        }
        self.toolbar_size = 20
        self.toolbar_start_x = 10
        self.toolbar_start_y = 10
        self.toolbar_gap = 10
        self.toolbar_buttons = [
            "tool_select",
            "tool_flood_select",
            "tool_backgroud_select",
            "tool_fill",
            "tool_move",
            "tool_rotate",
            "tool_flip",
            "tool_pen",
            "tool_eraser",
            "size_increase",
            "size_decrease",
            "color_black",
            "color_white",
            "color_blue",
            "color_red",
            "color_green",
            "color_yellow",
            "color_grey",
            "color_darkred",
            "color_darkblue",
            "color_brown",
        ]

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
        }

        self.draw_toolbar()
        self.draw_grid_region()

    def draw_text_layer(self):
        self.text_content = json.dumps(self.state, indent=2)
        self.text_position = (600, 600)
        self.text_color = "black"
        self.text_font_size = 20
        self.text_image = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 0))
        self.text_layer = ImageDraw.Draw(self.text_image)
        self.text_layer.text(self.text_position, self.text_content, fill=self.text_color)

    def draw_grid_region(self):
        task_sample_size = len(self.dataset[self.task_id]["input"])

        def draw_single_grid(grid: np.array, base):
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    color = self.color_map[grid[i][j]]
                    bbox = [
                        base[0] + i * self.object_size,
                        base[1] + j * self.object_size,
                        base[0] + (i + 1) * self.object_size,
                        base[1] + (j + 1) * self.object_size,
                    ]
                    #  50% 灰度 (128, 128, 128)
                    self.board_layer.rectangle(
                        bbox, fill=color, outline=(128, 128, 128), width=1
                    )

        task_train_region_base = [100, 100]
        task_test_region_base = [400 + 100, 100]
        grid_gap = 20
        self.object_size = 20
        result_region_pixels_size = [self.object_size * 20, self.object_size * 20]

        for task_sample_index in range(task_sample_size):
            input = self.dataset[self.task_id]["input"][task_sample_index]
            output = self.dataset[self.task_id]["output"][task_sample_index]
            input_pixel_size = [
                input.shape[0] * self.object_size,
                input.shape[1] * self.object_size,
            ]
            output_pixel_size = [
                output.shape[0] * self.object_size,
                output.shape[1] * self.object_size,
            ]
            if task_sample_index < task_sample_size - 1:
                input_region_baze = task_train_region_base
                task_train_region_base = [
                    task_train_region_base[0],
                    task_train_region_base[1]
                    + max(input_pixel_size[1], output_pixel_size[1])
                    + grid_gap,
                ]
            else:
                input_region_baze = task_test_region_base
            output_region_base = [
                input_region_baze[0] + input_pixel_size[0] + grid_gap,
                input_region_baze[1],
            ]
            draw_single_grid(input, input_region_baze)

            if task_sample_index < task_sample_size - 1:
                draw_single_grid(output, output_region_base)
            else:
                self.board_layer.rectangle(
                    [
                        output_region_base[0],
                        output_region_base[1],
                        output_region_base[0] + result_region_pixels_size[0],
                        output_region_base[1] + result_region_pixels_size[1],
                    ],
                    outline="black",
                    width=2,
                )

    def draw_toolbar(self):
        self.tool_layer.rectangle(
            [0, 0, self.width, self.height], fill=(255, 255, 255, 0)
        )
        for i, button in enumerate(self.toolbar_buttons):
            bbox = [
                self.toolbar_start_x + i * (self.toolbar_size + self.toolbar_gap),
                self.toolbar_start_y,
                self.toolbar_start_x
                + i * (self.toolbar_size + self.toolbar_gap)
                + self.toolbar_size,
                self.toolbar_start_y + self.toolbar_size,
            ]
            if button.startswith("color"):
                color = button.split("_")[1]
                self.tool_layer.rectangle(bbox, fill=color, outline="black", width=2)
            elif button == "tool_pen":
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                radius = (bbox[2] - bbox[0]) / 2
                self.tool_layer.circle(
                    (center_x, center_y), radius, outline="black", width=2
                )
            elif button == "tool_eraser":
                self.tool_layer.rectangle(bbox, outline="black", width=2)
            elif button == "size_increase":
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                self.tool_layer.line(
                    [(center_x - 10, center_y), (center_x + 10, center_y)],
                    fill="black",
                    width=4,
                )
                self.tool_layer.line(
                    [(center_x, center_y - 10), (center_x, center_y + 10)],
                    fill="black",
                    width=4,
                )
            elif button == "size_decrease":
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                self.tool_layer.line(
                    [(center_x - 10, center_y), (center_x + 10, center_y)],
                    fill="black",
                    width=4,
                )

    def draw_mouse_trajectory_and_select(self):
        self.pen_layer.rectangle(
            [0, 0, self.width, self.height], fill=(255, 255, 255, 0)
        )

        if self.state["tool"] == "pen":
            self.pen_layer.circle(
                (self.state["pre_x"], self.state["pre_y"]),
                self.state["pen_size"],
                outline=self.state["color"],
                width=2,
            )
        else:
            self.pen_layer.rectangle(
                [
                    (
                        self.state["pre_x"] - self.state["eraser_size"] / 2,
                        self.state["pre_y"] - self.state["eraser_size"] / 2,
                    ),
                    (
                        self.state["pre_x"] + self.state["eraser_size"] / 2,
                        self.state["pre_y"] + self.state["eraser_size"] / 2,
                    ),
                ],
                fill="white",
                outline="black",
                width=2,
            )

    def handle_mouse_action(self, press: bool, x: float, y: float):
        pre_button = self.state["button"]
        cur_button = None
        pre_press = self.state["pre_press"]
        cur_press = press
        for i, button in enumerate(self.toolbar_buttons):
            bbox = [
                self.toolbar_start_x + i * (self.toolbar_size + self.toolbar_gap),
                self.toolbar_start_y,
                self.toolbar_start_x + (i + 1) * (self.toolbar_size + self.toolbar_gap),
                self.toolbar_start_y + self.toolbar_size,
            ]
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                cur_button = button

        if pre_press and cur_press:
            if pre_button is None and cur_button is None:
                if self.state["tool"] == "pen":
                    self.board_layer.line(
                        [(self.state["pre_x"], self.state["pre_y"]), (x, y)],
                        fill=self.state["color"],
                        width=self.state["pen_size"],
                        joint="curve",
                    )
        elif not pre_press and cur_press:
            if pre_button is None and cur_button is not None:
                self.state["button"] = cur_button
        elif pre_press and not cur_press:
            if pre_button is not None and cur_button is not None:
                if pre_button == cur_button:
                    if self.state["tool"] == "pen":
                        if cur_button == "eraser":
                            self.state["tool"] = "eraser"
                        elif cur_button == "increase_size":
                            if self.state["pen_size"] <= 6:
                                self.state["pen_size"] += 2
                        elif cur_button == "decrease_size":
                            if self.state["pen_size"] >= 2:
                                self.state["pen_size"] -= 2
                        elif cur_button.startswith("color"):
                            self.state["color"] = cur_button.split("_")[1]
                    elif self.state["tool"] == "eraser":
                        if cur_button == "pen":
                            self.state["tool"] = "pen"
                        elif cur_button == "increase_size":
                            if self.state["eraser_size"] <= 18:
                                self.state["eraser_size"] += 2
                        elif cur_button == "decrease_size":
                            if self.state["eraser_size"] >= 2:
                                self.state["eraser_size"] -= 2
                self.state["button"] = None
        self.draw_mouse_trajectory_and_select()
        self.draw_text_layer()
        self.state["pre_press"] = cur_press
        self.state["pre_x"] = x
        self.state["pre_y"] = y

    def render(self):
        buffered = BytesIO()
        self.final_image = Image.new("RGB", (self.width, self.height), "white")
        self.final_image.paste(self.board_image, (0, 0))
        self.final_image.paste(
            self.pen_image, (0, 0), mask=self.pen_image.convert("RGBA").split()[3]
        )
        self.final_image.paste(
            self.tool_image, (0, 0), mask=self.tool_image.convert("RGBA").split()[3]
        )
        self.final_image.paste(
            self.text_image, (0, 0), mask=self.text_image.split()[3]
        )
        self.final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"board_image": img_str}


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

board_core = BoardCore()


@app.get("/")
async def redirect_to_index():
    return RedirectResponse(url="/static/index.html")


@app.post("/mouse_action/")
async def mouse_action(action: MouseAction):
    board_core.handle_mouse_action(action.press, action.x, action.y)
    return {"status": "success"}


@app.get("/render/")
async def render():
    return board_core.render()


@app.post("/chat/")
async def chat(message: str):
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer YOUR_OPENAI_API_KEY",
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": message.message}],
                "max_tokens": 300,
            },
        )
        response.raise_for_status()
        data = response.json()
        bot_message = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"response": bot_message}


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
        self.canvas.bind("<Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.ms_per_fps = 1000 // 120
        # self.update_canvas_loop()

    def on_press(self, event):
        self.board_core.handle_mouse_action(True, event.x, event.y)
        self.update_canvas()

    def on_move(self, event):
        if event.state & 0x0100:  # left mouse button is pressed
            self.board_core.handle_mouse_action(True, event.x, event.y)
        else:
            self.board_core.handle_mouse_action(False, event.x, event.y)
        self.update_canvas()

    def on_release(self, event):
        self.board_core.handle_mouse_action(True, event.x, event.y)
        self.board_core.handle_mouse_action(False, event.x, event.y)
        self.update_canvas()

    def update_canvas(self):
        self.board_core.render()
        rendered = self.board_core.final_image.copy()
        self.tk_image = ImageTk.PhotoImage(rendered)
        self.canvas.itemconfig(self.canvas_image, image=self.tk_image)

    def update_canvas_loop(self):
        self.update_canvas()
        self.root.after(self.ms_per_fps, self.update_canvas_loop)

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)


def run_gui():
    root = tk.Tk()
    DrawingApp(root)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        run_gui()
    else:
        run_server()
