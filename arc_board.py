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


def contains(trajectory, bbox):
    """
    Args:
        trajectory: List[Dict], 鼠标轨迹点列表，每个点包含 {"x": float, "y": float}
        bbox: List[float]
    """
    def point_in_polygon(x, y, poly):
        """
        使用射线法判断点 (x, y) 是否在多边形 poly 内部。
        """
        n = len(poly)
        inside = False
        points = [(p["x"], p["y"]) for p in poly]
        p1x, p1y = points[0]
        for i in range(n + 1):
            p2x, p2y = points[i % n]
            # 检查水平射线与多边形边的交点
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    left, top, right, bottom = bbox
    corners = [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom)
    ]    
    for corner_x, corner_y in corners:
        if not point_in_polygon(corner_x, corner_y, trajectory):
            return False
    return True


def intersects(trajectory, bbox):
    """    
    Args:
        trajectory: List[Dict], 轨迹点列表，每个点包含 {"x": float, "y": float}
        bbox: List[float], 矩形的边界框 [left, top, right, bottom]
    """
    def orientation(p, q, r):
        """
        计算点 p, q, r 的方向，用于判断线段是否相交。
        返回值：
            0 --> 共线
            1 --> 顺时针
            2 --> 逆时针
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        """
        判断点 q 是否在线段 p-r 上。
        """
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    def do_segments_intersect(p1, q1, p2, q2):
        """
        判断线段 p1-q1 和 p2-q2 是否相交。
        """
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1): return True
        if o2 == 0 and on_segment(p1, q2, q1): return True
        if o3 == 0 and on_segment(p2, p1, q2): return True
        if o4 == 0 and on_segment(p2, q1, q2): return True

        return False

    # 如果轨迹少于2个点，无法形成线段
    if len(trajectory) < 2:
        return False

    left, top, right, bottom = bbox
    min_x = min(p["x"] for p in trajectory)
    max_x = max(p["x"] for p in trajectory)
    min_y = min(p["y"] for p in trajectory)
    max_y = max(p["y"] for p in trajectory)
    if max_x < left or min_x > right or max_y < top or min_y > bottom:
        return False

    # 检查轨迹的每个端点是否在 bbox 内部
    for point in trajectory:
        x, y = point["x"], point["y"]
        if left < x < right and top < y < bottom:
            return True

    # 定义矩形的四条边
    bbox_edges = [
        ((left, top), (right, top)),      # 上边
        ((right, top), (right, bottom)),  # 右边
        ((right, bottom), (left, bottom)),# 下边
        ((left, bottom), (left, top))     # 左边
    ]

    # 检查轨迹的每条线段是否与矩形的任意一边相交
    for i in range(len(trajectory) - 1):
        p1 = (trajectory[i]["x"], trajectory[i]["y"])
        p2 = (trajectory[i + 1]["x"], trajectory[i + 1]["y"])
        
        # 快速拒绝测试每条线段
        if (max(p1[0], p2[0]) < left or min(p1[0], p2[0]) > right or
            max(p1[1], p2[1]) < top or min(p1[1], p2[1]) > bottom):
            continue
        
        for edge_start, edge_end in bbox_edges:
            if do_segments_intersect(p1, p2, edge_start, edge_end):
                return True

    return False

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
        self.static_image = Image.new("RGB", (self.width, self.height), "white")
        self.static_layer = ImageDraw.Draw(self.static_image)
        self.pen_and_debug_text_image = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 0))
        self.pen_and_debug_text_layer = ImageDraw.Draw(self.pen_and_debug_text_image)
        self.pen_trajectory_image = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 0))
        self.pen_trajectory_layer = ImageDraw.Draw(self.pen_trajectory_image)
        self.select_trajectory_image = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 0))
        self.select_trajectory_layer = ImageDraw.Draw(self.select_trajectory_image)
        self.state = {
            "tool": "select",
            "color": "black",
            "background_color": "white",
            "pen_size": 4,
            "eraser_size": 20,
            "mouse_state":{},
            "pre_mouse_state":{},
            "selected_objects": [],
            "press_mouse_trajectory": [],
            "rendered_static_bbox":{}
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
            "tool_inc",
            "tool_dec",
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
        for i, button in enumerate(self.toolbar_buttons):
            bbox = [
                self.toolbar_start_x,
                self.toolbar_start_y + i * (self.toolbar_size + self.toolbar_gap),
                self.toolbar_start_x + + self.toolbar_size,
                self.toolbar_start_y + i * (self.toolbar_size + self.toolbar_gap)
                + self.toolbar_size ,
            ]
            self.state["rendered_static_bbox"][button] = bbox

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


    def draw_grid_region(self):
        def draw_single_grid(grid: np.array, base, grid_name):
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    color = self.color_map[grid[i][j]]
                    bbox = [
                        base[0] + i * self.object_pixel_size,
                        base[1] + j * self.object_pixel_size,
                        base[0] + (i + 1) * self.object_pixel_size,
                        base[1] + (j + 1) * self.object_pixel_size,
                    ]
                    #  50% 灰度 (128, 128, 128)
                    self.static_layer.rectangle(
                        bbox, fill=color, outline=(128, 128, 128), width=1
                    )
                    self.state["rendered_static_bbox"][f"{grid_name}_{i}_{j}"] = bbox
        self.train_region_base = [250, 50]
        self.train_test_region_gap = 400
        self.test_region_base = [self.train_region_base[0] + self.train_test_region_gap, self.train_region_base[1]]
        self.grid_gap = 20
        self.object_pixel_size = 20
        self.max_result_grid_size = 20
        self.result_region_pixels_size = [self.object_pixel_size * self.max_result_grid_size, self.object_pixel_size * self.max_result_grid_size]

        task_sample_size = len(self.dataset[self.task_id]["input"])
        task_train_region_base = self.train_region_base
        task_test_region_base = self.test_region_base
        for task_sample_index in range(task_sample_size):
            mode = "train" if task_sample_index < task_sample_size - 1 else "test"
            input_pixel_size = None
            output_pixel_size = None
            for put in ["input", "output"]:
                grid = self.dataset[self.task_id][put][task_sample_index]
                grid_pixel_size = [
                    grid.shape[0] * self.object_pixel_size,
                    grid.shape[1] * self.object_pixel_size,
                ]
                if put == "input":
                    input_pixel_size = grid_pixel_size
                else:
                    output_pixel_size = grid_pixel_size
                if mode == "train":
                    region_base = task_train_region_base
                else:
                    region_base = task_test_region_base
                
                if put == "output":
                    region_base = [
                            region_base[0] + input_pixel_size[0] + self.grid_gap,
                            region_base[1],
                        ]
                if mode == "test" and put == "output":
                    self.static_layer.rectangle(
                        [
                            region_base[0],
                            region_base[1],
                            region_base[0] + self.result_region_pixels_size[0],
                            region_base[1] + self.result_region_pixels_size[1],
                        ],
                        outline="black",
                        width=2,
                    )
                else:
                    draw_single_grid(grid, region_base, mode + "_" + str(task_sample_index) + put)
            task_train_region_base = [
                    task_train_region_base[0],
                    task_train_region_base[1]
                    + max(input_pixel_size[1], output_pixel_size[1])
                    + self.grid_gap,
                ]

    def draw_toolbar(self):
        self.static_layer.rectangle(
            [0, 0, self.width, self.height], fill=(255, 255, 255, 0)
        )
        for i, button in enumerate(self.toolbar_buttons):
            bbox = self.state["rendered_static_bbox"][button]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            radius = (bbox[2] - bbox[0]) / 3
            self.static_layer.circle(
                (center_x, center_y), radius, outline="black", width=2
            )
            bbox = [
                bbox[0] + self.toolbar_size + self.toolbar_gap/2,
                bbox[1],
                bbox[2] + self.toolbar_size + self.toolbar_gap/2,
                bbox[3],
            ]
            if button.startswith("color"):
                color = button.split("_")[1]
                self.static_layer.rectangle(bbox, fill=color, outline="black", width=2)
            else:
                content_text = button[5:].upper()
                center_x = bbox[0]
                center_y = bbox[1]
                font_size = 16
                from PIL import ImageFont
                import matplotlib.font_manager as fm
                font_path = fm.findfont(fm.FontProperties(family='Arial'))
                font = ImageFont.truetype(font_path, size=font_size)
                font_height = font.getbbox(content_text)[3] - font.getbbox(content_text)[1]
                offset_y = (bbox[3] - bbox[1] - font_height) / 2
                self.static_layer.text(
                    (center_x, center_y + offset_y),
                    content_text,
                    fill="black",
                    font=font,
                    anchor="lt",)

    def draw_pen_and_debug_text(self):
        self.pen_and_debug_text_layer.rectangle(
            [0, 0, self.width, self.height], fill=(255, 255, 255, 0)
        )
        # debug
        self.text_content = json.dumps(self.state, indent=2)
        self.text_position = (600, 0)
        self.text_color = "black"
        self.text_font_size = 20
        self.pen_and_debug_text_layer.text(self.text_position, self.text_content, fill=self.text_color)

        # Draw diagonal lines on selected blocks
        for bbox in self.state["selected_objects"]:
            # Draw two diagonal lines to form an "X"
            self.pen_trajectory_layer.line(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                fill="black",
                width=1
            )
            self.pen_trajectory_layer.line(
                [(bbox[0], bbox[3]), (bbox[2], bbox[1])],
                fill="black",
                width=1
            )

        if self.state["tool"] == "pen" or "select" in self.state["tool"]:
            self.pen_and_debug_text_layer.circle(
                (self.state["mouse_state"]["x"], self.state["mouse_state"]["y"]),
                self.state["pen_size"],
                outline=self.state["color"],
                width=2,
            )
        elif self.state["tool"] == "eraser":
            self.pen_and_debug_text_layer.rectangle(
                [
                    (
                        self.state["pre_mouse_state"]["x"] - self.state["eraser_size"] / 2,
                        self.state["pre_mouse_state"]["y"] - self.state["eraser_size"] / 2,
                    ),
                    (
                        self.state["mouse_state"]["x"] + self.state["eraser_size"] / 2,
                        self.state["mouse_state"]["y"] + self.state["eraser_size"] / 2,
                    ),
                ],
                fill="white",
                outline="black",
                width=2,
            )

    def handle_mouse_action(self, press: bool, x: float, y: float):
        self.state["mouse_state"] = {"press":press, "x":x, "y":y}
        if press:
            trajectory_point = {"press":press, "x":x, "y":y}
            self.state["press_mouse_trajectory"].append(trajectory_point)
            if self.state["tool"] == "pen":
                self.pen_trajectory_layer.line(
                    [(self.state["mouse_state"]["x"], self.state["mouse_state"]["y"]), (x, y)],
                    fill=self.state["color"],
                    width=self.state["pen_size"],
                    joint="curve",
                )
            elif self.state["tool"] == "eraser":
                self.pen_trajectory_layer.rectangle(
                    [
                        (
                            x - self.state["eraser_size"] / 2,
                            y - self.state["eraser_size"] / 2,
                        ),
                        (
                            x + self.state["eraser_size"] / 2,
                            y + self.state["eraser_size"] / 2,
                        ),
                    ],
                    fill="white",
                    outline="black",
                    width=2,
                )
            elif self.state["tool"] == "select":
                # 绘制选择轨迹
                if len(self.state["press_mouse_trajectory"]) > 1:
                    cur_xy = self.state["press_mouse_trajectory"][-1]
                    pre_xy = self.state["press_mouse_trajectory"][-2]
                    self.select_trajectory_layer.line(
                        [(pre_xy["x"], pre_xy["y"]), (cur_xy["x"], cur_xy["y"])],
                        fill="black",
                        width=3,
                        joint="curve",
                    )
        else:
            if len(self.state["press_mouse_trajectory"]) > 0:
                trajectory_point = {"press":press, "x":x, "y":y}
                self.state["press_mouse_trajectory"].append(trajectory_point)
                if self.state["tool"] == "select":
                    # 清除选择轨迹 TODO maybe use pen layer more effcient?
                    self.select_trajectory_layer.rectangle([0, 0, self.width, self.height], fill=(255, 255, 255, 0))
                    # 鼠标抬起后，判断轨迹是否闭合，判断标准起点和终点的像素点小于5 pixel
                    start_point = self.state["press_mouse_trajectory"][0]
                    end_point = self.state["press_mouse_trajectory"][-1]
                    if abs(start_point["x"] - end_point["x"]) < 5 and abs(start_point["y"] - end_point["y"]) < 5:
                        # 闭合，选择完全在区域内部的方块
                        for name, bbox in self.state["rendered_static_bbox"].items():
                            if "train" in name or "test" in name:
                                if contains(self.state["press_mouse_trajectory"], bbox):
                                    self.state["selected_objects"].append(bbox)
                    else:
                        # 不闭合，选择与轨迹相交的方块
                        for name, bbox in self.state["rendered_static_bbox"].items():
                            if "train" in name or "test" in name:
                                if "train_input_1_1" in name:
                                    print("------------")
                                if intersects(self.state["press_mouse_trajectory"], bbox):
                                    self.state["selected_objects"].append(bbox)
                    # 选中的方块颜色不变，填充斜线 draw in pen layer

                    # TODO
                    # 如何取消选中的物体？点击空白区域？
                    # 选中物体，点击fill，选择颜色，填充颜色
                    # 选中物体，如果物体在result区域，点击rotate，旋转物体
                    # 选中物体，如果物体在result区域，点击flip，翻转物体
                    # 选中物体，如果物体在input区域，点击move，移动物体，背景区域填充
                    # result区域背景颜色？被移动的物体的原来不覆盖区域自动填充为背景颜色
                    # 结果边框怎么定？
                self.state["press_mouse_trajectory"] = []
            else:
                pass

                # if pre_button == cur_button:
                #     if self.state["tool"] == "pen":
                #         if cur_button == "eraser":
                #             self.state["tool"] = "eraser"
                #         elif cur_button == "increase_size":
                #             if self.state["pen_size"] <= 6:
                #                 self.state["pen_size"] += 2
                #         elif cur_button == "decrease_size":
                #             if self.state["pen_size"] >= 2:
                #                 self.state["pen_size"] -= 2
                #         elif cur_button.startswith("color"):
                #             self.state["color"] = cur_button.split("_")[1]
                #     elif self.state["tool"] == "eraser":
                #         if cur_button == "pen":
                #             self.state["tool"] = "pen"
                #         elif cur_button == "increase_size":
                #             if self.state["eraser_size"] <= 18:
                #                 self.state["eraser_size"] += 2
                #         elif cur_button == "decrease_size":
                #             if self.state["eraser_size"] >= 2:
                #                 self.state["eraser_size"] -= 2
        
        self.draw_pen_and_debug_text()
        self.state["pre_mouse_state"] = {"press":press, "x":x, "y":y}


    def render(self):
        buffered = BytesIO()
        self.final_image = Image.new("RGB", (self.width, self.height), "white")
        self.final_image.paste(self.static_image, (0, 0))
        self.final_image.paste(
            self.pen_trajectory_image, (0, 0), mask=self.pen_trajectory_image.split()[3]
        )
        self.final_image.paste(
            self.select_trajectory_image, (0, 0), mask=self.select_trajectory_image.split()[3]
        )
        self.final_image.paste(
            self.pen_and_debug_text_image, (0, 0), mask=self.pen_and_debug_text_image.split()[3]
        )

        self.final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"static_image": img_str}


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
