import sys
import math
import pygame
from pygame.locals import *

# ---------------------------------------------------------------------------
# 2D Vector
# ---------------------------------------------------------------------------
class Vec2:
    __slots__ = ('x', 'y')
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    def __add__(self, other): return Vec2(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return Vec2(self.x - other.x, self.y - other.y)
    def __mul__(self, s): return Vec2(self.x * s, self.y * s)
    def __rmul__(self, s): return self * s
    def __truediv__(self, s): return Vec2(self.x / s, self.y / s)
    def __neg__(self): return Vec2(-self.x, -self.y)
    def dot(self, other): return self.x * other.x + self.y * other.y
    def length_sq(self): return self.x * self.x + self.y * self.y
    def length(self): return math.sqrt(self.length_sq())
    def normalized(self):
        l = self.length()
        if l == 0:
            return Vec2(0, 0)
        return self / l
    def copy(self): return Vec2(self.x, self.y)
    def __repr__(self): return f"Vec2({self.x:.2f}, {self.y:.2f})"

# ---------------------------------------------------------------------------
# Rigid Body
# ---------------------------------------------------------------------------
class Body:
    WALL = 0
    PLAYER = 1
    BOX = 2
    BUTTON = 3
    GOAL = 4
    ZONE = 5
    TEXT = 6
    ZONE_COPY = 7
    ZONE_REMOVE = 8
    ZONE_RESET = 9
    ZONE_NEXT = 10
    ZONE_PREV = 11

    SENSOR_TYPES = (ZONE, ZONE_COPY, ZONE_REMOVE, ZONE_RESET, ZONE_NEXT, ZONE_PREV)

    def __init__(self, pos, size, mass, body_type, color, name="", color_name="", parts=None, number=None, operator=None, label="", required_number=None):
        self.pos = pos.copy()
        self.size = size.copy()
        self.vel = Vec2(0, 0)
        self.acc = Vec2(0, 0)
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        self.body_type = body_type
        self.color = color
        self.name = name
        self.color_name = color_name
        self.parts = parts if parts is not None else [(Vec2(0, 0), size.copy())]
        self.number = number
        self.operator = operator
        self.label = label
        self.required_number = required_number
        self.pressed = False
        self.in_zone = False
        self.zone_rewarded = False
        self.cooldown = 0.0
        self.floating_text = ""
        self.floating_timer = 0.0

    def min_x(self): return self.pos.x - self.size.x
    def max_x(self): return self.pos.x + self.size.x
    def min_y(self): return self.pos.y - self.size.y
    def max_y(self): return self.pos.y + self.size.y

    def apply_force(self, f: Vec2):
        self.acc = self.acc + f * self.inv_mass

    def center_in_zone(self, zone):
        return (zone.min_x() <= self.pos.x <= zone.max_x() and
                zone.min_y() <= self.pos.y <= zone.max_y())

# ---------------------------------------------------------------------------
# Physics World
# ---------------------------------------------------------------------------
class World:
    GRAVITY = 120.0
    MU_STATIC = 0.25
    MU_KINETIC = 0.15
    INPUT_FORCE = 1500.0
    CONTACT_STIFF = 6000.0
    CONTACT_DAMP = 90.0
    LINEAR_DAMP = 0.985
    MAX_SPEED = 300.0
    FORCE_RAMP = 8.0

    def __init__(self):
        self.bodies = []
        self.gate_open = False
        self.gate_body = None
        self._input_force = Vec2(0, 0)

    def add(self, b: Body):
        self.bodies.append(b)
        return b

    def clear(self):
        self.bodies.clear()
        self.gate_body = None
        self.gate_open = False
        self._input_force = Vec2(0, 0)

    def is_sensor(self, b):
        return b.body_type in Body.SENSOR_TYPES

    def step(self, dt):
        for b in self.bodies:
            if b.mass <= 0:
                continue
            normal_force = b.mass * self.GRAVITY
            v = b.vel
            speed = v.length()
            if speed > 2.0:
                f_friction = -v.normalized() * (self.MU_KINETIC * normal_force)
                b.apply_force(f_friction)
            elif speed <= 2.0:
                b.vel = b.vel * 0.5
            if b.body_type == Body.PLAYER:
                b.apply_force(self._input_force)
            if b.cooldown > 0:
                b.cooldown -= dt
            if b.floating_timer > 0:
                b.floating_timer -= dt
                if b.floating_timer <= 0:
                    b.floating_text = ""

        for b in self.bodies:
            if b.mass <= 0:
                continue
            b.vel = b.vel + b.acc * dt
            if b.vel.length() > self.MAX_SPEED:
                b.vel = b.vel.normalized() * self.MAX_SPEED
            b.vel = b.vel * self.LINEAR_DAMP
            b.acc = Vec2(0, 0)

        for b in self.bodies:
            if b.mass <= 0:
                continue
            b.pos = b.pos + b.vel * dt

        iterations = 6
        for _ in range(iterations):
            self._solve_contacts(dt)

        self._check_buttons()
        self._check_zones()
        result = self._check_special_zones()
        self._check_goal()

        if self.gate_body:
            if self.gate_open:
                target_y = 150
                self.gate_body.pos.y += (target_y - self.gate_body.pos.y) * 5 * dt
            else:
                target_y = 350
                self.gate_body.pos.y += (target_y - self.gate_body.pos.y) * 5 * dt
        return result

    def _solve_contacts(self, dt):
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                a = self.bodies[i]
                b = self.bodies[j]
                if a.mass <= 0 and b.mass <= 0:
                    continue
                if self.is_sensor(a) or self.is_sensor(b):
                    continue
                if min(a.max_x(), b.max_x()) - max(a.min_x(), b.min_x()) <= 0:
                    continue
                if min(a.max_y(), b.max_y()) - max(a.min_y(), b.min_y()) <= 0:
                    continue

                for off_a, sz_a in a.parts:
                    for off_b, sz_b in b.parts:
                        pos_a = a.pos + off_a
                        pos_b = b.pos + off_b
                        overlap_x = min(pos_a.x + sz_a.x, pos_b.x + sz_b.x) - max(pos_a.x - sz_a.x, pos_b.x - sz_b.x)
                        overlap_y = min(pos_a.y + sz_a.y, pos_b.y + sz_b.y) - max(pos_a.y - sz_a.y, pos_b.y - sz_b.y)
                        if overlap_x <= 0 or overlap_y <= 0:
                            continue

                        if overlap_x < overlap_y:
                            nx = -1.0 if pos_a.x < pos_b.x else 1.0
                            ny = 0.0
                            penetration = overlap_x
                        else:
                            nx = 0.0
                            ny = -1.0 if pos_a.y < pos_b.y else 1.0
                            penetration = overlap_y

                        normal = Vec2(nx, ny)
                        rel_vel = a.vel.copy()
                        if b.mass > 0:
                            rel_vel = rel_vel - b.vel
                        vel_along_normal = rel_vel.dot(normal)
                        if vel_along_normal > 0:
                            continue

                        contact_force_mag = self.CONTACT_STIFF * penetration - self.CONTACT_DAMP * vel_along_normal
                        if contact_force_mag < 0:
                            contact_force_mag = 0
                        contact_force = normal * contact_force_mag

                        if a.mass > 0:
                            a.vel = a.vel + contact_force * a.inv_mass * dt
                        if b.mass > 0:
                            b.vel = b.vel - contact_force * b.inv_mass * dt

                        percent = 0.4
                        slop = 0.01
                        correction = max(penetration - slop, 0) / (a.inv_mass + b.inv_mass) * percent
                        if a.mass > 0:
                            a.pos = a.pos + normal * correction * a.inv_mass
                        if b.mass > 0:
                            b.pos = b.pos - normal * correction * b.inv_mass

    def _check_buttons(self):
        for b in self.bodies:
            if b.body_type != Body.BUTTON:
                continue
            was_pressed = b.pressed
            b.pressed = False
            button_top = b.pos.y - b.size.y
            for other in self.bodies:
                if other is b:
                    continue
                if other.body_type not in (Body.PLAYER, Body.BOX):
                    continue
                if other.max_x() > b.min_x() and other.min_x() < b.max_x():
                    other_bottom = other.pos.y + other.size.y
                    if 0 <= (other_bottom - button_top) <= 8:
                        b.pressed = True
                        break
            if b.pressed != was_pressed:
                pass
        all_pressed = all(b.pressed for b in self.bodies if b.body_type == Body.BUTTON)
        self.gate_open = all_pressed

    def _check_zones(self):
        zones = [b for b in self.bodies if b.body_type == Body.ZONE]
        for b in self.bodies:
            if b.body_type == Body.BOX:
                b.in_zone = False
                for z in zones:
                    if z.required_number is not None:
                        if b.center_in_zone(z):
                            b.in_zone = True
                            break
                    elif z.color_name and b.color_name == z.color_name:
                        if b.center_in_zone(z):
                            b.in_zone = True
                            break
                    elif z.color_name == "" and b.center_in_zone(z):
                        b.in_zone = True
                        break

    def _check_special_zones(self):
        to_remove = []
        to_add = []
        for b in self.bodies:
            if b.body_type != Body.BOX:
                continue
            for z in self.bodies:
                if z.body_type == Body.ZONE_REMOVE:
                    if b.center_in_zone(z):
                        to_remove.append(b)
                        break
                elif z.body_type == Body.ZONE_COPY:
                    if b.center_in_zone(z) and b.cooldown <= 0:
                        copy = Body(
                            Vec2(z.pos.x + z.size.x + 40, z.pos.y),
                            b.size.copy(), b.mass, Body.BOX, b.color,
                            name=b.name + "_copy", color_name=b.color_name,
                            parts=b.parts, number=b.number, operator=b.operator
                        )
                        to_add.append(copy)
                        b.cooldown = 1.5
                        break
        for b in to_remove:
            if b in self.bodies:
                self.bodies.remove(b)
        for b in to_add:
            self.bodies.append(b)

    def _check_goal(self):
        goal = None
        player = None
        for b in self.bodies:
            if b.body_type == Body.GOAL:
                goal = b
            elif b.body_type == Body.PLAYER:
                player = b
        if goal and player:
            dx = abs(player.pos.x - goal.pos.x)
            dy = abs(player.pos.y - goal.pos.y)
            goal.reached = dx < (player.size.x + goal.size.x) * 0.5 and dy < (player.size.y + goal.size.y) * 0.5

    def apply_player_input(self, player, keys, dt):
        tx, ty = 0.0, 0.0
        if keys[K_LEFT] or keys[K_a]:
            tx -= self.INPUT_FORCE
        if keys[K_RIGHT] or keys[K_d]:
            tx += self.INPUT_FORCE
        if keys[K_UP] or keys[K_w]:
            ty -= self.INPUT_FORCE
        if keys[K_DOWN] or keys[K_s]:
            ty += self.INPUT_FORCE
        self._input_force.x += (tx - self._input_force.x) * self.FORCE_RAMP * dt
        self._input_force.y += (ty - self._input_force.y) * self.FORCE_RAMP * dt
        if tx == 0 and ty == 0 and player.vel.length() < 5.0:
            player.vel = Vec2(0, 0)
            self._input_force = Vec2(0, 0)

    def get_boxes(self):
        return [b for b in self.bodies if b.body_type == Body.BOX]

    def get_zones(self):
        return [b for b in self.bodies if b.body_type == Body.ZONE]

    def get_players(self):
        return [b for b in self.bodies if b.body_type == Body.PLAYER]

    def get_goal(self):
        for b in self.bodies:
            if b.body_type == Body.GOAL:
                return b
        return None


# ---------------------------------------------------------------------------
# Level Helpers
# ---------------------------------------------------------------------------
RED = (220, 50, 50)
BLUE = (50, 120, 220)
GREEN = (50, 200, 50)
YELLOW = (255, 200, 0)
PURPLE = (160, 50, 200)
ORANGE = (255, 140, 0)
CYAN = (0, 200, 200)
BROWN = (160, 100, 50)
GRAY = (120, 120, 120)
PINK = (255, 105, 180)
WALL_C = (80, 80, 80)

ARC_COLORS = [
    (0,0,0), (0,116,217), (255,65,54), (46,204,64),
    (255,220,0), (170,170,170), (240,18,190), (255,133,27),
    (127,219,255), (139,69,19)
]

def _walls(w):
    w.add(Body(Vec2(480, 620), Vec2(480, 20), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(480, -20), Vec2(480, 20), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(-20, 320), Vec2(20, 360), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(980, 320), Vec2(20, 360), 0, Body.WALL, WALL_C))

def L_shape(pos, mass, color, name, color_name=""):
    parts = [
        (Vec2(-15, -25), Vec2(15, 25)),
        (Vec2(35, 15), Vec2(35, 15)),
    ]
    return Body(pos, Vec2(70, 50), mass, Body.BOX, color, name=name, color_name=color_name, parts=parts)

def _level_1(w):
    _walls(w)
    w.add(Body(Vec2(100, 500), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="player1"))
    w.add(Body(Vec2(300, 500), Vec2(30, 30), 25.0, Body.BOX, BROWN, name="box1", color_name="brown"))
    w.add(Body(Vec2(800, 500), Vec2(60, 60), 0, Body.ZONE, (0, 200, 0), name="z1", color_name="brown", label="TARGET"))

def _level_2(w):
    _walls(w)
    w.add(Body(Vec2(150, 320), Vec2(20, 20), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(480, 420), Vec2(20, 120), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(200, 320), Vec2(100, 20), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(760, 320), Vec2(100, 20), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(100, 480), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="player1"))
    w.add(Body(Vec2(250, 480), Vec2(28, 28), 18.0, Body.BOX, RED, name="r", color_name="red"))
    w.add(Body(Vec2(350, 480), Vec2(28, 28), 25.0, Body.BOX, BLUE, name="b", color_name="blue"))
    w.add(Body(Vec2(450, 480), Vec2(28, 28), 30.0, Body.BOX, GREEN, name="g", color_name="green"))
    w.add(Body(Vec2(550, 480), Vec2(28, 28), 22.0, Body.BOX, YELLOW, name="y", color_name="yellow"))
    w.add(Body(Vec2(200, 220), Vec2(40, 40), 0, Body.ZONE, RED, name="zr", color_name="red", label="RED"))
    w.add(Body(Vec2(400, 220), Vec2(40, 40), 0, Body.ZONE, BLUE, name="zb", color_name="blue", label="BLUE"))
    w.add(Body(Vec2(600, 220), Vec2(40, 40), 0, Body.ZONE, GREEN, name="zg", color_name="green", label="GREEN"))
    w.add(Body(Vec2(800, 220), Vec2(40, 40), 0, Body.ZONE, YELLOW, name="zy", color_name="yellow", label="YELLOW"))

def _level_3(w):
    _walls(w)
    w.add(Body(Vec2(300, 350), Vec2(20, 100), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(700, 350), Vec2(20, 100), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(500, 200), Vec2(80, 20), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(80, 550), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="player1"))
    w.add(Body(Vec2(130, 550), Vec2(14, 14), 8.0, Body.PLAYER, PINK, name="player2"))
    w.add(Body(Vec2(220, 550), Vec2(35, 35), 30.0, Body.BOX, RED, name="rb", color_name="red"))
    w.add(Body(Vec2(360, 550), Vec2(22, 22), 15.0, Body.BOX, BLUE, name="bb", color_name="blue"))
    w.add(Body(Vec2(500, 550), Vec2(28, 28), 22.0, Body.BOX, GREEN, name="gb", color_name="green"))
    w.add(Body(Vec2(640, 550), Vec2(22, 22), 15.0, Body.BOX, YELLOW, name="yb", color_name="yellow"))
    w.add(Body(Vec2(220, 100), Vec2(35, 35), 0, Body.ZONE, RED, name="zr", color_name="red", label="BIG"))
    w.add(Body(Vec2(420, 100), Vec2(22, 22), 0, Body.ZONE, BLUE, name="zb", color_name="blue", label="SMALL"))
    w.add(Body(Vec2(580, 100), Vec2(28, 28), 0, Body.ZONE, GREEN, name="zg", color_name="green", label="MED"))
    w.add(Body(Vec2(780, 100), Vec2(22, 22), 0, Body.ZONE, YELLOW, name="zy", color_name="yellow", label="SMALL"))

def _level_4(w):
    _walls(w)
    w.add(Body(Vec2(480, 300), Vec2(20, 100), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(100, 550), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="player1"))
    w.add(Body(Vec2(150, 550), Vec2(14, 14), 8.0, Body.PLAYER, PINK, name="player2"))
    w.add(L_shape(Vec2(300, 550), 28.0, BROWN, name="L1", color_name="brown"))
    w.add(Body(Vec2(800, 200), Vec2(70, 70), 0, Body.ZONE, (0, 200, 0), name="z1", color_name="brown", label="TARGET"))

def _level_5(w):
    _walls(w)
    w.add(Body(Vec2(200, 300), Vec2(20, 80), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(600, 450), Vec2(80, 20), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(100, 550), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="player1"))
    w.add(Body(Vec2(150, 550), Vec2(14, 14), 8.0, Body.PLAYER, PINK, name="player2"))
    w.add(L_shape(Vec2(350, 550), 28.0, RED, name="L1", color_name="red"))
    big_zone = Body(Vec2(830, 170), Vec2(70, 60), 0, Body.ZONE, RED, name="zl", color_name="red", label="L-SHAPE")
    big_zone.parts = [
        (Vec2(-15, -20), Vec2(15, 30)),
        (Vec2(35, 15), Vec2(35, 15)),
    ]
    w.add(big_zone)

def _level_copy(w):
    _walls(w)
    w.add(Body(Vec2(480, 300), Vec2(20, 100), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(100, 550), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="player1"))
    w.add(Body(Vec2(150, 550), Vec2(14, 14), 8.0, Body.PLAYER, PINK, name="player2"))
    w.add(Body(Vec2(250, 550), Vec2(28, 28), 22.0, Body.BOX, BROWN, name="main", color_name="brown"))
    w.add(Body(Vec2(400, 550), Vec2(28, 28), 22.0, Body.BOX, GRAY, name="trash"))
    w.add(Body(Vec2(700, 550), Vec2(50, 50), 0, Body.ZONE_COPY, CYAN, name="copy_zone", label="COPY"))
    w.add(Body(Vec2(200, 300), Vec2(50, 50), 0, Body.ZONE_REMOVE, (200, 50, 50), name="remove_zone", label="REMOVE"))
    w.add(Body(Vec2(800, 200), Vec2(60, 60), 0, Body.ZONE, (0, 200, 0), name="target", color_name="brown", label="TARGET"))

def _level_6(w):
    _walls(w)
    w.add(Body(Vec2(480, 250), Vec2(20, 80), 0, Body.WALL, WALL_C))
    w.add(Body(Vec2(100, 550), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="player1"))
    w.add(Body(Vec2(150, 550), Vec2(14, 14), 8.0, Body.PLAYER, PINK, name="player2"))
    w.add(Body(Vec2(250, 550), Vec2(28, 28), 20.0, Body.BOX, RED, name="n1", number=1))
    w.add(Body(Vec2(360, 550), Vec2(28, 28), 20.0, Body.BOX, BLUE, name="n2", number=2))
    w.add(Body(Vec2(470, 550), Vec2(28, 28), 20.0, Body.BOX, GREEN, name="n3", number=3))
    w.add(Body(Vec2(300, 450), Vec2(15, 15), 5.0, Body.BOX, (200,200,200), name="op1", operator="+"))
    w.add(Body(Vec2(410, 450), Vec2(15, 15), 5.0, Body.BOX, (200,200,200), name="op2", operator="+"))
    w.add(Body(Vec2(520, 450), Vec2(15, 15), 5.0, Body.BOX, (200,200,200), name="op3", operator="="))
    z = Body(Vec2(800, 300), Vec2(80, 60), 0, Body.ZONE, (0, 200, 0), name="zsum", label="=?", required_number=6)
    w.add(z)
    w.add(Body(Vec2(800, 100), Vec2(0, 0), 0, Body.TEXT, (255,255,255), label="1 + 2 + 3 = ?"))

def _level_7(w):
    _walls(w)
    w.add(Body(Vec2(100, 550), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="player1"))
    w.add(Body(Vec2(150, 550), Vec2(14, 14), 8.0, Body.PLAYER, PINK, name="player2"))
    w.add(Body(Vec2(480, 60), Vec2(0, 0), 0, Body.TEXT, (255,255,255), label="12 x 3 = 36"))
    w.add(Body(Vec2(250, 550), Vec2(30, 30), 22.0, Body.BOX, ORANGE, name="r1", number=36))
    w.add(Body(Vec2(300, 300), Vec2(60, 40), 0, Body.ZONE, (0,200,0), name="zs1", label="12x3", required_number=36))
    w.add(Body(Vec2(700, 300), Vec2(60, 40), 0, Body.ZONE, (0,200,0), name="zs2", label="RESULT", required_number=36))

def _level_8(w):
    _walls(w)
    w.add(Body(Vec2(100, 550), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="player1"))
    w.add(Body(Vec2(150, 550), Vec2(14, 14), 8.0, Body.PLAYER, PINK, name="player2"))
    w.add(Body(Vec2(480, 60), Vec2(0, 0), 0, Body.TEXT, (255,255,255), label="123 x 2 = 246"))
    w.add(Body(Vec2(250, 550), Vec2(30, 30), 22.0, Body.BOX, PURPLE, name="r1", number=246))
    w.add(Body(Vec2(300, 300), Vec2(60, 40), 0, Body.ZONE, (0,200,0), name="zs1", label="123x2", required_number=246))
    w.add(Body(Vec2(700, 300), Vec2(60, 40), 0, Body.ZONE, (0,200,0), name="zs2", label="RESULT", required_number=246))

def _level_9(w):
    """Two AI agents cooperate."""
    _walls(w)
    w.add(Body(Vec2(100, 550), Vec2(14, 14), 8.0, Body.PLAYER, BLUE, name="ai1"))
    w.add(Body(Vec2(150, 550), Vec2(14, 14), 8.0, Body.PLAYER, PINK, name="ai2"))
    w.add(Body(Vec2(250, 550), Vec2(28, 28), 22.0, Body.BOX, RED, name="r", color_name="red"))
    w.add(Body(Vec2(350, 550), Vec2(28, 28), 22.0, Body.BOX, BLUE, name="b", color_name="blue"))
    w.add(Body(Vec2(450, 550), Vec2(28, 28), 22.0, Body.BOX, GREEN, name="g", color_name="green"))
    w.add(Body(Vec2(200, 120), Vec2(40, 40), 0, Body.ZONE, RED, name="zr", color_name="red", label="RED"))
    w.add(Body(Vec2(400, 120), Vec2(40, 40), 0, Body.ZONE, BLUE, name="zb", color_name="blue", label="BLUE"))
    w.add(Body(Vec2(600, 120), Vec2(40, 40), 0, Body.ZONE, GREEN, name="zg", color_name="green", label="GREEN"))


# ---------------------------------------------------------------------------
# ARC Puzzle Level (Level 10)
# ---------------------------------------------------------------------------
class ARCPuzzle:
    def __init__(self, game):
        self.game = game
        self.grid_size = 14
        self.cell_size = 28
        self.offset_x = 120
        self.offset_y = 120
        self.initial_grid = [[0]*self.grid_size for _ in range(self.grid_size)]
        self._draw_L(self.initial_grid, 4, 4, 1)
        self.grid = [row[:] for row in self.initial_grid]
        self.target_grid = [[0]*self.grid_size for _ in range(self.grid_size)]
        self._draw_L(self.target_grid, 4, 4, 1)
        self._rotate_grid(self.target_grid, 1)
        self.buttons = []
        self._init_buttons()
        self.won = False
        try:
            self.bg_image = pygame.image.load("ls20_level0.png")
        except Exception:
            self.bg_image = None

    def _draw_L(self, grid, cx, cy, color):
        grid[cy][cx] = color
        grid[cy][cx+1] = color
        grid[cy][cx+2] = color
        grid[cy+1][cx] = color
        grid[cy+2][cx] = color

    def _rotate_grid(self, grid, times=1):
        n = len(grid)
        for _ in range(times % 4):
            new_grid = [[0]*n for _ in range(n)]
            for y in range(n):
                for x in range(n):
                    new_grid[x][n-1-y] = grid[y][x]
            grid[:] = new_grid

    def _init_buttons(self):
        bx = 600
        by = 150
        bw = 220
        bh = 60
        gap = 20
        self.buttons = [
            {"rect": pygame.Rect(bx, by, bw, bh), "label": "ROTATE LEFT", "action": lambda: self._rotate_grid(self.grid, 3)},
            {"rect": pygame.Rect(bx, by + bh + gap, bw, bh), "label": "ROTATE RIGHT", "action": lambda: self._rotate_grid(self.grid, 1)},
            {"rect": pygame.Rect(bx, by + 2*(bh+gap), bw, bh), "label": "RESET", "action": self._reset},
            {"rect": pygame.Rect(bx, by + 3*(bh+gap), bw, bh), "label": "SUBMIT", "action": self._submit},
        ]

    def _reset(self):
        self.grid = [row[:] for row in self.initial_grid]
        self.won = False

    def _submit(self):
        if self.grid == self.target_grid:
            self.won = True
        else:
            self.game.level_hint = "NOT YET, TRY AGAIN!"
            self.game.hint_alpha = 255
            self.game.hint_timer = 0.0

    def handle_click(self, pos):
        for btn in self.buttons:
            if btn["rect"].collidepoint(pos):
                btn["action"]()
                return True
        return False

    def update(self, dt):
        pass

    def draw(self, screen, fonts):
        if self.bg_image:
            screen.blit(self.bg_image, (0, 0))
        else:
            screen.fill((30, 30, 40))
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    color = ARC_COLORS[self.grid[y][x]]
                    rect = pygame.Rect(self.offset_x + x*self.cell_size, self.offset_y + y*self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, (60, 60, 60), rect, 1)
            border = pygame.Rect(self.offset_x, self.offset_y, self.grid_size*self.cell_size, self.grid_size*self.cell_size)
            pygame.draw.rect(screen, (255, 255, 255), border, 3)

        for btn in self.buttons:
            r = btn["rect"]
            pygame.draw.rect(screen, (60, 60, 80), r, border_radius=8)
            pygame.draw.rect(screen, (255, 255, 255), r, 3, border_radius=8)
            lbl = fonts["big"].render(btn["label"], True, (255, 255, 255))
            screen.blit(lbl, (r.centerx - lbl.get_width()//2, r.centery - lbl.get_height()//2))

        preview_size = 12
        px = 600
        py = 500
        title = fonts["small"].render("TARGET:", True, (255, 255, 255))
        screen.blit(title, (px, py - 30))
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = ARC_COLORS[self.target_grid[y][x]]
                rect = pygame.Rect(px + x*preview_size, py + y*preview_size, preview_size, preview_size)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (60, 60, 60), rect, 1)


# ---------------------------------------------------------------------------
# Level Definitions
# ---------------------------------------------------------------------------
LEVELS = [
    {"title": "1 First Push", "hint": "push some block to some region", "win_type": "zone_all", "setup": _level_1},
    {"title": "2 Color Sort", "hint": "Push boxes into matching color zones", "win_type": "zone_all", "setup": _level_2},
    {"title": "3 Size Match", "hint": "Match size and color to zones", "win_type": "zone_all", "setup": _level_3},
    {"title": "4 Odd Shape", "hint": "Push the L-shaped box to target", "win_type": "zone_all", "setup": _level_4},
    {"title": "5 Shape Match", "hint": "Push L-shape into L-shaped zone", "win_type": "zone_all", "setup": _level_5},
    {"title": "6 Copy & Remove", "hint": "Remove gray box, copy brown box to target", "win_type": "zone_all", "setup": _level_copy},
    {"title": "7 Addition", "hint": "Push 1, 2, 3 into =? zone (sum=6)", "win_type": "zone_sum", "setup": _level_6},
    {"title": "8 2-Digit Multiply", "hint": "Push 36 to the correct zone", "win_type": "zone_sum", "setup": _level_7},
    {"title": "9 3-Digit Multiply", "hint": "Push 246 to the correct zone", "win_type": "zone_sum", "setup": _level_8},
    {"title": "10 AI Cooperation", "hint": "Two AI agents cooperate to sort colors", "win_type": "zone_all", "setup": _level_9},
]


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class Game:
    def __init__(self, start_level=0):
        pygame.init()
        self.screen = pygame.display.set_mode((960, 640))
        pygame.display.set_caption("Physics Puzzle - Push the Box!")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 36)
        self.big_font = pygame.font.SysFont("monospace", 48)
        self.huge_font = pygame.font.SysFont("monospace", 64)
        self.small_font = pygame.font.SysFont("monospace", 28)
        self.fonts = {"font": self.font, "big": self.big_font, "huge": self.huge_font, "small": self.small_font}
        self.world = World()
        self.level_idx = max(0, min(start_level, len(LEVELS) - 1))
        self.running = True
        self.won = False
        self.hint_alpha = 255
        self.hint_timer = 0.0
        self.arc_puzzle = None
        self.coins = 0
        self._load_level(self.level_idx)

    def _load_level(self, idx):
        self.world.clear()
        self.level_idx = idx
        self.won = False
        self.hint_alpha = 255
        self.hint_timer = 0.0
        self.arc_puzzle = None
        self.zone_progress = {}  # body_name -> 0.0..1.0
        lvl = LEVELS[idx]
        lvl["setup"](self.world)
        # Add uniform control zones at top-right
        self.world.add(Body(Vec2(900, 50), Vec2(40, 20), 0, Body.ZONE_RESET, (200, 50, 50), name="reset_zone", label="RESET"))
        self.world.add(Body(Vec2(900, 100), Vec2(40, 20), 0, Body.ZONE_NEXT, (255, 140, 0), name="next_zone", label="NEXT"))
        self.world.add(Body(Vec2(900, 150), Vec2(40, 20), 0, Body.ZONE_PREV, (100, 100, 255), name="prev_zone", label="PREV"))
        players = self.world.get_players()
        self.player1 = players[0] if len(players) > 0 else None
        self.player2 = players[1] if len(players) > 1 else None
        self.level_title = lvl["title"]
        self.level_hint = lvl["hint"]
        self.win_type = lvl["win_type"]

    def fit_text(self, text, max_w, max_h, color=(255, 255, 255)):
        """Render text scaled to fit within max_w x max_h region."""
        for font in (self.big_font, self.font, self.small_font):
            surf = font.render(text, True, color)
            if surf.get_width() <= max_w and surf.get_height() <= max_h:
                return surf
        surf = self.small_font.render(text, True, color)
        ratio = min(max_w / max(surf.get_width(), 1), max_h / max(surf.get_height(), 1))
        new_w = max(1, int(surf.get_width() * ratio))
        new_h = max(1, int(surf.get_height() * ratio))
        return pygame.transform.smoothscale(surf, (new_w, new_h))

    def _start_arc_level(self):
        self.arc_puzzle = ARCPuzzle(self)
        self.level_title = "Level 11: ARC ls20 Rotation"
        self.level_hint = "Click buttons to rotate pattern to match target"
        self.win_type = "arc"

    def _next_level(self):
        if self.level_idx + 1 < len(LEVELS):
            self._load_level(self.level_idx + 1)
        elif self.arc_puzzle is None:
            self._start_arc_level()
        else:
            self.level_idx = 0
            self._load_level(0)
            self.arc_puzzle = None

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            dt = min(dt, 0.05)
            self._handle_events()
            self._update(dt)
            self._draw()
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
            if event.type == MOUSEBUTTONDOWN and self.arc_puzzle and event.button == 1:
                if self.arc_puzzle.handle_click(event.pos):
                    if self.arc_puzzle.won:
                        self.won = True

    def _update(self, dt):
        if self.arc_puzzle:
            self.arc_puzzle.update(dt)
            return

        keys = pygame.key.get_pressed()
        if self.player1:
            self.world.apply_player_input(self.player1, keys, dt)
        if self.player2:
            self._ai_update(self.player2, dt)

        if self.level_idx == 9 and self.player1:
            self._ai_update(self.player1, dt)

        self.world.step(dt)

        # Per-box coin reward when a box enters the correct zone
        for b in self.world.bodies:
            if b.body_type == Body.BOX and b.in_zone and not b.zone_rewarded:
                b.zone_rewarded = True
                self.coins += 5
                for p in self.world.get_players():
                    if (p.pos - b.pos).length_sq() < 200 * 200:
                        p.floating_text = "+5 COINS!"
                        p.floating_timer = 2.0
                        break

        # Progress-bar logic for RESET / NEXT / PREV zones
        FILL_TIME = 1.2  # seconds to fill progress bar
        for z in self.world.bodies:
            if z.body_type not in (Body.ZONE_RESET, Body.ZONE_NEXT, Body.ZONE_PREV):
                continue
            inside = False
            for p in self.world.get_players():
                if p.center_in_zone(z):
                    inside = True
                    break
            key = z.name
            if inside:
                self.zone_progress[key] = self.zone_progress.get(key, 0.0) + dt / FILL_TIME
                if self.zone_progress[key] >= 1.0:
                    self.zone_progress[key] = 0.0
                    if z.body_type == Body.ZONE_RESET:
                        self._load_level(self.level_idx)
                        return
                    elif z.body_type == Body.ZONE_NEXT:
                        self._next_level()
                        return
                    elif z.body_type == Body.ZONE_PREV:
                        if self.level_idx > 0:
                            self._load_level(self.level_idx - 1)
                        return
            else:
                self.zone_progress[key] = 0.0

        if not self.won:
            if self._check_win():
                self.won = True
                reward = 10 * (self.level_idx + 1)
                self.coins += reward
                for p in self.world.get_players():
                    p.floating_text = f"+{reward} COINS!"
                    p.floating_timer = 3.0

    def _ai_update(self, ai, dt):
        boxes = [b for b in self.world.bodies if b.body_type == Body.BOX and not b.in_zone]
        if not boxes:
            ai.floating_text = "ALL DONE!"
            ai.vel = ai.vel * 0.9
            return

        target_box = min(boxes, key=lambda b: (b.pos - ai.pos).length_sq())
        zones = [z for z in self.world.bodies if z.body_type == Body.ZONE]
        target_zone = None
        for z in zones:
            if z.color_name and z.color_name == target_box.color_name:
                target_zone = z
                break
            if z.required_number is not None and target_box.number == z.required_number:
                target_zone = z
                break

        if target_zone is None:
            for z in zones:
                if z.required_number is None and not z.color_name:
                    target_zone = z
                    break

        if target_zone:
            dist_to_box = (target_box.pos - ai.pos).length()
            if dist_to_box < 70:
                dir_to_zone = (target_zone.pos - target_box.pos).normalized()
                ai.apply_force(dir_to_zone * self.world.INPUT_FORCE * 0.7)
                dir_to_box = (target_box.pos - ai.pos).normalized()
                ai.apply_force(dir_to_box * self.world.INPUT_FORCE * 0.3)
                ai.floating_text = "PUSH!"
                ai.floating_timer = 3.0
            else:
                dir_to_box = (target_box.pos - ai.pos).normalized()
                ai.apply_force(dir_to_box * self.world.INPUT_FORCE * 0.8)
                ai.floating_text = "I WILL HELP"
                ai.floating_timer = 3.0
        else:
            ai.floating_text = "FINDING BOX"
            ai.floating_timer = 3.0

        for other in self.world.get_players():
            if other is ai:
                continue
            if (other.pos - ai.pos).length() < 50:
                if ai.floating_text == "I WILL HELP":
                    ai.floating_text = "TOGETHER!"
                    ai.floating_timer = 3.0
                elif other.floating_text == "ALL DONE!":
                    ai.floating_text = "GREAT!"
                    ai.floating_timer = 3.0

    def _check_win(self):
        boxes = self.world.get_boxes()
        if not boxes:
            return False
        zones = self.world.get_zones()

        if self.win_type == "zone":
            return any(b.in_zone for b in boxes)
        elif self.win_type == "zone_all":
            return all(b.in_zone for b in boxes)
        elif self.win_type == "zone_sum":
            for z in zones:
                if z.required_number is None:
                    continue
                inside = [b for b in boxes if b.center_in_zone(z)]
                s = sum(b.number or 0 for b in inside)
                if s != z.required_number:
                    return False
            return all(any(b.center_in_zone(z) for z in zones) for b in boxes if b.number is not None)
        elif self.win_type == "goal":
            goal = self.world.get_goal()
            return goal is not None and goal.reached
        return False

    def _draw_floating_text(self, b):
        if not b.floating_text:
            return
        txt_surf = self.small_font.render(b.floating_text, True, (255, 255, 255))
        padding = 8
        box_w = txt_surf.get_width() + padding * 2
        box_h = txt_surf.get_height() + padding * 2
        top_y = int(b.pos.y - b.size.y - box_h - 10)
        left_x = int(b.pos.x - box_w / 2)
        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((30, 30, 30, 200))
        self.screen.blit(bg, (left_x, top_y))
        pygame.draw.rect(self.screen, (255, 255, 255), (left_x, top_y, box_w, box_h), 2, border_radius=6)
        self.screen.blit(txt_surf, (left_x + padding, top_y + padding))

    def _draw(self):
        self.screen.fill((30, 30, 40))

        if self.arc_puzzle:
            self.arc_puzzle.draw(self.screen, self.fonts)
            pygame.display.flip()
            return

        for b in self.world.bodies:
            if b.body_type == Body.TEXT:
                txt = self.big_font.render(b.label, True, b.color)
                self.screen.blit(txt, (int(b.pos.x - txt.get_width() / 2), int(b.pos.y - txt.get_height() / 2)))

        for b in self.world.bodies:
            if b.body_type not in Body.SENSOR_TYPES:
                continue
            is_func = b.body_type in (Body.ZONE_COPY, Body.ZONE_REMOVE, Body.ZONE_RESET, Body.ZONE_NEXT, Body.ZONE_PREV)
            for off, sz in b.parts:
                cx = int(b.pos.x + off.x)
                cy = int(b.pos.y + off.y)
                rect = pygame.Rect(cx - int(sz.x), cy - int(sz.y), int(sz.x * 2), int(sz.y * 2))
                if is_func:
                    # Functional zones: dark gray
                    pygame.draw.rect(self.screen, (60, 60, 80), rect, border_radius=4)
                    pygame.draw.rect(self.screen, (200, 200, 220), rect, 2, border_radius=4)
                else:
                    # Non-functional zones: hollow outline only
                    pygame.draw.rect(self.screen, b.color, rect, 5, border_radius=4)
            if b.label and is_func:
                zone_w = b.size.x * 2
                zone_h = b.size.y * 2
                lbl = self.fit_text(b.label, zone_w - 8, zone_h - 8)
                self.screen.blit(lbl, (int(b.pos.x - lbl.get_width() / 2), int(b.pos.y - lbl.get_height() / 2)))
            # Progress bar for functional control zones
            if is_func:
                prog = self.zone_progress.get(b.name, 0.0)
                if prog > 0:
                    bar_w = b.size.x * 2
                    bar_h = 10
                    bar_x = int(b.pos.x - bar_w / 2)
                    bar_y = int(b.pos.y + b.size.y + 4)
                    pygame.draw.rect(self.screen, (40, 40, 50), (bar_x, bar_y, bar_w, bar_h), border_radius=0)
                    fill_w = max(1, int(bar_w * prog))
                    pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, fill_w, bar_h), border_radius=0)

        for b in self.world.bodies:
            if b.body_type == Body.BUTTON:
                color = (50, 220, 50) if b.pressed else (200, 50, 50)
                for off, sz in b.parts:
                    cx = int(b.pos.x + off.x)
                    cy = int(b.pos.y + off.y)
                    rect = pygame.Rect(cx - int(sz.x), cy - int(sz.y), int(sz.x * 2), int(sz.y * 2))
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=4)
            elif b.body_type == Body.GOAL:
                for off, sz in b.parts:
                    cx = int(b.pos.x + off.x)
                    cy = int(b.pos.y + off.y)
                    rect = pygame.Rect(cx - int(sz.x), cy - int(sz.y), int(sz.x * 2), int(sz.y * 2))
                    pygame.draw.rect(self.screen, (255, 215, 0), rect, border_radius=8)
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 3, border_radius=8)
                lbl = self.fit_text("GOAL", b.size.x * 2 - 6, b.size.y * 2 - 6, (0, 0, 0))
                self.screen.blit(lbl, (int(b.pos.x - lbl.get_width() / 2), int(b.pos.y - lbl.get_height() / 2)))
            elif b.body_type == Body.WALL:
                for off, sz in b.parts:
                    cx = int(b.pos.x + off.x)
                    cy = int(b.pos.y + off.y)
                    rect = pygame.Rect(cx - int(sz.x), cy - int(sz.y), int(sz.x * 2), int(sz.y * 2))
                    pygame.draw.rect(self.screen, b.color, rect)
                    pygame.draw.rect(self.screen, (60, 60, 60), rect, 2)
            elif b.body_type in (Body.PLAYER, Body.BOX):
                for off, sz in b.parts:
                    cx = int(b.pos.x + off.x)
                    cy = int(b.pos.y + off.y)
                    rect = pygame.Rect(cx - int(sz.x), cy - int(sz.y), int(sz.x * 2), int(sz.y * 2))
                    pygame.draw.rect(self.screen, b.color, rect, border_radius=0)
                if b.body_type == Body.BOX:
                    box_w = b.size.x * 2
                    box_h = b.size.y * 2
                    if b.number is not None:
                        num_txt = self.fit_text(str(b.number), box_w - 6, box_h - 6)
                        self.screen.blit(num_txt, (int(b.pos.x - num_txt.get_width() / 2), int(b.pos.y - num_txt.get_height() / 2)))
                    elif b.operator:
                        op_txt = self.fit_text(b.operator, box_w - 6, box_h - 6)
                        self.screen.blit(op_txt, (int(b.pos.x - op_txt.get_width() / 2), int(b.pos.y - op_txt.get_height() / 2)))

            self._draw_floating_text(b)

        # Top-left info panel: LEVEL / GOAL / COINS
        x_off = 10
        y_off = 10
        line_h = 32
        lvl_num = self.level_title.split()[0] if self.level_title else "?"
        lvl_txt = self.small_font.render(f"LEVEL: {lvl_num}", True, (255, 255, 255))
        self.screen.blit(lvl_txt, (x_off, y_off))
        y_off += line_h
        hint_txt = self.small_font.render(f"HINT: {self.level_hint}", True, (200, 255, 200))
        self.screen.blit(hint_txt, (x_off, y_off))
        y_off += line_h
        coin_txt = self.small_font.render(f"COINS: {self.coins}", True, (255, 220, 0))
        self.screen.blit(coin_txt, (x_off, y_off))

        pygame.display.flip()


if __name__ == "__main__":
    import sys
    start_level = 0
    if len(sys.argv) > 1:
        try:
            start_level = int(sys.argv[1]) - 1
        except ValueError:
            pass
    game = Game(start_level=start_level)
    game.run()
