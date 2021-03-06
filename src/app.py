#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import statistics
import numpy as np
from math import sqrt
from collections import deque
from scipy.spatial import ConvexHull

from state import State
from framework import Keys
from ui import UserInterface
from bridge import create_bridge
from renderer import PygletFramework as Framework

from Box2D import (b2CircleShape, b2EdgeShape,
                   b2FixtureDef, b2PolygonShape,
                   b2_pi)

def generate_random_vertices():
    points = np.random.rand(30, 2) * 3.0
    hull = ConvexHull(points)
    return list(zip(points[hull.vertices,0], points[hull.vertices,1]))

def get_random_car_gene():
    return {
        'wheel_friction': random.random(),
        'wheel_radius_1': random.random(),
        'wheel_radius_2': random.random(),
        'density': random.random() * 5.0,
        'wheel_torques': [random.random() * 20, random.random() * 20],
        'wheel_drives': [True, False],
        'motor_speeds': [random.random() * -20.0, random.random() * -20.0],
        'hz': random.random() * 15.0,
        'zeta': random.random(),
        'vertices': generate_random_vertices()
    }

def create_car(world, offset, wheel_radius_1, wheel_radius_2,
               wheel_separation, density=1.0, wheel_friction=0.9,
               scale=(1.0, 1.0), chassis_vertices=None,
               wheel_axis=(0.0, 1.0), wheel_torques=[20.0, 10.0],
               wheel_drives=[True, False], hz=4.0, zeta=0.7,
               motor_speeds=[0.0, 0.0], **kwargs):
    """
    """
    x_offset, y_offset = offset
    scale_x, scale_y = scale
    if chassis_vertices is None:
        chassis_vertices = [
            (-1.5, -0.5),
            (1.5, -0.5),
            (1.5, 0.0),
            (0.0, 0.9),
            (-1.15, 0.9),
            (-1.5, 0.2),
        ]

    chassis_vertices = [(scale_x * x, scale_y * y)
                        for x, y in chassis_vertices]
    radius_scale = sqrt(scale_x ** 2 + scale_y ** 2)
    wheel_radius_1 *= radius_scale
    wheel_radius_2 *= radius_scale

    chassis = world.CreateDynamicBody(
        position=(x_offset, y_offset),
        fixtures=b2FixtureDef(
            shape=b2PolygonShape(vertices=chassis_vertices),
            density=density,
            groupIndex=-1
        )
    )

    wheels = []
    springs = []
    wheel_1_pos = chassis_vertices[0]
    wheel_2_pos = chassis_vertices[-1]

    wheel = world.CreateDynamicBody(
        position=(x_offset + wheel_1_pos[0], y_offset + wheel_1_pos[1]),
        fixtures=b2FixtureDef(
            shape=b2CircleShape(radius=wheel_radius_1),
            density=density,
            friction=0.9,
            groupIndex=-1
        )
    )

    spring = world.CreateWheelJoint(
        bodyA=chassis,
        bodyB=wheel,
        anchor=wheel.position,
        axis=wheel_axis,
        motorSpeed=motor_speeds[0],
        maxMotorTorque=wheel_torques[0],
        enableMotor=wheel_drives[0],
        frequencyHz=hz,
        dampingRatio=zeta
    )
    wheels.append(wheel)
    springs.append(spring)

    wheel = world.CreateDynamicBody(
        position=(x_offset + wheel_2_pos[0], y_offset + wheel_2_pos[1]),
        fixtures=b2FixtureDef(
            shape=b2CircleShape(radius=wheel_radius_2),
            density=density,
            friction=0.9,
            groupIndex=-1
        )
    )

    spring = world.CreateWheelJoint(
        bodyA=chassis,
        bodyB=wheel,
        anchor=wheel.position,
        axis=wheel_axis,
        motorSpeed=motor_speeds[1],
        maxMotorTorque=wheel_torques[1],
        enableMotor=wheel_drives[1],
        frequencyHz=hz,
        dampingRatio=zeta
    )
    wheels.append(wheel)
    springs.append(spring)

    return chassis, wheels, springs

def car_from_gene(world, gene):
    return create_car(world,
        offset=(0.0, 1.0),
        wheel_radius_1=gene['wheel_radius_1'],
        wheel_radius_2=gene['wheel_radius_2'],
        wheel_drives=gene['wheel_drives'],
        wheel_friction=gene['wheel_friction'],
        chassis_vertices=gene['vertices'],
        wheel_separation=2.0,
        motor_speeds=gene['motor_speeds'],
        scale=(1, 1))

def destroy_car(world, car):
    world.DestroyBody(car['car'])
    world.DestroyBody(car['wheels'][0])
    world.DestroyBody(car['wheels'][1])

class App (Framework):
    name = "Car"
    description = "Keys: left = a, brake = s, right = d, hz down = q, hz up = e"
    hz = 4
    zeta = 0.7
    speed = 5
    bridgePlanks = 20

    def __init__(self):
        super(App, self).__init__()

        # The ground -- create some terrain
        ground = self.world.CreateStaticBody(
            shapes=b2EdgeShape(vertices=[(-20, 0), (20, 0)])
        )

        x, y1, dx = 20, 0, 5
        vertices = [0.25, 1, 4, 0, 0, -1, -2, -2, -1.25, 0]
        for y2 in vertices * 2:  # iterate through vertices twice
            ground.CreateEdgeFixture(
                vertices=[(x, y1), (x + dx, y2)],
                density=0,
                friction=0.8,
            )
            y1 = y2
            x += dx

        x_offsets = [0, 80, 40, 20, 40]
        x_lengths = [40, 40, 10, 40, 0]
        y2s = [0, 0, 5, 0, 20]

        for x_offset, x_length, y2 in zip(x_offsets, x_lengths, y2s):
            x += x_offset
            ground.CreateEdgeFixture(
                vertices=[(x, 0), (x + x_length, y2)],
                density=0,
                friction=0.6,
            )

        # Teeter
        body = self.world.CreateDynamicBody(
            position=(140, 0.90),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(10, 0.25)),
                density=1.0,
            )
        )

        self.world.CreateRevoluteJoint(
            bodyA=ground,
            bodyB=body,
            anchor=body.position,
            lowerAngle=-8.0 * b2_pi / 180.0,
            upperAngle=8.0 * b2_pi / 180.0,
            enableLimit=True,
        )

        # Bridge
        create_bridge(self.world, ground, (2.0, 0.25),
                      (161.0, -0.125), self.bridgePlanks)

        # Boxes
        for y_pos in [0.5, 1.5, 2.5, 3.5, 4.5]:
            self.world.CreateDynamicBody(
                position=(230, y_pos),
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(box=(0.5, 0.5)),
                    density=0.5,
                )
            )

        State.cars = self.add_cars()
        self.ui = UserInterface(self.window)

    def add_cars(self):
        cars = []
        for i in range(10):
            gene = get_random_car_gene()
            car_id = i
            car, wheels, springs = car_from_gene(self.world, gene)
            cars.append({
                'car': car,
                'wheels': wheels,
                'springs': springs,
                'gene': gene,
                'id': car_id,
                'alive': True,
                'position_hist': deque(250*[0], 250),
                'mean_pos': 0.0,
                'life': 100
            })
        return cars

    def start_new_generation(self):
        for car in State.cars:
            destroy_car(self.world, car)
        State.cars = self.add_cars()
        State.generation += 1
       
    def Keyboard(self, key):
        if key == Keys.K_a:
            self.springs[0].motorSpeed += self.speed
        elif key == Keys.K_s:
            self.springs[0].motorSpeed = 0
        elif key == Keys.K_d:
            self.springs[0].motorSpeed += -self.speed
        elif key in (Keys.K_q, Keys.K_e):
            if key == Keys.K_q:
                self.hz = max(0, self.hz - 1.0)
            else:
                self.hz += 1.0

            for spring in self.springs:
                spring.springFrequencyHz = self.hz
        elif key == Keys.K_r:
            self.start_new_generation()

    def sort_cars_by_score(self):
        State.cars  = sorted(State.cars, key=lambda item: item['car'].position.x, reverse=True)

    def track_car_positions(self):
        for car in State.cars:
            car['position_hist'].appendleft(car['car'].position.x)

    def compute_car_mean_pos_and_life(self):
        for car in State.cars:
            mean_pos = statistics.mean(car['position_hist'])
            if (mean_pos - car['mean_pos']) < 1e-03 and car['life'] > 0:
                car['life'] -= 1
            car['mean_pos'] = mean_pos

    def deactivate_dead_cars(self):
        for car in State.cars:
            if car['life'] == 0:
                car['car'].active = False
                car['wheels'][0].active = False
                car['wheels'][1].active = False

    def num_active_cars(self):
        num_active_cars = 0
        for car in State.cars:
            if car['car'].active:
                num_active_cars += 1
        return num_active_cars

    def Step(self, settings):
        super(App, self).Step(settings)
        self.sort_cars_by_score()
        self.track_car_positions()
        self.compute_car_mean_pos_and_life()
        self.deactivate_dead_cars()
        best_car_pos = State.cars[0]['car'].position.x
        self.viewCenter = (best_car_pos, 20)
        if self.num_active_cars() == 0:
            self.start_new_generation()
        self.ui.update()
        self.ui.render()

if __name__ == "__main__":
    sim = App()
    sim.run()
