import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import pyglet

import matplotlib.pyplot as plt
import torch
import random
from shapely.geometry import Polygon, Point
from shapely.affinity import affine_transform
from shapely.ops import triangulate

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# To see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v2
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.lander==contact.fixtureA.body or self.env.lander==contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

class LunarLanderReachability(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.param_dict = self._generate_param_dict({})
        self.initialize_simulator_variables(self.param_dict)
        self.bounds_simulation_one_player = np.array([
            [0, self.W],
            [0, self.H],
            [-self.vx_bound, self.vx_bound],
            [-self.vy_bound, self.vy_bound],
            [-self.theta_bound, self.theta_bound],
            [-self.theta_dot_bound, self.theta_dot_bound]])

        self.reset()

    def _generate_param_dict(self, input_dict):
        param_dict = {}

        param_dict["FPS"] = 50
        param_dict["SCALE"] = 30.0
        param_dict["MAIN_ENGINE_POWER"] = 13.0
        param_dict["SIDE_ENGINE_POWER"] = 0.6
        param_dict["LANDER_POLY"] = [
            (-14, +17), (-17, 0), (-17, -10),
            (+17, -10), (+17, 0), (+14, +17)
            ]
        param_dict["LEG_AWAY"] = 20
        param_dict["LEG_DOWN"] = 18
        param_dict["LEG_W"] = 2
        param_dict["LEG_H"] = 8
        param_dict["LEG_SPRING_TORQUE"] = 40
        param_dict["SIDE_ENGINE_HEIGHT"] = 14.0
        param_dict["SIDE_ENGINE_AWAY"] = 12.0
        param_dict["VIEWPORT_W"] = 600
        param_dict["VIEWPORT_H"] = 400
        param_dict["CHUNKS"] = 17
        param_dict["INITIAL_RANDOM"] = 1000.0
        param_dict["LIDAR_RANGE"] = 500

        param_dict["vx_bound"] = 10
        param_dict["vy_bound"] = 10
        param_dict["theta_bound"] = np.radians(45)
        param_dict["theta_dot_bound"] = np.radians(10)

        for key, value in input_dict:
            param_dict[key] = value
        return param_dict

    def initialize_simulator_variables(self, param_dict):
        self.FPS = param_dict["FPS"]
        self.SCALE = param_dict["SCALE"]   # affects how fast-paced the game is, forces should be adjusted as well
        self.MAIN_ENGINE_POWER = param_dict["MAIN_ENGINE_POWER"]
        self.SIDE_ENGINE_POWER = param_dict["SIDE_ENGINE_POWER"]
        self.LANDER_POLY = param_dict["LANDER_POLY"]
        self.LEG_AWAY = param_dict["LEG_AWAY"]
        self.LEG_DOWN = param_dict["LEG_DOWN"]
        self.LEG_W = param_dict["LEG_W"]
        self.LEG_H = param_dict["LEG_H"]
        self.LEG_SPRING_TORQUE = param_dict["LEG_SPRING_TORQUE"]
        self.SIDE_ENGINE_HEIGHT = param_dict["SIDE_ENGINE_HEIGHT"]
        self.SIDE_ENGINE_AWAY = param_dict["SIDE_ENGINE_AWAY"]
        self.VIEWPORT_W = param_dict["VIEWPORT_W"]
        self.VIEWPORT_H = param_dict["VIEWPORT_H"]
        self.CHUNKS = param_dict["CHUNKS"]
        self.INITIAL_RANDOM = param_dict["INITIAL_RANDOM"]
        self.LIDAR_RANGE = param_dict["LIDAR_RANGE"] / self.SCALE

        self.W = self.VIEWPORT_W / self.SCALE
        self.H = self.VIEWPORT_H / self.SCALE
        self.HELIPAD_Y = self.H / 2
        # height of lander body in simulator self.SCALE. self.LANDER_POLY has the (x,y) points that define the
        # shape of the lander in pixel self.SCALE
        self.LANDER_POLY_X = np.array(self.LANDER_POLY)[:, 0]
        self.LANDER_POLY_Y = np.array(self.LANDER_POLY)[:, 1]
        self.LANDER_W = (np.max(
            self.LANDER_POLY_X) - np.min(self.LANDER_POLY_X)) / self.SCALE
        self.LANDER_H = (np.max(
            self.LANDER_POLY_Y) - np.min(self.LANDER_POLY_Y)) / self.SCALE
        # distance of edge of legs from center of lander body in simulator self.SCALE
        self.LEG_X_DIST = self.LEG_AWAY / self.SCALE
        self.LEG_Y_DIST = self.LEG_DOWN / self.SCALE
        # radius around lander to check for collisions
        self.LANDER_RADIUS = (
            (self.LANDER_H / 2 + self.LEG_Y_DIST +
                self.LEG_H / self.SCALE) ** 2 +
            (self.LANDER_W / 2 + self.LEG_X_DIST +
                self.LEG_W / self.SCALE) ** 2) ** 0.5

        # set up state space bounds used in evaluating the q value function
        self.vx_bound = param_dict["vx_bound"]
        self.vy_bound = param_dict["vy_bound"]
        self.theta_bound = param_dict["theta_bound"]
        self.theta_dot_bound = param_dict["theta_dot_bound"]

        self.chunk_x = [self.W/(self.CHUNKS-1)*i for i in range(self.CHUNKS)]
        self.chunk_y = [self.H/(self.CHUNKS-1)*i for i in range(self.CHUNKS)]

        self.helipad_x1 = self.chunk_x[self.CHUNKS//2-1]
        self.helipad_x2 = self.chunk_x[self.CHUNKS//2+1]

        self.hover_min_y_dot = -0.1
        self.hover_max_y_dot = 0.1
        self.hover_min_x_dot = -0.1
        self.hover_max_x_dot = 0.1

        self.land_min_v = -1.6  # fastest that lander can be falling when it hits the ground

        self.theta_hover_max = np.radians(15.0)  # most the lander can be tilted when landing
        self.theta_hover_min = np.radians(-15.0)

        self.midpoint_x = self.W / 2
        self.width_x = self.W

        self.midpoint_y = self.H / 2
        self.width_y = self.H

        self.hover_min_x = self.W / (self.CHUNKS - 1) * (self.CHUNKS // 2 - 1)
        self.hover_max_x = self.W / (self.CHUNKS - 1) * (self.CHUNKS // 2 + 1)
        self.hover_min_y = self.HELIPAD_Y  # calc of edges of landing pad based
        self.hover_max_y = self.HELIPAD_Y + 2  # on calc in parent reset()

        self.polygon_target = [
            (self.helipad_x1, self.HELIPAD_Y),
            (self.helipad_x2, self.HELIPAD_Y),
            (self.helipad_x2, self.HELIPAD_Y + 2),
            (self.helipad_x1, self.HELIPAD_Y + 2),
            (self.helipad_x1, self.HELIPAD_Y)]
        self.target_xy_polygon = Polygon(self.polygon_target)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        self.generate_terrain()

        initial_y = VIEWPORT_H/SCALE
        initial_state = self.rejection_sample()
        assert isinstance(initial_state[0], np.float64), "Float64!"
        initial_x = initial_state[0]  # self.VIEWPORT_W/self.SCALE/2
        initial_y = initial_state[1]
        self.lander = self.world.CreateDynamicBody(
            position = (initial_x, initial_y),
            angle=0.0,
            linearVelocity=(initial_state[2], initial_state[3]),
            angularVelocity=initial_state[5],
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.lander.color1 = (0.5,0.4,0.9)
        self.lander.color2 = (0.3,0.3,0.5)
        # self.lander.ApplyForceToCenter( (
        #     self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
        #     self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        #     ), True)

        self.legs = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (initial_x  - i*LEG_AWAY/SCALE, initial_y),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5,0.4,0.9)
            leg.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self.step(np.array([0,0]) if self.continuous else 0)[0]

    def generate_terrain(self, terrain_polyline=None):
        # terrain
        if terrain_polyline is None:
            height = np.ones((self.CHUNKS+1,))
        else:
            height = terrain_polyline
        height[self.CHUNKS//2-3] = self.HELIPAD_Y + 2.5
        height[self.CHUNKS//2-2] = self.HELIPAD_Y
        height[self.CHUNKS//2-1] = self.HELIPAD_Y
        height[self.CHUNKS//2+0] = self.HELIPAD_Y
        height[self.CHUNKS//2+1] = self.HELIPAD_Y
        height[self.CHUNKS//2+2] = self.HELIPAD_Y
        # smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(self.CHUNKS)]
        smooth_y = list(height[:-1])
        # print(smooth_y)
        # assert len(smooth_y) == len(height)
        # smooth_y = list(height)

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (self.W, 0)]))
        self.sky_polys = []
        self.moon_chunk = []
        obstacle_polyline = [(self.chunk_x[0], smooth_y[0])]
        for i in range(self.CHUNKS-1):
            p1 = (self.chunk_x[i], smooth_y[i])
            p2 = (self.chunk_x[i+1], smooth_y[i+1])
            obstacle_polyline.append(p2)
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], self.H), (p1[0], self.H)])
            self.moon_chunk.append([p1, p2, (p2[0], 0), (p1[0], 0)])
        # Enclose terrain within window.
        obstacle_polyline.append((self.W, self.H))
        obstacle_polyline.append((0, self.H))
        obstacle_polyline.append(obstacle_polyline[0])
        self.obstacle_polyline = Polygon(obstacle_polyline)
        self.paint_obstacle = obstacle_polyline

        self.moon.color1 = (0.6, 0.6, 0.6)
        self.moon.color2 = (0.6, 0.6, 0.6)

    def rejection_sample(self, sample_inside_obs=False, zero_vel=False):
        flag_sample = False
        # Repeat sampling until outside obstacle if needed.
        while True:
            xy_sample = np.random.uniform(low=[0,
                                               0],
                                          high=[self.W,
                                                self.H])
            p = Point(xy_sample[0], xy_sample[1])
            flag_sample = self.obstacle_polyline.contains(p)
            if flag_sample is True:
                break
            if sample_inside_obs:
                break
        # Sample within simulation space bounds.
        state_in = np.random.uniform(
            low=self.bounds_simulation_one_player[:, 0],
            high=self.bounds_simulation_one_player[:, 1])
        state_in[:2] = xy_sample
        # If zero_vel active, remove any initial rates.
        if zero_vel:
            state_in[[2, 3, -1]] = 0.0
        return np.float64(state_in)

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x,y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0,0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl<0):
            self.world.DestroyBody(self.particles.pop(0))

    def target_margin(self, state):
        # States come in sim_space.
        if not self.parent_init:
            return 0
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        speed = np.sqrt(vx**2 + vy**2)
        p = Point(x, y)
        L2_distance = self.target_xy_polygon.exterior.distance(p)
        inside = 2*self.target_xy_polygon.contains(p) - 1
        vel_l = speed - self.vy_bound/10.0
        # return max(-inside*L2_distance, vel_l)
        return -inside*L2_distance

    def safety_margin(self, state):
        # States come in sim_space.
        if not self.parent_init:
            return 0
        x = state[0]
        y = state[1]
        p = Point(x, y)
        L2_distance = self.obstacle_polyline.exterior.distance(p)
        inside = 2*self.obstacle_polyline.contains(p) - 1
        return -inside*L2_distance

    def parent_step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0]);
        # dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        dispersion = [0.0 ,0.0]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action==2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power>=0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)    # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse(           ( ox*MAIN_ENGINE_POWER*m_power,  oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*MAIN_ENGINE_POWER*m_power, -oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1,3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power>=0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0]*17/SCALE, self.lander.position[1] + oy + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(           ( ox*SIDE_ENGINE_POWER*s_power,  oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        unscaled_state = [
            pos.x,
            pos.y,
            vel.x,
            vel.y,
            self.lander.angle,
            self.lander.angularVelocity,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
            ]
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.HELIPAD_Y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
            ]
        assert len(state)==8

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]   # And ten points for legs contact, the idea is if you
                                                              # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done   = True
            reward = -100
        if not self.lander.awake:
            done   = True
            reward = +100

        self.l_x = self.target_margin(unscaled_state)
        self.g_x = self.safety_margin(unscaled_state)
        info = {"g_x": self.g_x, "l_x": self.l_x}
        return np.array(state, dtype=np.float32), reward, done, info

    def step(self, action):

        state, _, done, info = self.parent_step(action)

        fail = self.g_x > 0
        success = self.l_x <= 0
        reward = 1.0 * success - 5.0 * fail

        return state, reward, done, info

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
            obj.color2 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0,0,0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.HELIPAD_Y
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(1,1,1) )
            self.viewer.draw_polygon( [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)], color=(0.8,0.8,0) )

        self.viewer.draw_polyline(self.paint_obstacle, color=(1, 0, 0), linewidth=10)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class LunarLanderContinuous(LunarLanderReachability):
    continuous = True

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
        elif angle_todo < -0.05: a = 3
        elif angle_todo > +0.05: a = 1
    return a

def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    return total_reward


if __name__ == '__main__':
    demo_heuristic_lander(LunarLander(), render=True)


