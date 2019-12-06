# Parametric Bipedal Walker environment
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements. There's no coordinates
# in the state vector.
#
# Initially Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
# Modified by Rémy Portelas, taking inspiration from https://eng.uber.com/poet-open-ended-deep-learning/

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
import math

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
            self.env.head_contact = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True
    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False

def Rotate2D(pts,cnt,ang=np.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    m1 = pts-cnt
    m2 = np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]])
    return np.dot(m1,m2)+cnt

FPS = 50
class BipedalWalkerContinuous(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        EzPickle.__init__(self)
        
        # Set environment's constants:
        self.SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
        self.VIEWPORT_W = 600  # 1000 for vizu
        self.VIEWPORT_H = 400  # 300 for vizu
        self.TERRAIN_STEP = 14 / self.SCALE
        self.TERRAIN_LENGTH = 200  # in steps
        self.TERRAIN_HEIGHT = self.VIEWPORT_H / self.SCALE / 4
        self.TERRAIN_GRASS = 10  # how long are grass spots, in steps
        self.TERRAIN_STARTPAD = 20  # in steps
        self.FRICTION = 2.5
        self.MOTORS_TORQUE = 80
        self.SPEED_HIP = 4
        self.SPEED_KNEE = 6
        self.NB_LIDAR = 10
        self.LIDAR_RANGE = 160 / self.SCALE
        self.INITIAL_RANDOM = 5
        
        # Set walker's shape constants:
        self.HULL_POLY = [
            (-30, +9), (+6, +9), (+34, +1),
            (+34, -8), (-30, -8)
        ]
        self.HULL_WIDTH = 34 + 30
        self.LEG_DOWN = -8 / self.SCALE
        self.LEG_W, self.LEG_H = 8 / self.SCALE, 34 / self.SCALE
        self.nb_leg_pairs = 1
        self.leg_pairs_x = [0]
        
        # Create Box2D fixture definition for walker
        self.HULL_FD = fixtureDef(
            shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in self.HULL_POLY]),
            density=5.0,
            friction=0.1,
            categoryBits=0x0020,
            maskBits=0x001,  # collide only with ground
            restitution=0.0)  # 0.99 bouncy

        self.LEG_FD = fixtureDef(
            shape=polygonShape(box=(self.LEG_W / 2, self.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001)

        self.LOWER_FD = fixtureDef(
            shape=polygonShape(box=(0.8 * self.LEG_W / 2, self.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001)
        
        # Seed env and init Box2D
        self.seed()
        self.viewer = None
        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None
        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = self.FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = self.FRICTION,
                    categoryBits=0x0001,
                )

        # Init default hexagon fixture and shape, used only for Hexagon Tracks
        self.fd_default_hexagon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = self.FRICTION)
        self.default_hexagon = [(-0.5,0),(-0.5,0.25),(-0.25,0.5),(0.25,0.5),(0.5,0.25),(0.5,0)] 
        
        
        self.action_space = spaces.Box(np.array([-1]*self.nb_leg_pairs*4),
                                       np.array([1]*self.nb_leg_pairs*4), dtype=np.float32)
        high = np.array([np.inf] * (14+self.NB_LIDAR))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.torque_penalty = 0.00035

    # Use this init to initialize the environment configurations
    # So far you can choose between using a 'short', 'default' and quadrupedal ('quadru') walker
    def my_init(self, params):
        leg_size = params['leg_size']
        self.nb_leg_pairs = 1
        self.leg_pairs_x = [0]
        if leg_size == "short":
            self.LEG_W = 8 / self.SCALE
            self.LEG_H = 17 / self.SCALE
        elif leg_size == "quadru":  # quadrupedal walker has 2 pairs of legs and bigger body

            self.MOTORS_TORQUE = 300 #400 # increase motorself. torque for big boy
            self.torque_penalty /= 8 #20 # reduce torque penalty
            self.LEG_W = 10 / self.SCALE
            self.LEG_H = 51 / self.SCALE

            self.nb_leg_pairs = 2
            self.leg_pairs_x = [-1.2, 0.85]

            self.HULL_POLY = [
                (-46, +13), (+6, +13), (+50, +5),
                (+50, -12), (-46, -12)
            ]
            self.HULL_WIDTH = 96
        elif leg_size == "default":
            self.LEG_W = 8 / self.SCALE
            self.LEG_H = 34 / self.SCALE
        else:
            print(leg_size + ' what ?!')
            raise NotImplementedError

        # Update action space and observation space
        self.action_space = spaces.Box(np.array([-1] * self.nb_leg_pairs * 4),
                                       np.array([1] * self.nb_leg_pairs * 4), dtype=np.float32)
        high = np.array([np.inf] * (4 + self.nb_leg_pairs*2*5 + self.NB_LIDAR))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Update walker's fixtures
        self.HULL_FD = fixtureDef(
            shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in self.HULL_POLY]),
            density=5.0,
            friction=0.1,
            categoryBits=0x0020,
            maskBits=0x001,  # collide only with ground
            restitution=0.0)  # 0.99 bouncy

        self.LEG_FD = fixtureDef(
            shape=polygonShape(box=(self.LEG_W / 2, self.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001)

        self.LOWER_FD = fixtureDef(
            shape=polygonShape(box=(0.8 * self.LEG_W / 2, self.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Gather parameters for procedural track generation, make sure to call this before each new episode
    def set_environment(self, roughness=None, stump_height=None, stump_width=None, stump_rot=None,
                        obstacle_spacing=None, poly_shape=None, stump_seq=None):

        self.roughness = roughness if roughness else 0
        self.obstacle_spacing = max(0.01, obstacle_spacing) if obstacle_spacing is not None else 8.0
        self.stump_height = [stump_height, 0.1] if stump_height is not None else None
        self.stump_width = stump_width
        self.stump_rot = stump_rot
        self.hexa_shape = poly_shape
        self.stump_seq = stump_seq
        if poly_shape is not None:
            self.hexa_shape = np.interp(poly_shape,[0,4],[0,4]).tolist()
            assert(len(poly_shape) == 12)
            self.hexa_shape = self.hexa_shape[0:12]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _generate_terrain(self):
        GRASS, STUMP, HEXA = 0, None, None
        cpt=1
        if self.stump_height:
            STUMP = cpt
            cpt += 1
        if self.hexa_shape:
            HEXA = cpt
            cpt += 1
        if self.stump_seq is not None:
            SEQ = cpt
            cpt += 1
        _STATES_ = cpt

        state = self.np_random.randint(1, _STATES_)
        velocity = 0.0
        y = self.TERRAIN_HEIGHT
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        x = 0
        max_x = self.TERRAIN_LENGTH * self.TERRAIN_STEP

        # Add startpad
        max_startpad_x = self.TERRAIN_STARTPAD * self.TERRAIN_STEP
        self.terrain_x.append(x)
        self.terrain_y.append(y)
        x += max_startpad_x
        self.terrain_x.append(x)
        self.terrain_y.append(y)
        oneshot = True

        # Generation of terrain
        while x < max_x:
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(self.TERRAIN_HEIGHT - y)
                if x > max_startpad_x: velocity += self.np_random.uniform(-self.roughness, self.roughness)/self.SCALE
                y += velocity
                x += self.obstacle_spacing

            elif state==STUMP and oneshot:
                stump_height = max(0.05, self.np_random.normal(self.stump_height[0], self.stump_height[1]))
                stump_width = self.TERRAIN_STEP
                if self.stump_width is not None:
                    stump_width *= max(0.05, np.random.normal(self.stump_width[0], self.stump_width[1]))
                poly = [
                    (x, y),
                    (x+stump_width, y),
                    (x+stump_width, y+stump_height * self.TERRAIN_STEP),
                    (x,y+stump_height * self.TERRAIN_STEP),
                    ]
                x += stump_width
                if self.stump_rot is not None:
                    anchor = (np.array(poly[0]) + np.array(poly[1]))/2
                    rotation = np.clip(self.np_random.normal(self.stump_rot[0], self.stump_rot[1]),0,2*np.pi)
                    poly = Rotate2D(np.array(poly), anchor, rotation).tolist()
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon,
                    userData='stump')
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
            elif state==HEXA and oneshot:
                # first point do not move
                poly = []
                delta_pos = []
                for i in range(0,len(self.hexa_shape),2):
                    delta_pos.append(tuple(np.random.normal(self.hexa_shape[i:i+2],0.1)))
                for i,(b,d) in enumerate(zip(self.default_hexagon, delta_pos)):
                    if i != 0 and i != (len(self.default_hexagon)-1):
                        poly.append((x + (b[0]*self.TERRAIN_STEP) + (d[0]*self.TERRAIN_STEP),
                                     y + (b[1]*self.TERRAIN_STEP) + (d[1]*self.TERRAIN_STEP)))
                    else:
                        poly.append((x + (b[0]*self.TERRAIN_STEP) + (d[0]*self.TERRAIN_STEP),
                                     y + (b[1]*self.TERRAIN_STEP)))
                x += 1
                self.fd_default_hexagon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_default_hexagon)
                t.color1, t.color2 = (1.0, np.clip(delta_pos[0][1]/3,0,1), np.clip(delta_pos[-1][1]/3,0,1)), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state==SEQ and oneshot:
                for height, width in zip(self.stump_seq[0::2], self.stump_seq[1::2]):
                    stump_height = max(0.05, self.np_random.normal(height, 0.1))
                    stump_width = max(0.05, self.np_random.normal(width, 0.1))
                    poly = [
                        (x, y),
                        (x + stump_width, y),
                        (x + stump_width, y + stump_height * self.TERRAIN_STEP),
                        (x, y + stump_height * self.TERRAIN_STEP),
                    ]
                    x += stump_width
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_polygon,
                        userData='stump')
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)

            oneshot = False
            self.terrain_y.append(y)
            if state==GRASS:
                state = self.np_random.randint(1, _STATES_)
                oneshot = True
            else:
                state = GRASS
                oneshot = False

        # Draw terrain
        self.terrain_poly = []
        assert len(self.terrain_x) == len(self.terrain_y)
        for i in range(len(self.terrain_x)-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge,
                userData='grass')
            color = (0.3, 1.0 if (i % 2) == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(self.TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, self.TERRAIN_LENGTH)*self.TERRAIN_STEP
            y = self.VIEWPORT_H/self.SCALE*3/4
            poly = [
                (x+15*self.TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*self.TERRAIN_STEP),
                 y+ 5*self.TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*self.TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def draw_walker(self):
        self._generate_terrain()
        self._generate_clouds()

        init_x = self.TERRAIN_STEP*self.TERRAIN_STARTPAD/2
        init_y = self.TERRAIN_HEIGHT+2*self.LEG_H
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            fixtures = self.HULL_FD
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (0.3,0.3,0.5)
        self.hull.ApplyForceToCenter((self.np_random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM), 0), True)

        self.legs = []
        self.joints = []
        for x_anchor in self.leg_pairs_x:
            absolute_x = init_x + np.interp(x_anchor,[-1,1],[-self.HULL_WIDTH/2/self.SCALE,self.HULL_WIDTH/2/self.SCALE])
            for i in [-1, +1]:
                leg = self.world.CreateDynamicBody(
                    position=(absolute_x, init_y - self.LEG_H / 2 - self.LEG_DOWN),
                    angle=(i * 0.05),
                    fixtures=self.LEG_FD
                )
                leg.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
                leg.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
                rjd = revoluteJointDef(
                    bodyA=self.hull,
                    bodyB=leg,
                    localAnchorA=(x_anchor, self.LEG_DOWN),
                    localAnchorB=(0, self.LEG_H / 2),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=self.MOTORS_TORQUE,
                    motorSpeed=i,
                    lowerAngle=-0.8,
                    upperAngle=1.1,
                )
                self.legs.append(leg)
                self.joints.append(self.world.CreateJoint(rjd))

                lower = self.world.CreateDynamicBody(
                    position=(absolute_x, init_y - self.LEG_H * 3 / 2 - self.LEG_DOWN),
                    angle=(i * 0.05),
                    fixtures=self.LOWER_FD
                )
                lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
                lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
                rjd = revoluteJointDef(
                    bodyA=leg,
                    bodyB=lower,
                    localAnchorA=(0, -self.LEG_H / 2),
                    localAnchorB=(0, self.LEG_H / 2),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=self.MOTORS_TORQUE,
                    motorSpeed=1,
                    lowerAngle=-1.6,
                    upperAngle=-0.1,
                )
                lower.ground_contact = False
                self.legs.append(lower)
                self.joints.append(self.world.CreateJoint(rjd))

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.head_contact = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = self.VIEWPORT_W/self.SCALE
        H = self.VIEWPORT_H/self.SCALE

        self.draw_walker()

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(self.NB_LIDAR)]

        return self.step(np.array([0]*self.nb_leg_pairs*4))[0]

    def step(self, action):
        #self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(self.SPEED_HIP  * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(self.SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(self.SPEED_HIP  * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(self.SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            for i in range(len(self.joints)):
                if i%2 == 0:
                    self.joints[i].motorSpeed = float(self.SPEED_HIP * np.sign(action[i]))
                else:
                    self.joints[i].motorSpeed = float(self.SPEED_KNEE * np.sign(action[i]))
                self.joints[i].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[i]), 0, 1))

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(self.NB_LIDAR):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/self.NB_LIDAR)*self.LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/self.NB_LIDAR)*self.LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)
        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*vel.x*(self.VIEWPORT_W/self.SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(self.VIEWPORT_H/self.SCALE)/FPS]

        # add leg-related state
        for i in range(0,len(self.legs),2):
            state += [self.joints[i].angle,   # gives 1.1 on high up (spikes on hiting the ground, that's normal too)
                      self.joints[i].speed / self.SPEED_HIP,
                      self.joints[i+1].angle + 1.0,
                      self.joints[i+1].speed / self.SPEED_KNEE,
                      1.0 if self.legs[i+1].ground_contact else 0.0]

        state += [l.fraction for l in self.lidar]
        assert len(state)== (4+5*self.nb_leg_pairs*2+self.NB_LIDAR)

        self.scroll = pos.x - self.VIEWPORT_W/self.SCALE/5

        shaping  = 130*pos[0]/self.SCALE  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= self.torque_penalty * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.head_contact or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (self.TERRAIN_LENGTH-self.TERRAIN_GRASS)*self.TERRAIN_STEP:
            done   = True

        return np.array(state), reward, done, {}

    def render(self, mode='human'):
        #self.scroll = 1
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, self.VIEWPORT_W/self.SCALE + self.scroll, 0, self.VIEWPORT_H/self.SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+self.VIEWPORT_W/self.SCALE, 0),
            (self.scroll+self.VIEWPORT_W/self.SCALE, self.VIEWPORT_H/self.SCALE),
            (self.scroll,                  self.VIEWPORT_H/self.SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + self.VIEWPORT_W/self.SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + self.VIEWPORT_W/self.SCALE: continue
            self.viewer.draw_polygon(poly, color=color)
        for i in range(len(self.lidar)):
            l = self.lidar[i]
            self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = self.TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/self.SCALE
        x = self.TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/self.SCALE), (x+25/self.SCALE, flagy2-5/self.SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__=="__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = BipedalWalkerContinuous()
    env.set_environment()
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state==STAY_ON_ONE_LEG:
            hip_targ[moving_leg]  = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED: supporting_knee_angle += 0.03
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state==PUT_OTHER_DOWN:
            hip_targ[moving_leg]  = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base+4]:
                state = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if state==PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

        env.render()
        if done: break