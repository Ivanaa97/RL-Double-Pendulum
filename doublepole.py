import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class DoublePoleEnv(gym.Env):
    """
    Description:
        Two poles are attached by a hinge to a cart, which moves along
        a track. The pendulums start upright, and the goal is to prevent it from
        falling over by increasing and reducing the cart's velocity.

    Source:
        The equation of motion are defined in the next link
        https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html

    Observation:
        Type: Box(6)
        Num     Observation                Min                    Max
        0       Cart Position              -4.8                   4.8
        1       Cart Velocity              -Inf                   Inf
        2       Pole Angle1                -2*36 grados           2*36 grados
        3       Pole Angular Velocity1     -Inf                   Inf
        4       Pole Angle2                -2*36 grados           2*36 grados
        5       Pole Angular Velocity2     -Inf                   Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle1 is more than 36 degrees.
        Pole Angle2 is more than 36 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        #Defining the parameters of the dynamics
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole1 = 0.10
        self.masspole2 = 0.05
        self.length = 1
        self.total_mass = (self.masspole1 + self.masspole2 + self.masscart) 
        self.polemass_length1 = 1 
        self.polemass_length2 = 0.5
        self.force_mag = 50.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        
        #friction coefficients
        self.friction_cart = 5*10**(-4)
        self.friction_pole1 = 2*10**(-6)
        self.friction_pole2 = self.friction_pole1 

        # Angle at which to fail the episode
        self.theta_threshold_radians1 = 36 * 2 * math.pi / 360
        self.theta_threshold_radians2 = 36 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians1 * 2,
                         np.finfo(np.float32).max,
                        self.theta_threshold_radians2 * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.steps_beyond_done = None
        return np.array(self.state) 
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
          
        #Equation of motion
        
        x, x_dot, theta1, theta_dot1, theta2, theta_dot2 = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        x_dot1 = x_dot if x_dot>=0 else -x_dot
        
        costheta1 = math.cos(theta1)
        sintheta1 = math.sin(theta1)
        
        costheta2 = math.cos(theta2)
        sintheta2 = math.sin(theta2)
        
        div =  (self.masspole1*costheta1**2 + self.masspole2*costheta2**2) - (7.0/3.0)*self.total_mass 
        
        xacc = (costheta1*sintheta1*self.gravity + costheta2*sintheta2*self.gravity - (7.0/3.0)*(force - self.friction_cart* x_dot + self.masspole1*self.polemass_length1*sintheta1*theta_dot1**2 + self.masspole2*self.polemass_length2*sintheta2*theta_dot2**2) - (self.friction_pole1*theta_dot1*costheta1/self.polemass_length1) - (self.friction_pole2*theta_dot2*costheta2/self.polemass_length2))/div
        
        thetaacc1 = (3.0/7.0)*(self.gravity*sintheta1 - xacc*costheta1 - (self.friction_pole1*theta_dot1/(self.masspole1*self.polemass_length1)))/self.polemass_length1
        
        thetaacc2 = (3.0/7.0)*(self.gravity*sintheta2 - xacc*costheta2 - (self.friction_pole2*theta_dot2/(self.masspole2*self.polemass_length2)))/self.polemass_length2
        
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta1 = theta1 + self.tau * theta_dot1
            theta_dot1 = theta_dot1 + self.tau * thetaacc1
            theta2 = theta2 + self.tau * theta_dot2
            theta_dot2 = theta_dot2 +self.tau * thetaacc2
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot1 = theta_dot1 + self.tau * thetaacc1
            theta1 = theta1 + self.tau * theta_dot1
            theta_dot2 = theta_dot2 + self.tau * thetaacc2
            theta2 = theta2 + self.tau * theta_dot2

        self.state = (x, x_dot, theta1, theta_dot1,theta2, theta_dot2)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta1 < -self.theta_threshold_radians1
            or theta1 > self.theta_threshold_radians1
            or theta2 < -self.theta_threshold_radians2
            or theta2 > self.theta_threshold_radians2
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth1 = 10.0
        polelen1 = scale * (2 * self.length)
        polewidth2 = 8.0
        polelen2 = scale * ( self.length)
        cartwidth = 50.0
        cartheight = 25.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            
            #Pole1
            l, r, t, b = -polewidth1 / 2, polewidth1 / 2, polelen1 - polewidth1 / 2, -polewidth1 / 2
            pole1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole1.set_color(.3, .1, .2)
            self.pole1trans = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.pole1trans)
            pole1.add_attr(self.carttrans)
            self.viewer.add_geom(pole1)
            
            #Pole2
            l, r, t, b = -polewidth2 / 2, polewidth2 / 2, polelen2 - polewidth2 / 2, -polewidth2 / 2
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(.3, .7, .9)
            self.pole2trans = rendering.Transform(translation=(0, axleoffset))
            pole2.add_attr(self.pole2trans)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)
            
            self.axle = rendering.make_circle(polewidth1/2)
            self.axle.add_attr(self.pole1trans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .5)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            
            self._pole1_geom = pole1
            self._pole2_geom = pole2
         
 
        if self.state is None:
            return None

        # Edit the pole1 polygon vertex
        pole1 = self._pole1_geom
        l, r, t, b = -polewidth1 / 2, polewidth1 / 2, polelen1 - polewidth1 / 2, -polewidth1 / 2
        pole1.v = [(l, b), (l, t), (r, t), (r, b)]
        
        # Edit the pole2 polygon vertex
        pole2 = self._pole2_geom
        l, r, t, b = -polewidth2 / 2, polewidth2 / 2, polelen2 - polewidth2 / 2, -polewidth2 / 2
        pole2.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.pole1trans.set_rotation(-x[2])
        self.pole2trans.set_rotation(-x[4])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
