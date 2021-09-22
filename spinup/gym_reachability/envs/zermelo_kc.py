# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import gym.spaces
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import random


class ZermeloKCEnv(gym.Env):


    def __init__(self, device, mode='normal', doneType='toEnd'):

        # State bounds.
        self.bounds = np.array([[-2, 2],  # axis_0 = state, axis_1 = bounds.
                                [-2, 10]])

        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Time step parameter.
        self.time_step = 0.05

        # Dubins car parameters.
        self.upward_speed = 2.0

        # Control parameters.
        self.horizontal_rate = 1
        self.discrete_controls = np.array([-self.horizontal_rate,
                                           0,
                                           self.horizontal_rate])

        # Constraint set parameters.
        # X,Y position and Side Length.
        self.box1_x_y_length = np.array([1.25, 2, 1.5])  # Bottom right.
        self.box2_x_y_length = np.array([-1.25, 2, 1.5])  # Bottom left.
        self.box3_x_y_length = np.array([0, 6, 1.5])  # Top middle.

        # Target set parameters.
        self.box4_x_y_length = np.array([0, 9.25, 1.5])  # Top.

        # Gym variables.
        self.action_space = gym.spaces.Discrete(3)  # horizontal_rate = {-1,0,1}
        self.midpoint = (self.low + self.high)/2.0
        self.interval = self.high - self.low
        self.observation_space = gym.spaces.Box(np.float32(self.midpoint - self.interval/2),
                                                np.float32(self.midpoint + self.interval/2))
        self.viewer = None

        # Set random seed.
        self.seed_val = 0
        np.random.seed(self.seed_val)

        # Cost Params
        self.penalty = 1
        self.reward = -1
        self.costType = 'dense_ell'
        self.scaling = 1.

        # mode: normal or extend (keep track of ell & g)
        self.mode = mode
        if mode == 'extend':
            self.state = np.zeros(3)
        else:
            self.state = np.zeros(2)
        self.doneType = doneType

        # Visualization params
        self.vis_init_flag = True
        (self.x_box1_pos, self.x_box2_pos,
         self.x_box3_pos, self.y_box1_pos,
         self.y_box2_pos, self.y_box3_pos) = self.constraint_set_boundary()
        (self.x_box4_pos, self.y_box4_pos) = self.target_set_boundary()
        self.visual_initial_states = [np.array([ 0,  0]),
                                      np.array([-1, -2]),
                                      np.array([ 1, -2]),
                                      np.array([-1,  4]),
                                      np.array([ 1,  4])]
        if mode == 'extend':
            self.visual_initial_states = self.extend_state(self.visual_initial_states)

        print("Env: mode---{:s}; doneType---{:s}".format(mode, doneType))

        # for torch
        self.device = device


    def extend_state(self, states):
        new_states = []
        for state in states:
            l_x = self.target_margin(state)
            g_x = self.safety_margin(state)
            new_states.append(np.append(state, max(l_x, g_x)))
        return new_states


    def reset(self, start=None):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        if start is None:
            self.state = self.sample_random_state()
        else:
            self.state = start
        return np.copy(self.state)


    def sample_random_state(self, keepOutOf=False):
        flag = True
        while flag:
            rnd_state = np.random.uniform(low=self.low,
                                      high=self.high)
            l_x = self.target_margin(rnd_state)
            g_x = self.safety_margin(rnd_state)

            if self.mode == 'extend':
                rnd_state = np.append(rnd_state, max(l_x, g_x))

            terminal = (g_x > 0) or (l_x <= 0)
            flag = terminal and keepOutOf

        return rnd_state


    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: Input action.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """

        if self.mode == 'extend':
            x, y, z = self.state
        else:
            x, y = self.state

        l_x_cur = self.target_margin(self.state[:2])
        g_x_cur = self.safety_margin(self.state[:2])

        u = self.discrete_controls[action]
        state, [l_x_nxt, g_x_nxt] = self.integrate_forward(self.state, u)
        self.state = state

        # cost
        if self.mode == 'extend' or self.mode == 'RA':
            fail = g_x_cur > 0
            success = l_x_cur <= 0
            if fail:
                cost = self.penalty
            elif success:
                cost = self.reward
            else:
                cost = 0.
        else:
            fail = g_x_nxt > 0
            success = l_x_nxt <= 0
            if g_x_nxt > 0 or g_x_cur > 0:
                cost = self.penalty
            elif l_x_nxt <= 0 or l_x_cur <= 0:
                cost = self.reward
            else:
                if self.costType == 'dense_ell':
                    cost = l_x_nxt
                elif self.costType == 'dense_ell_g':
                    cost = l_x_nxt + g_x_nxt
                elif self.costType == 'imp_ell_g':
                    cost = (l_x_nxt-l_x_cur) + (g_x_nxt-g_x_cur)
                elif self.costType == 'imp_ell':
                    cost = (l_x_nxt-l_x_cur)
                elif self.costType == 'sparse':
                    cost = 0. * self.scaling
                elif self.costType == 'max_ell_g':
                    cost = max(l_x_nxt, g_x_nxt)
                else:
                    cost = 0.
        # done
        if self.doneType == 'toEnd':
            outsideTop   = (self.state[1] >= self.bounds[1,1])
            outsideLeft  = (self.state[0] <= self.bounds[0,0])
            outsideRight = (self.state[0] >= self.bounds[0,1])
            done = outsideTop or outsideLeft or outsideRight
        else:
            done = fail or success
            assert self.doneType == 'TF', 'invalid doneType'

        info = {"g_x": g_x_cur, "l_x": l_x_cur, "g_x_nxt": g_x_nxt, "l_x_nxt": l_x_nxt}
        return np.copy(self.state), cost, done, info


    def integrate_forward(self, state, u):
        """ Integrate the dynamics forward by one step.

        Args:
            state:  x, y - position
                    [z]  - optional, extra state dimension capturing
                                     reach-avoid outcome so far)
            u: Contol input.

        Returns:
            State variables (x, y, [z])  integrated one step forward in time.
        """
        if self.mode == 'extend':
            x, y, z = state
        else:
            x, y = state

        # one step forward
        x = x + self.time_step * u
        y = y + self.time_step * self.upward_speed

        l_x = self.target_margin(np.array([x, y]))
        g_x = self.safety_margin(np.array([x, y]))

        if self.mode == 'extend':
            z = min(z, max(l_x, g_x) )
            state = np.array([x, y, z])
        else:
            state = np.array([x, y])

        info = np.array([l_x, g_x])

        return state, info


    def safety_margin(self, s):
        """ Computes the margin (e.g. distance) between state and failue set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        box1_safety_margin = -(np.linalg.norm(s - self.box1_x_y_length[:2],
                               ord=np.inf) - self.box1_x_y_length[-1]/2.0)
        box2_safety_margin = -(np.linalg.norm(s - self.box2_x_y_length[:2],
                               ord=np.inf) - self.box2_x_y_length[-1]/2.0)
        box3_safety_margin = -(np.linalg.norm(s - self.box3_x_y_length[:2],
                               ord=np.inf) - self.box3_x_y_length[-1]/2.0)

        vertical_margin = (np.abs( s[1] - (self.low[1] + self.high[1])/2.0)
                           - self.interval[1]/2.0)
        horizontal_margin = (np.abs( s[0] - (self.low[0] + self.high[0])/2.0)
                           - self.interval[0]/2.0)
        enclosure_safety_margin = max(horizontal_margin, vertical_margin)

        safety_margin = max(box1_safety_margin,
                            box2_safety_margin,
                            box3_safety_margin,
                            enclosure_safety_margin)

        return self.scaling * safety_margin


    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        box4_target_margin = (np.linalg.norm(s - self.box4_x_y_length[:2],
                              ord=np.inf) - self.box4_x_y_length[-1]/2.0)

        target_margin = box4_target_margin
        return self.scaling * target_margin


    def set_costParam(self, penalty=1, reward=-1, costType='normal', scaling=4.):
        self.penalty = penalty
        self.reward = reward
        self.costType = costType
        self.scaling = scaling


    def set_seed(self, seed):
        """ Set the random seed.

        Args:
            seed: Random seed.
        """
        self.seed_val = seed
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)  # if you are using multi-GPU.
        random.seed(self.seed_val) 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def set_bounds(self, bounds):
        """ Set state bounds.

        Args:
            bounds: Bounds for the state.
        """
        self.bounds = bounds

        # Get lower and upper bounds
        self.low = np.array(self.bounds)[:, 0]
        self.high = np.array(self.bounds)[:, 1]

        # Double the range in each state dimension for Gym interface.
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(np.float32(midpoint - interval/2),
                                                np.float32(midpoint + interval/2))


    def set_doneType(self, doneType):
        self.doneType = doneType


    def render(self):
        pass


    def constraint_set_boundary(self):
        """ Computes the safe set boundary based on the analytic solution.

        The boundary of the safe set for the double integrator is determined by
        two parabolas and two line segments.

        Returns:
            Set of discrete points describing each parabola. The first and last
            two elements of the list describe the set of coordinates for the
            first and second parabola respectively.
        """
        x_box1_pos = np.array([
            self.box1_x_y_length[0] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] - self.box1_x_y_length[-1]/2.0])
        x_box2_pos = np.array([
            self.box2_x_y_length[0] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] - self.box2_x_y_length[-1]/2.0])
        x_box3_pos = np.array([
            self.box3_x_y_length[0] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] - self.box3_x_y_length[-1]/2.0])

        y_box1_pos = np.array([
            self.box1_x_y_length[1] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] - self.box1_x_y_length[-1]/2.0])
        y_box2_pos = np.array([
            self.box2_x_y_length[1] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] - self.box2_x_y_length[-1]/2.0])
        y_box3_pos = np.array([
            self.box3_x_y_length[1] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] - self.box3_x_y_length[-1]/2.0])

        return (x_box1_pos, x_box2_pos, x_box3_pos,
                y_box1_pos, y_box2_pos, y_box3_pos)


    def target_set_boundary(self):
        """ Computes the safe set boundary based on the analytic solution.

        The boundary of the safe set for the double integrator is determined by
        two parabolas and two line segments.

        Returns:
            Set of discrete points describing each parabola. The first and last
            two elements of the list describe the set of coordinates for the
            first and second parabola respectively.
        """
        x_box4_pos = np.array([
            self.box4_x_y_length[0] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] - self.box4_x_y_length[-1]/2.0])

        y_box4_pos = np.array([
            self.box4_x_y_length[1] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] - self.box4_x_y_length[-1]/2.0])

        return (x_box4_pos, y_box4_pos)


    def get_value(self, q_func, nx=41, ny=121, addBias=False):
        v = np.zeros((nx, ny))
        it = np.nditer(v, flags=['multi_index'])
        xs = np.linspace(self.bounds[0,0], self.bounds[0,1], nx)
        ys = np.linspace(self.bounds[1,0], self.bounds[1,1], ny)
        while not it.finished:
            idx = it.multi_index

            x = xs[idx[0]]
            y = ys[idx[1]]
            l_x = self.target_margin(np.array([x, y]))
            g_x = self.safety_margin(np.array([x, y]))

            if self.mode == 'normal' or self.mode == 'RA':
                state = torch.FloatTensor([x, y]).to(self.device).unsqueeze(0)
            else:
                z = max([l_x, g_x])
                state = torch.FloatTensor([x, y, z]).to(self.device).unsqueeze(0)

            if addBias:
                v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
            else:
                v[idx] = q_func(state).min(dim=1)[0].item()
            it.iternext()
        return xs, ys, v


    def get_axes(self):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
        """
        aspect_ratio = (self.bounds[0,1]-self.bounds[0,0])/(self.bounds[1,1]-self.bounds[1,0])
        axes = np.array([self.bounds[0,0], self.bounds[0,1], self.bounds[1,0], self.bounds[1,1]])
        return [axes, aspect_ratio]


    def get_warmup_examples(self, num_warmup_samples=100):
        x_min, x_max = self.bounds[0,:]
        y_min, y_max = self.bounds[1,:]

        xs = np.random.uniform(x_min, x_max, num_warmup_samples)
        ys = np.random.uniform(y_min, y_max, num_warmup_samples)
        heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
        states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

        for i in range(num_warmup_samples):
            x, y = xs[i], ys[i]
            l_x = self.target_margin(np.array([x, y]))
            g_x = self.safety_margin(np.array([x, y]))
            heuristic_v[i,:] = np.maximum(l_x, g_x)
            states[i, :] = x, y

        return states, heuristic_v


    def simulate_one_trajectory(self, q_func, T=10, state=None, keepOutOf=False, toEnd=False):

        if state is None:
            state = self.sample_random_state(keepOutOf=keepOutOf)
        x, y = state[:2]
        traj_x = [x]
        traj_y = [y]
        result = 0 # not finished

        for t in range(T):
            if toEnd:
                outsideTop   = (state[1] >= self.bounds[1,1])
                outsideLeft  = (state[0] <= self.bounds[0,0])
                outsideRight = (state[0] >= self.bounds[0,1])
                done = outsideTop or outsideLeft or outsideRight
                if done:
                    result = 1
                    break
            else:
                if self.safety_margin(state[:2]) > 0:
                    result = -1 # failed
                    break
                elif self.target_margin(state[:2]) <= 0:
                    result = 1 # succeeded
                    break

            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action_index = q_func(state_tensor).min(dim=1)[1].item()
            u = self.discrete_controls[action_index]

            state, _ = self.integrate_forward(state, u)
            traj_x.append(state[0])
            traj_y.append(state[1])

        return traj_x, traj_y, result


    def simulate_trajectories(self, q_func, T=10,
                              num_rnd_traj=None, states=None,
                              keepOutOf=False, toEnd=False):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            results = np.empty(shape=(num_rnd_traj,), dtype=int)
            for idx in range(num_rnd_traj):
                traj_x, traj_y, result = self.simulate_one_trajectory(q_func, T=T,
                                                                      keepOutOf=keepOutOf, toEnd=toEnd)
                trajectories.append((traj_x, traj_y))
                results[idx] = result
        else:
            results = np.empty(shape=(len(states),), dtype=int)
            for idx, state in enumerate(states):
                traj_x, traj_y, result = self.simulate_one_trajectory(q_func, T=T, state=state, toEnd=toEnd)
                trajectories.append((traj_x, traj_y))
                results[idx] = result

        return trajectories, results


    def visualize(  self, q_func, no_show=False,
                    vmin=-1, vmax=1, nx=81, ny=241, cmap='seismic',
                    labels=['', ''], boolPlot=False, addBias=False):
        """ Overlays analytic safe set on top of state value function.

        Args:
            v: State value function.
        """
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        axStyle = self.get_axes()
        cbarPlot = True

        #== Plot failure / target set ==
        self.plot_target_failure_set(ax)

        #== Plot reach-avoid set ==
        self.plot_reach_avoid_set(ax)

        #== Plot V ==
        self.plot_v_values(q_func, ax=ax, fig=fig,
                            vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
                            boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias)
        
        #== Plot Trajectories ==
        self.plot_trajectories(q_func, states=self.visual_initial_states, toEnd=False, ax=ax)

        #== Formatting ==
        self.plot_formatting(ax=ax, labels=labels)

        if not no_show:
            plt.show()


    def plot_v_values(self, q_func, ax=None, fig=None,
                        vmin=-1, vmax=1, nx=201, ny=201, cmap='seismic',
                        boolPlot=False, cbarPlot=True, addBias=False):
        axStyle = self.get_axes()

        #== Plot V ==
        xs, ys, v = self.get_value(q_func, nx, ny, addBias=addBias)

        if boolPlot:
            im = ax.imshow(v.T>0., interpolation='none', extent=axStyle[0], origin="lower",
                       cmap=cmap)
        else:
            im = ax.imshow(v.T, interpolation='none', extent=axStyle[0], origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax)
            if cbarPlot:
                cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax])
                cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)


    def plot_trajectories(self, q_func, T=200, num_rnd_traj=None, states=None,
                          keepOutOf=False, toEnd=False, ax=None, c='y', lw=2):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories, results = self.simulate_trajectories(q_func, T=T,
                                                           num_rnd_traj=num_rnd_traj, states=states,
                                                           keepOutOf=keepOutOf, toEnd=toEnd)
        for traj in trajectories:
            traj_x, traj_y = traj
            ax.scatter(traj_x[0], traj_y[0], s=48, c=c)
            ax.plot(traj_x, traj_y, color=c, linewidth=lw)

        return results


    def plot_target_failure_set(self, ax=None, c='m', c_f='k', lw=1.5):
        # Plot bounadries of constraint set.
        ax.plot(self.x_box1_pos, self.y_box1_pos, color=c_f, lw=lw)
        ax.plot(self.x_box2_pos, self.y_box2_pos, color=c_f, lw=lw)
        ax.plot(self.x_box3_pos, self.y_box3_pos, color=c_f, lw=lw)

        # Plot boundaries of target set.
        ax.plot(self.x_box4_pos, self.y_box4_pos, color=c, lw=lw)


    def plot_reach_avoid_set(self, ax=None, c='g', lw=3):
        slope = self.upward_speed / self.horizontal_rate

        def get_line(slope, end_point, x_limit, ns=100):
            x_end, y_end = end_point
            b = y_end - slope * x_end

            xs = np.linspace(x_limit, x_end, ns)
            ys = xs * slope + b
            return xs, ys

        # left unsafe set
        x = self.box2_x_y_length[0] + self.box2_x_y_length[2]/2.0
        y = self.box2_x_y_length[1] - self.box2_x_y_length[2]/2.0
        xs, ys = get_line(slope, end_point=[x,y], x_limit=-2.)
        ax.plot(xs, ys, color=c, linewidth=lw)

        # right unsafe set
        x = self.box1_x_y_length[0] - self.box1_x_y_length[2]/2.0
        y = self.box1_x_y_length[1] - self.box1_x_y_length[2]/2.0
        xs, ys = get_line(-slope, end_point=[x,y], x_limit=2.)
        ax.plot(xs, ys, color=c, linewidth=lw)

        # middle unsafe set
        x1 = self.box3_x_y_length[0] - self.box3_x_y_length[2]/2.0
        x2 = self.box3_x_y_length[0] + self.box3_x_y_length[2]/2.0
        x3 = self.box3_x_y_length[0]
        y = self.box3_x_y_length[1] - self.box3_x_y_length[2]/2.0
        xs, ys = get_line(-slope, end_point=[x1,y], x_limit=x3)
        ax.plot(xs, ys, color=c, linewidth=lw)
        xs, ys = get_line(slope, end_point=[x2,y], x_limit=x3)
        ax.plot(xs, ys, color=c, linewidth=lw)

        # border unsafe set
        x1 = self.box4_x_y_length[0] - self.box4_x_y_length[2]/2.0
        x2 = self.box4_x_y_length[0] + self.box4_x_y_length[2]/2.0
        y = self.box4_x_y_length[1] + self.box4_x_y_length[2]/2.0
        xs, ys = get_line(slope, end_point=[x1,y], x_limit=-2.)
        ax.plot(xs, ys, color=c, linewidth=lw)
        xs, ys = get_line(-slope, end_point=[x2,y], x_limit=2.)
        ax.plot(xs, ys, color=c, linewidth=lw)


    def plot_formatting(self, ax=None, labels=None):
        axStyle = self.get_axes()
        #== Formatting ==
        ax.axis(axStyle[0])
        ax.set_aspect(axStyle[1])  # makes equal aspect ratio
        ax.grid(False)
        if labels is not None:
            ax.set_xlabel(labels[0], fontsize=52)
            ax.set_ylabel(labels[1], fontsize=52)

        ax.tick_params( axis='both', which='both',  # both x and y axes, both major and minor ticks are affected
                        bottom=False, top=False,    # ticks along the top and bottom edges are off
                        left=False, right=False)    # ticks along the left and right edges are off
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_title(r"$\theta$={:.1f}".format(theta * 180 / np.pi), fontsize=24)
