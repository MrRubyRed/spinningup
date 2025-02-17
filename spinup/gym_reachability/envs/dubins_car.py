# Copyright (c) 2020–2021, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )

import gym.spaces
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
import spinup.algos.pytorch.ra_ppo.ra_core as core


class DubinsCarEnv(gym.Env):
    def __init__(self, device='cpu', mode='normal', doneType='toEnd', discrete=True):

        # State bounds.
        # self.bounds = np.array([[-1.1, 1.1],  # axis_0 = state, axis_1 = bounds.
        #                         [-1.1, 1.1],
        #                         [-np.pi, np.pi]])
        self.bounds = np.array([[-1.1, 1.1],  # VRR Added for cos sin angles.
                                [-1.1, 1.1],
                                [-1.0, 1.0],
                                [-1.0, 1.0]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Time step parameter.
        self.time_step = 0.05

        # Dubins car parameters.
        self.speed = 0.5 # 0.5 v

        # Control parameters.
        self.R_turn = .6 # 0.6
        self.max_turning_rate = self.speed / self.R_turn  # w
        self.discrete = discrete
        self.discrete_controls = np.array([-self.max_turning_rate,
                                   0.,
                                   self.max_turning_rate])
        if self.discrete:
            self.action_space = gym.spaces.Discrete(self.discrete_controls.shape[0])
        else:
            # np.array([1.], dtype=np.float32)
            self.action_space = gym.spaces.Box(
                np.array([self.discrete_controls[0]], dtype=np.float32),
                np.array([self.discrete_controls[-1]], dtype=np.float32))

        # Constraint set parameters.
        self.constraint_center = np.array([0, 0])
        self.constraint_radius = 1.0

        # Target set parameters.
        self.target_center = np.array([0, 0])
        self.target_radius = .3

        # Gym variables.
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(np.float32(midpoint - interval/2),
                                                np.float32(midpoint + interval/2))

        # Internal state.
        self.mode = mode
        # self.state = np.zeros(3)
        self.state = np.zeros(4)    # VRR Added for cos sin angles.
        self.doneType = doneType

        # Set random seed.
        self.seed_val = 0
        np.random.seed(self.seed_val)

        # Visualization params
        # self.fig = None
        # self.axes = None
        self.visual_initial_states =[   np.array([ .6*self.constraint_radius,  -.5, np.cos(np.pi/2), np.sin(np.pi/2)]),
                                        np.array([ -.4*self.constraint_radius, -.5, np.cos(np.pi/2), np.sin(np.pi/2)]),
                                        np.array([ -0.95*self.constraint_radius, 0., np.cos(np.pi/2), np.sin(np.pi/2)]),
                                        np.array([ self.R_turn, 0.95*(self.constraint_radius-self.R_turn), np.cos(np.pi/2), np.sin(np.pi/2)]),
                                    ]
        # Cost Params
        self.targetScaling = 1.
        self.safetyScaling = 1.
        self.penalty = 1.
        self.reward = -1.
        self.costType = 'sparse'
        self.device = device

        print("Env: mode---{:s}; doneType---{:s}".format(mode, doneType))

#== Reset Functions ==
    def reset(self, start=None):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        if start is None:
            x_rnd, y_rnd, cos_theta_rnd, sin_theta_rnd = self.sample_random_state()
            self.state = np.array([x_rnd, y_rnd, cos_theta_rnd, sin_theta_rnd])
        else:
            self.state = start
        return np.copy(self.state)

    def sample_random_state(self, theta=None):
        if theta is None:
            theta_rnd = (2.0 * np.random.uniform() - 1.0) * np.pi
        else:
            theta_rnd = theta

        angle = (2.0 * np.random.uniform() - 1.0) * np.pi  # the position angle
        dist = self.constraint_radius * np.sqrt(np.random.uniform())
        x_rnd = dist * np.cos(angle)
        y_rnd = dist * np.sin(angle)

        return x_rnd, y_rnd, np.cos(theta_rnd), np.sin(theta_rnd)

#== Dynamics Functions ==
# Try good coverage of the state action space.
    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: Input action.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """
        # The signed distance must be computed before the environment steps forward.
        x, y, cos_theta, sin_theta = self.state

        l_x_cur = self.target_margin(self.state[:2])
        g_x_cur = self.safety_margin(self.state[:2])

        if self.discrete:
            u = self.discrete_controls[action]
        else:
            u = action
        self.state, info = self.integrate_forward(self.state, u)
        l_x_nxt, g_x_nxt = info

        # # cost
        # if self.mode == 'RA':
        #     fail = g_x_cur > 0
        #     success = l_x_cur <= 0
        #     if fail:
        #         cost = self.penalty
        #     elif success:
        #         cost = self.reward
        #     else:
        #         cost = 0.
        # else:
        #     fail = g_x_nxt > 0
        #     success = l_x_nxt <= 0
        #     if g_x_nxt > 0 or g_x_cur > 0:
        #         cost = self.penalty
        #     elif l_x_nxt <= 0 or l_x_cur <= 0:
        #         cost = self.reward
        #     else:
        #         if self.costType == 'dense_ell':
        #             cost = l_x_nxt
        #         elif self.costType == 'dense_ell_g':
        #             cost = l_x_nxt + g_x_nxt
        #         elif self.costType == 'sparse':
        #             cost = 0. #* self.scaling
        #         else:
        #             cost = 0.
        fail = g_x_cur > 0
        success = l_x_cur <= 0
        reward = 1.0 * success - 5.0 * fail
        # done
        # done = fail
        # If done flag has not triggered, just collect normal info.
        # if not done:
        #     info = {"g_x": g_x_cur, "l_x": l_x_cur}
        # else:
        #     info = {"g_x": self.penalty, "l_x": l_x_cur}
        # if self.doneType == 'toEnd':
        #     done = not self.check_within_bounds(self.state)
        #     if (l_x_cur < 0 and (l_x_nxt - l_x_cur) > 0):
        #         done = True
        # else:
        # done = ((l_x_cur <= 0) and (l_x_nxt > 0)) or fail #fail #or success
        done = fail
        # assert self.doneType == 'TF', 'invalid doneType'

        info = {"g_x": g_x_cur, "l_x": l_x_cur, "g_x_nxt": g_x_nxt, "l_x_nxt": l_x_nxt}
        return np.copy(self.state), reward, done, info

    def integrate_forward(self, state, u):
        """ Integrate the dynamics forward by one step.

        Args:
            x: Position in x-axis.
            y: Position in y-axis
            theta: Heading.
            u: Contol input.

        Returns:
            State variables (x,y,theta) integrated one step forward in time.
        """
        x, y, cos_theta, sin_theta = state
        if not self.discrete:
            u = max(self.discrete_controls[0],
                    min(self.discrete_controls[-1], u[0]))

        x = x + self.time_step * self.speed * cos_theta
        y = y + self.time_step * self.speed * sin_theta
        theta = np.arctan2(sin_theta, cos_theta)
        theta = np.mod((theta + np.pi) + self.time_step * u, 2*np.pi) - np.pi
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        l_x = self.target_margin(np.array([x, y]))
        g_x = self.safety_margin(np.array([x, y]))

        state = np.array([x, y, cos_theta, sin_theta])
        info = np.array([l_x, g_x])

        return state, info

#== Setting Hyper-Parameter Functions ==
    def set_costParam(self, penalty=1, reward=-1, costType='normal', targetScaling=1., safetyScaling=1.):
        self.penalty = penalty
        self.reward = reward
        self.costType = costType
        self.safetyScaling = safetyScaling
        self.targetScaling = targetScaling

    # def set_radius(self, target_radius=.3, constraint_radius=1., R_turn=.6):
    #     self.target_radius = target_radius
    #     self.constraint_radius = constraint_radius
    #     self.R_turn = R_turn
    #     self.max_turning_rate = self.speed / self.R_turn # w
    #     self.discrete_controls = np.array([ -self.max_turning_rate,
    #                                         0.,
    #                                         self.max_turning_rate])

    # def set_constraint(self, center=np.array([0.,0.]), radius=1.):
    #     self.constraint_center = center
    #     self.constraint_radius = radius

    # def set_target(self, center=np.array([0.,0.]), radius=.4):
    #     self.target_center = center
    #     self.target_radius = radius

    # def set_radius_rotation(self, R_turn=.6, verbose=False):
    #     self.R_turn = R_turn
    #     self.max_turning_rate = self.speed / self.R_turn # w
    #     self.discrete_controls = np.array([ -self.max_turning_rate,
    #                                         0.,
    #                                         self.max_turning_rate])
    #     if verbose:
    #         print(self.discrete_controls)

    def set_seed(self, seed):
        """ Set the random seed.

        Args:
            seed: Random seed.
        """
        self.seed_val = seed
        np.random.seed(self.seed_val)

    # def set_bounds(self, bounds):
    #     """ Set state bounds.

    #     Args:
    #         bounds: Bounds for the state.
    #     """
    #     self.bounds = bounds

    #     # Get lower and upper bounds
    #     self.low = np.array(self.bounds)[:, 0]
    #     self.high = np.array(self.bounds)[:, 1]

    #     # Double the range in each state dimension for Gym interface.
    #     midpoint = (self.low + self.high)/2.0
    #     interval = self.high - self.low
    #     self.observation_space = gym.spaces.Box(np.float32(midpoint - interval/2),
    #                                             np.float32(midpoint + interval/2))

# == Margin Functions ==
    def _calculate_margin_rect(self, s, x_y_w_h, negativeInside=True):
        x, y, w, h = x_y_w_h
        delta_x = np.abs(s[0] - x)
        delta_y = np.abs(s[1] - y)
        margin = max(delta_y - h/2, delta_x - w/2)

        if negativeInside:
            return margin
        else:
            return - margin

    def _calculate_margin_circle(self, s, c_r, negativeInside=True):
        center, radius = c_r
        dist_to_center = np.linalg.norm(s[:2] - center)
        margin = dist_to_center - radius

        if negativeInside:
            return margin
        else:
            return - margin

    def safety_margin(self, s):
        """ Computes the margin (e.g. distance) between state and failue set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        # x, y = (self.low + self.high)[:2] / 2.0
        # w, h = (self.high - self.low)[:2]
        # boundary_margin = self._calculate_margin_rect(s, [x, y, w, h], negativeInside=True)
        # g_xList = [boundary_margin]

        if self.constraint_center is not None and self.constraint_radius is not None:
            g_x = self._calculate_margin_circle(s, [self.constraint_center, self.constraint_radius],
                negativeInside=True)
            # g_xList.append(g_x)

        safety_margin = g_x # np.max(np.array(g_xList))
        return self.safetyScaling * safety_margin

    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        if self.target_center is not None and self.target_radius is not None:
            target_margin = self._calculate_margin_circle(s, [self.target_center, self.target_radius],
                    negativeInside=True)
            return self.targetScaling * target_margin
        else:
            return None

# == Getting Functions ==
    def check_within_bounds(self, state):
        for dim, bound in enumerate(self.bounds[:2]):
            flagLow = state[dim] <= bound[0]
            flagHigh = state[dim] >= bound[1]
            if flagLow or flagHigh:
                return False
        return True

    def get_warmup_examples(self, num_warmup_samples=100):

        rv = np.random.uniform(low=self.low[:2],
                               high=self.high[:2],
                               size=(num_warmup_samples,2))
        x_rnd, y_rnd = rv[:, 0], rv[:, 1]
        angle = np.random.uniform(low=-np.pi,
                                  high=np.pi,
                                  size=(num_warmup_samples,1))
        cos_theta_rnd, sin_theta_rnd = np.cos(angle[:, 0]), np.sin(angle[:, 0])

        heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
        states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

        for i in range(num_warmup_samples):
            x, y, cos_theta, sin_theta = (x_rnd[i], y_rnd[i], cos_theta_rnd[i],
                                          sin_theta_rnd[i])
            l_x = self.target_margin(np.array([x, y]))
            g_x = self.safety_margin(np.array([x, y]))
            heuristic_v[i, :] = np.maximum(l_x, g_x)
            states[i, :] = x, y, cos_theta, sin_theta

        return states, heuristic_v

    def get_axes(self):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
        """
        aspect_ratio = (self.bounds[0,1]-self.bounds[0,0])/(self.bounds[1,1]-self.bounds[1,0])
        axes = np.array([self.bounds[0,0], self.bounds[0,1], self.bounds[1,0], self.bounds[1,1]])
        return [axes, aspect_ratio]

    def get_value(self, q_func, theta, nx=101, ny=101, addBias=False):
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
                state = torch.FloatTensor([x, y, np.cos(theta), np.sin(theta)], device=self.device).unsqueeze(0)
            else:
                z = max([l_x, g_x])
                state = torch.FloatTensor([x, y, np.cos(theta), np.sin(theta), z], device=self.device).unsqueeze(0)
            if addBias:
                v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
            else:
                d = q_func(state).item()
                v[idx] = d*l_x + (1.0 - d)*g_x
                # v[idx] = max(g_x, min(l_x, q_func(state).item()))
            it.iternext()
        return v

    def get_margins(self, obs):
        s_margin = self.safety_margin(obs)
        t_margin = self.target_margin(obs)
        return t_margin, s_margin

# == Trajectory Functions ==
    def simulate_one_trajectory(self, policy, T=10, state=None, theta=None, toEnd=False):

        if state is None:
            state = self.sample_random_state(theta=theta)
        x, y = state[:2]
        traj_x = [x]
        traj_y = [y]
        margin_g = []
        margin_l = []
        result = 0  # not finished

        for t in range(T):
            if toEnd:
                done = not self.check_within_bounds(state)
                if done:
                    result = 1
                    break
            else:
                l = self.target_margin(state[:2])
                g = self.safety_margin(state[:2])
                margin_l.append(l)
                margin_g.append(g)
                if g > 0:
                    result = -1  # failed
                    break
                elif l <= 0:
                    result = 1  # succeeded
                    break

            state_tensor = torch.FloatTensor(state, device=self.device).unsqueeze(0)
            if self.discrete:
                action_index = policy.logits_net(state_tensor).max(dim=1)[1].item()
                u = self.discrete_controls[action_index]
            else:
                u = policy.mu_net(state_tensor).detach()

            state, _ = self.integrate_forward(state, u)
            traj_x.append(state[0])
            traj_y.append(state[1])

        if len(margin_l) > 0:
            rollout_vals = core.discount_minmax_overtime(margin_l, margin_g, 0.9999)

        return traj_x, traj_y, result, rollout_vals

    def simulate_trajectories(self, policy, T=10, num_rnd_traj=None, states=None, theta=None, toEnd=False):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []
        traj_vals = []

        # plt.figure(2)
        if states is None:
            results = np.empty(shape=(num_rnd_traj,), dtype=int)
            for idx in range(num_rnd_traj):
                traj_x, traj_y, result, rollout_vals = self.simulate_one_trajectory(  policy, T=T, theta=theta,
                                                                                    toEnd=toEnd)
                # plt.plot(traj_x, traj_y)
                trajectories.append((traj_x, traj_y))
                traj_vals.append(rollout_vals)
                results[idx] = result
        else:
            results = np.empty(shape=(len(states),), dtype=int)
            for idx, state in enumerate(states):
                traj_x, traj_y, result, rollout_vals = self.simulate_one_trajectory(policy, T=T, state=state, toEnd=toEnd)
                trajectories.append((traj_x, traj_y))
                traj_vals.append(rollout_vals)
                results[idx] = result

        # import pdb
        # plt.show()
        # import pdb
        # pdb.set_trace()
        return trajectories, results, traj_vals

# == Plotting Functions ==
    def render(self):
        pass

    def visualize(  self, q_func, policy, no_show=False,
                    vmin=-1, vmax=1, nx=101, ny=101, cmap='seismic',
                    labels=None, boolPlot=False, addBias=False, theta=np.pi/2,
                    rndTraj=False, num_rnd_traj=15):
        """ Overlays analytic safe set on top of state value function.

        Args:
            q_func: NN or Tabular-Q
        """
        plt.close()
        axStyle = self.get_axes()
        thetaList = [0, np.pi/2, np.pi]  # [np.pi/6, np.pi/3, np.pi/2]
        # numX = 1
        # numY = 3
        # if self.axes is None:
        #     self.fig, self.axes = plt.subplots(
        #         numX, numY, figsize=(4*numY, 4*numX), sharex=True, sharey=True)
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        axList = [ax1, ax2, ax3]

        for i, (ax, theta) in enumerate(zip(axList, thetaList)):
        # for i, (ax, theta) in enumerate(zip(self.axes, thetaList)):
            ax.cla()
            if i == len(thetaList)-1:
                cbarPlot=True
            else:
                cbarPlot=False

            #== Plot failure / target set ==
            self.plot_target_failure_set(ax)

            #== Plot reach-avoid set ==
            self.plot_reach_avoid_set(ax, orientation=theta)

            #== Plot V ==
            self.plot_v_values( q_func, ax=ax, fig=fig, theta=theta,
                                vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
                                boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias)
            #== Formatting ==
            self.plot_formatting(ax=ax, labels=labels)

            #== Plot Trajectories ==
            if rndTraj:
                self.plot_trajectories( policy, T=300, num_rnd_traj=num_rnd_traj, theta=theta,
                                        toEnd=False,
                                        ax=ax, c='y', lw=2, orientation=0)
            else:
                # `visual_initial_states` are specified for theta = pi/2. Thus,
                # we need to use "orientation = theta-pi/2"
                self.plot_trajectories( policy, T=300, states=self.visual_initial_states, toEnd=False,
                                        ax=ax, c='y', lw=2, orientation=theta-np.pi/2)

            ax.set_xlabel(r'$\theta={:.0f}^\circ$'.format(theta*180/np.pi), fontsize=28)

        plt.tight_layout()
        plt.pause(0.01)
        # plt.show()

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

    def plot_v_values(  self, q_func, theta=np.pi/2, ax=None, fig=None, vmin=-1, vmax=1, nx=201, ny=201, cmap='seismic', boolPlot=False, cbarPlot=True, addBias=False):
        axStyle = self.get_axes()
        ax.plot([0., 0.], [axStyle[0][2], axStyle[0][3]], c='k')
        ax.plot([axStyle[0][0], axStyle[0][1]], [0., 0.], c='k')

        #== Plot V ==
        if theta == None:
            theta = 2.0 * np.random.uniform() * np.pi
        v = self.get_value(q_func, theta, nx, ny, addBias=addBias)

        if boolPlot:
            im = ax.imshow(v.T>0., interpolation='none', extent=axStyle[0], origin="lower", cmap=cmap)
        else:
            im = ax.imshow( v.T, interpolation='none', extent=axStyle[0], origin="lower",
                            cmap=cmap, vmin=vmin, vmax=vmax)
            if cbarPlot:
                cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax])
                cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)

        xs = np.linspace(self.bounds[0,0], self.bounds[0,1], nx)
        ys = np.linspace(self.bounds[1,0], self.bounds[1,1], ny)
        X, Y = np.meshgrid(xs, ys)
        ax.contour(X, Y, v.T, levels=[-0.01], colors=('k',),
                   linestyles=('--',), linewidths=(1,))

    def plot_trajectories(  self, q_func, T=10, num_rnd_traj=None, states=None, theta=None,
                            toEnd=False, ax=None, c='y', lw=1.5, orientation=0):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))

        if states != None:
            tmpStates = []
            for state in states:
                x, y, cos_theta, sin_theta = state
                theta = np.arctan2(sin_theta, cos_theta)
                xtilde = x*np.cos(orientation) - y*np.sin(orientation)
                ytilde = y*np.cos(orientation) + x*np.sin(orientation)
                thetatilde = theta+orientation
                tmpStates.append(np.array([xtilde, ytilde,
                                 np.cos(thetatilde), np.sin(thetatilde)]))
            states = tmpStates

        trajectories, results, traj_vals = self.simulate_trajectories(
                                    q_func, T=T, num_rnd_traj=num_rnd_traj,
                                    states=states, theta=theta,
                                    toEnd=toEnd)

        if ax == None:
            ax = plt.gca()
        all_vals = np.concatenate(traj_vals)
        norm = plt.Normalize(-1, 1) #self.target_radius, self.target_radius)  # all_vals.max())
        for traj, vals in zip(trajectories, traj_vals):
            traj_x, traj_y = traj
            x_ini, y_ini = traj_x[0], traj_y[0]
            points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # norm = plt.Normalize(vals.min(), vals.max())
            lc = LineCollection(segments, cmap='seismic', norm=norm)
            lc.set_array(vals)
            lc.set_linewidth(2)
            ax.scatter(x_ini, y_ini, c='k')
            ax.add_collection(lc)
            # import pdb
            # pdb.set_trace()
            # ax.scatter(traj_x[0], traj_y[0], s=48, c=vals[0])#c=c)
            # for x, y, c in zip(traj_x, traj_y, vals):
            #     ax.plot(traj_x, traj_y, color=c,  linewidth=lw)

        return results

    def plot_reach_avoid_set(self, ax, c='g', lw=3, orientation=0):
        r = self.target_radius
        R = self.constraint_radius
        R_turn = self.R_turn
        if r >=  2*R_turn - R:
            # plot arc
            tmpY = (r**2 - R**2 + 2*R_turn*R) / (2*R_turn)
            tmpX = np.sqrt(r**2 - tmpY**2)
            tmpTheta = np.arcsin(tmpX / (R-R_turn))
            # two sides
            self.plot_arc((0.,  R_turn), R-R_turn, (tmpTheta-np.pi/2, np.pi/2),  ax, c=c, lw=lw, orientation=orientation)
            self.plot_arc((0., -R_turn), R-R_turn, (-np.pi/2, np.pi/2-tmpTheta), ax, c=c, lw=lw, orientation=orientation)
            # middle
            tmpPhi = np.arcsin(tmpX/r)
            self.plot_arc((0., 0), r, (tmpPhi - np.pi/2, np.pi/2-tmpPhi), ax, c=c, lw=lw, orientation=orientation)
            # outer boundary
            self.plot_arc((0., 0), R, (np.pi/2, 3*np.pi/2), ax, c=c, lw=lw, orientation=orientation)
        else:
            # two sides
            tmpY = (R**2 + 2*R_turn*r - r**2) / (2*R_turn)
            tmpX = np.sqrt(R**2 - tmpY**2)
            tmpTheta = np.arcsin( tmpX / (R_turn-r))
            self.plot_arc((0.,  R_turn), R_turn-r, (np.pi/2+tmpTheta, 3*np.pi/2), ax, c=c, lw=lw, orientation=orientation)
            self.plot_arc((0., -R_turn), R_turn-r, (np.pi/2, 3*np.pi/2-tmpTheta), ax, c=c, lw=lw, orientation=orientation)
            # middle
            self.plot_arc((0., 0), r, (np.pi/2, -np.pi/2), ax, c=c, lw=lw, orientation=orientation)
            # outer boundary
            self.plot_arc((0., 0), R, (np.pi/2, 3*np.pi/2), ax, c=c, lw=lw, orientation=orientation)

    def plot_target_failure_set(self, ax):
        self.plot_circle(self.constraint_center, self.constraint_radius, ax, c='k', lw=3)
        self.plot_circle(self.target_center,     self.target_radius, ax, c='m', lw=3)

    def plot_arc(self, p, r, thetaParam, ax, c='b', lw=1.5, orientation=0):
        x, y = p
        thetaInit, thetaFinal = thetaParam

        xtilde = x*np.cos(orientation) - y*np.sin(orientation)
        ytilde = y*np.cos(orientation) + x*np.sin(orientation)

        theta = np.linspace(thetaInit+orientation, thetaFinal+orientation, 100)
        xs = xtilde + r * np.cos(theta)
        ys = ytilde + r * np.sin(theta)

        ax.plot(xs, ys, c=c, lw=lw)

    def plot_circle(self, center, r, ax, c='b', lw=1.5, orientation=0, scatter=False):
        x, y = center
        xtilde = x*np.cos(orientation) - y*np.sin(orientation)
        ytilde = y*np.cos(orientation) + x*np.sin(orientation)

        theta = np.linspace(0, 2*np.pi, 200)
        xs = xtilde + r * np.cos(theta)
        ys = ytilde + r * np.sin(theta)
        ax.plot(xs, ys, c=c, lw=lw)
        if scatter:
            ax.scatter(xtilde+r, ytilde, c=c, s=80)
            ax.scatter(xtilde-r, ytilde, c=c, s=80)
            print(xtilde+r, ytilde, xtilde-r, ytilde)