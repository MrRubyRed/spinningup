import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import gym
import time
import matplotlib.pyplot as plt
import spinup.algos.pytorch.ra_ppo.ra_core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class RA_PPOBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.l_x_buf = np.zeros(size, dtype=np.float32)
        self.g_x_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, l_x=None, g_x=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.l_x_buf[self.ptr] = l_x
        self.g_x_buf[self.ptr] = g_x
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=None):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # if last_val is None:
        #     l_xs = np.append(self.l_x_buf[path_slice], self.l_x_buf[path_slice][-1])
        # else:
        #g_xs = np.append(self.g_x_buf[path_slice], self.g_x_buf[path_slice][-1])
        # l_xs = np.append(self.l_x_buf[path_slice], last_val)
        # g_xs = np.append(self.g_x_buf[path_slice], last_val)
        l_xs = self.l_x_buf[path_slice]
        g_xs = self.g_x_buf[path_slice]

        # the next two lines implement GAE-Lambda advantage calculation
        # deltas = (rews[:-1] + self.gamma * vals[1:]) - vals[:-1]
        # self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf[path_slice] = core.discount_minmax_overtime(l_xs, g_xs, self.gamma, v=vals[-1]) - vals[:-1]
        # self.adv_buf[path_slice] = core.discount_minmax_overtime(l_xs, g_xs, 1.0, v=vals[-1]) - vals[:-1]
        # self.adv_buf[path_slice] = core.discount_min_overtime(l_xs, self.gamma)

        # the next line computes rewards-to-go, to be targets for the value function
        # self.ret_buf[path_slice] = self.adv_buf[path_slice].copy()
        self.ret_buf[path_slice] = core.discount_minmax_overtime(l_xs, g_xs, self.gamma, v=vals[-1])#[:-1]
        # self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-7)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, l_x=self.l_x_buf,
                    g_x=self.g_x_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def ra_ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),  seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Vanilla Policy Gradient

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = RA_PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    def warm_start():
        pass
        # Get examples
        x = np.zeros(core.combined_shape(4000, obs_dim), dtype=np.float32)
        y = np.zeros(4000, dtype=np.float32)
        for i in range(4000):
            o = env.reset()
            state_sim = env.obs_scale_to_simulator_scale(o)
            s_margin = env.safety_margin(state_sim)
            t_margin = env.target_margin(state_sim)
            x[i] = o
            y[i] = s_margin
        data = dict(obs=x, ret=y)
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        # Value function learning
        for i in range(25000):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            if i % 500 == 0:
                print(loss_v)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
        print(loss_v)

    # Set up function for computing VPG policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = (torch.max(ratio * adv, clip_adv)).mean() # Removed "-" since we are minimizing.
        # loss_pi = (logp * adv).mean()  # Removed "-" since we are minimizing.

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        l_x, g_x = data['l_x'], data['g_x']
        d = ac.v(obs)
        output = d*l_x + (1.0-d)*g_x
        return ((output - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    # pi_optimizer = RMSprop(ac.pi.parameters(), lr=pi_lr)
    # vf_optimizer = RMSprop(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                print("KL = ", kl, " | Target_KL = ", target_kl)
                logger.log('Early stopping at grad step %d - reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o = env.reset()
    ep_ret = np.inf
    max_viol = -np.inf
    ep_len = 0

    # Warm start
    # warm_start()

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        if epoch % 25 == 0:
            env.visualize(ac.v, ac.pi)
        num_resets = 0
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, info = env.step(a)
            assert ("g_x" in info and "l_x" in info)
            # Correct the values just in case.
            # v = max(info["g_x"], min(info["l_x"], v))
            v = v*info["l_x"] + (1.0-v)*info["g_x"]

            # Used for logging.
            max_viol = max(max_viol, info["g_x"])
            ep_ret = min(ep_ret, max(info["l_x"], max_viol))
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp, l_x=info["l_x"], g_x=info["g_x"])
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = (ep_len == max_ep_len)
            terminal = (d or timeout)
            epoch_ended = (t == local_steps_per_epoch-1)

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    t_margin, s_margin = env.get_margins(o)
                    # v = max(s_margin, min(v, t_margin))
                    v = v*t_margin + (1.0-v)*s_margin
                    # buf.finish_path(v)
                    # print("uno")
                else:
                    v = max(info["g_x"], info["l_x"])
                    # buf.finish_path()
                    # print("dos", timeout, epoch_ended, info["g_x"], info["l_x"], v)
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, max_viol, ep_len = env.reset(), np.inf, -np.inf, 0
                num_resets += 1


        # Save model.
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('NumResets', num_resets)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        # if (epoch % 25 == 0) and (epoch > 0):
        #     # results = env.simulate_trajectories(
        #     #     ac.pi, T=local_steps_per_epoch // 10, num_rnd_traj=100)[1]
        #     env.visualize(ac.v, ac.pi)#, rndTraj=True)  # , T=local_steps_per_epoch)
        #     # print("Percent reached = ", np.sum(results == 1))

        #     # # Show policy in velocity 0 space.
        #     # env.scatter_actions(ac.pi, num_states=200)
        #     # plt.show()

        #     # print(ac.pi.logits_net[0].weight[:6])

        #     # my_images = []
        #     # # fig, ax = plt.subplots(figsize=(12, 7))
        #     # s_trajs = []
        #     # total_reward = 0
        #     # o = env.reset(zero_vel=True)#state_in=env.visual_initial_states[0])
        #     # tmp_int = 0
        #     # tmp_ii = 0
        #     # while True:
        #     #     action, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
        #     #     o, r, done, info = env.step(action)
        #     #     # state_sim = env.obs_scale_to_simulator_scale(s)
        #     #     # s_margin = env.safety_margin(state_sim)
        #     #     # t_margin = env.target_margin(state_sim)
        #     #     # print("S_Margin: ", s_margin, " | T_Margin: ", t_margin, " | reward: ", r)
        #     #     # s_trajs.append([s[0], s[1]])
        #     #     total_reward += r
        #     #     tmp_ii += 1

        #     #     my_images.append(env.render(mode="rgb_array"))

        #     #     if done or tmp_ii > 1000:
        #     #       tmp_ii = 0
        #     #       # o = env.reset()
        #     #       o = env.reset(zero_vel=True)#state_in=env.visual_initial_states[0])
        #     #       if tmp_int > 10:
        #     #         break
        #     #       else:
        #     #         tmp_int += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='zermelo_cont-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--ep_len', type=int, default=150)
    parser.add_argument('--exp_name', type=str, default='ra_ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # print("========== EEEFEF ============")

    ra_ppo(
        lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, activation=nn.Tanh), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        max_ep_len=args.ep_len, logger_kwargs=logger_kwargs)
