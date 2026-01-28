import argparse
import os
import time
import numpy as np
import torch
import TD7
import floris_29_turbine_env_v1p4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_online(RL_agent, env, eval_env, args):
    evals = []
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset(visualize=False), False
    ep_total_reward, ep_timesteps, ep_num = np.zeros([eval_env.n_parallel, ]), 0, 1

    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)

        if allow_train:
            action = RL_agent.select_action(np.array(state))
        else:
            action = np.random.uniform(-env.max_action, env.max_action, [env.n_parallel, env.n_turbines])

            # action = RL_agent.select_action(np.array(state))

        next_state, reward, ep_finished, _ = env.step(action)

        ep_total_reward += reward
        ep_timesteps += 1

        done = float(ep_finished) if ep_timesteps < env._max_episode_steps else 0
        done = done * np.ones([env.n_parallel, ])

        RL_agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        if allow_train and not args.use_checkpoints:
            for i in range(int(env.n_parallel)):
                RL_agent.train()

        if ep_finished:
            # print(f'starting from the {env.day}th day in field measurements')
            # print(f'Wind speed: {env.wind_speed_profile[0, 0]}(m/s)  '
            #       f'direction: {env.wind_direction_profile[0, 0]}(deg)')
            print(f"Total T: {(t + 1) * env.n_parallel} Episode Num: {ep_num} Episode Steps: {ep_timesteps}"
                  f" Reward: {ep_total_reward.mean()}")
            if allow_train and args.use_checkpoints:
                RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

            if (t + 1) * env.n_parallel >= args.timesteps_before_training:
                allow_train = True

            state, done = env.reset(visualize=False), False
            ep_total_reward, ep_timesteps = np.zeros([eval_env.n_parallel, ]), 0
            ep_num += 1
            RL_agent.hp.exploration_noise = max(RL_agent.hp.exploration_noise * 0.95, 0.01)


def train_offline(RL_agent, env, eval_env, args):
    RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))

    evals = []
    start_time = time.time()

    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=True)
        RL_agent.train()


def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=False):
    if t % args.eval_freq == 0:
        print("---------------------------------------")
        print(f"Evaluation at {t} time steps")
        print(f"Total time passed: {round((time.time() - start_time) / 60., 2)} min(s)")

        total_reward = np.zeros([eval_env.n_parallel, ])
        WFenergy = 0
        for ep in range(args.eval_eps):
            state, done = eval_env.reset(visualize=False), False

            while not done:
                action = RL_agent.select_action(np.array(state), args.use_checkpoints, use_exploration=False)
                # if args.load_model:
                #     print(np.array(action))
                state, reward, done, WFpower = eval_env.step(action)
                WFenergy = WFenergy + np.sum(WFpower, 1) / 6
                total_reward += reward

        print(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f} WFenergy:{WFenergy.mean()} MWh")
        if d4rl:
            total_reward = eval_env.get_normalized_score(total_reward) * 100
            print(f"D4RL score: {total_reward.mean():.3f}")

        print("---------------------------------------")

        evals.append(total_reward)
        np.save(f"results/{args.file_name}", evals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="FloatingFarmYaw", type=str)
    parser.add_argument("--seed", default=20250715, type=int)
    parser.add_argument("--offline", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_checkpoints', default=False, action=argparse.BooleanOptionalAction)
    # Evaluation
    parser.add_argument("--timesteps_before_training", default=5e3, type=int)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--eval_eps", default=1, type=int)
    parser.add_argument("--max_timesteps", default=4e3, type=int)
    # save and load pretrained model
    parser.add_argument("--load_model", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--load_name", default="20250129", type=str)
    parser.add_argument("--save_name", default="20250715", type=str)
    # File
    parser.add_argument('--file_name', default=None)
    parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)
    args = parser.parse_args()

    if args.offline:
        import d4rl

        d4rl.set_dataset_path(args.d4rl_path)
        args.use_checkpoints = False

    if args.file_name is None:
        args.file_name = f"TD7_{args.env}_{args.seed}"

    if not os.path.exists("results"):
        os.makedirs("results")

    env = floris_29_turbine_env_v1p4.Environment(evaluation=True)
    eval_env = floris_29_turbine_env_v1p4.Environment(evaluation=True)
    env.reset(visualize=False)

    for i in range(1):
        args.seed += i
        torch.manual_seed(args.seed)
        args.file_name = f"TD7_{args.env}_{args.seed}"
        save_name = args.save_name + '-' + str(i)

        print("---------------------------------------")
        print(f"Algorithm: TD7, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        state_dim = env.observation_dim
        action_dim = env.action_dim
        max_action = env.max_action

        RL_agent = TD7.Agent(state_dim, action_dim, max_action, args)

        if args.offline:
            train_offline(RL_agent, env, eval_env, args)
        else:
            train_online(RL_agent, env, eval_env, args)

        RL_agent.save_model(save_name)
