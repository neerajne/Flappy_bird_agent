import flappy_bird_gymnasium
import gymnasium as gym
import random
from dqn import DQN
from experienceReplay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            # creating dictionary of all parameters
            all_param_set = yaml.safe_load(f)

            # particular key value inside dictionary
            params = all_param_set["flappybirdv0"]

            # from that particular dictionary now taking out partiular values
            self.alpha = params["alpha"]
            self.gamma = params["gamma"]

            self.epsilon_init = params["epsilon_init"]
            self.epsilon_min = params["epsilon_min"]
            self.epsilon_decay = params["epsilon_decay"]

            self.replay_memory_size = params["replay_memory_size"]
            self.mini_batch_size = params["mini_batch_size"]

            self.reward_threshold = params["reward_threshold"]
            self.network_sync_rate = params["network_sync_rate"]

            self.mini_batch_size = params["mini_batch_size"]

        # loss func
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):

        # Define Environment
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # training dqn
        policy_dqn = DQN(num_states, num_actions)

        # Memory_Size
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions)

            # copy the weight and bias from policy_dqn to target_dqn
            target_dqn.load_state_dict(
                policy_dqn.state_dict()
            )  # load_state_dict har nn object ke sath inbuilt pehle se hota h

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_reward = -float("inf")

        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float)

            episode_reward = 0
            terminated = False

            # while my agent does not gets terminated
            while not terminated and episode_reward < self.reward_threshold:

                # explore
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long)

                else:  # exploit
                    with torch.no_grad():
                        # becauase already lia hua optimal step le rhe h
                        # no learning here so gradients compute hi koi jarrt nhi h
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                next_state, reward, terminated, _, _ = env.step(action.item())

                # create tensors to pass in the experience
                reward = torch.tensor(reward, dtype=torch.float)
                next_state = torch.tensor(next_state, dtype=torch.float)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                episode_reward += reward.item()
                state = next_state

            print(
                f"episode {episode+1} with total rewards = "
                f"{episode_reward} & epsilon_value= {epsilon}"
            )

            # epsilon decay ise kam krte jayenge because we want our agent
            # to exploit instead of exploring as agent goes forward
            if is_training:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    print(f"Model saved with reward {best_reward}")

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

            # batch size ke according data ko nikalo and then train it
            if is_training and len(memory) > self.mini_batch_size:

                # get sample
                mini_batch = memory.sample(self.mini_batch_size)

                # creating func to do fwd propagation,
                # bwd propagation, loss calculation
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # sync the network
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0  # reset

        # env.close() manually stop

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float()

        # calculate target Q-values -- if termination then next q-value is zero
        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.gamma
                * target_dqn(next_states).max(dim=1)[0]
            )

        # calculate y_pred i.e. Q-value from current policy
        current_q = (
            policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )

        # compute loss
        loss = self.loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":

    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")

    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
