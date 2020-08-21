# Implementing your First Learning Agent - Solving the Mountain Car problem

Well done on making it this far! In previous chapters, we got a good introduction to OpenAI Gym, its features, and how to install, configure, and use it in your own programs. We also discussed the basics of reinforcement learning and what deep reinforcement learning is, and we set up the PyTorch deep learning library to develop deep reinforcement learning applications. In this chapter, you will start developing your first learning agent! You will develop an intelligent agent that will learn how to solve the Mountain Car problem. Gradually in the following chapters, we will solve increasingly challenging problems as you get more comfortable developing reinforcement learning algorithms to solve problems in OpenAI Gym. We will start this chapter by understanding the Mountain Car problem, which has been a popular problem in the reinforcement learning and optimal control community. We will develop our learning agent from scratch and then train it to solve the Mountain Car problem using the Mountain Car environment in the Gym. We will finally see how the agent progresses and briefly look at ways we can improve the agent to use it for solving more complex problems. The topics we will be covering in this chapter are as follows:

* Understanding the Mountain Car problem
* Implementing a reinforcement learning-based agent to solve the Mountain Car problem
* Training a reinforcement learning agent at the Gym
* Testing the performance of the agent

### Understanding the Mountain Car problem

For any reinforcement learning problem, two fundamental definitions concerning the problem are important, irrespective of the learning algorithm we use. They are the definitions of the state space and the action space.. Typically, in most problems, the state space consists of continuous values and is represented as a vector, matrix, or tensor (a multi-dimensional matrix). Problems and environments with discrete action spaces are relatively easy compared to continuous valued problems and environments. In this book, we will develop learning algorithms for a few problems and environments with a mix of state space and action space combinations so that you are comfortable dealing with any such variation when you start out on your own and develop intelligent agents and algorithms for your applications. Let's start by understanding the Mountain Car problem with a high-level description, before moving on to look at the state and action spaces of the Mountain Car environment.

### The Mountain Car problem and environment

In the Mountain Car Gym environment, a car is on a one-dimensional track, positioned between two mountains. The goal is to drive the car up the mountain on the right; however, the car's engine is not strong enough to driveup the mountain even at the maximum speed. Therefore, the only way to succeed is to drive back and forth to build up momentum. In short, the Mountain Car problem is to get an under-powered car to the top of a hill. Before you implement your agent algorithm, it will help tremendously to understand the environment, the problem, and the state and action spaces. How do we find out the state and action spaces of the Mountain Car environment in the Gym? Well, we already know how to do that from Chapter 4 , Exploring the Gym and its Features. We wrote a script named get_observation_action_space.py , which will print out the state and observation and action spaces of the environment whose name is passed as the first argument to the script. Let's ask it to print the spaces for the MountainCar-v0 environment with the following command:

```Linux
~$pytho3 get_observation_action_space.py 'MountainCar-v0'
```

We know that state and observation space are two. The dimensional box and motion space are three-dimensional and discrete.

The state and action space type, description, and range of allowed values are summarized in the following table for your reference:

| MountainCar-v0 environment  | Type     | Description  | Range  |
|-----------------------------|----------|--------------|--------|
| State space                 | Box(2,)  | (position,velocity) | Position: -1.2 to 0.6, Velocity: -0.07 to 0.07 |
| Action space | Discrete(3) | 0: Go left, 1: Coast/do-nothing, 2: Go right | 0, 1, 2 |

So for example, the car starts at a random position between -0.6 and -0.4 with zero velocity, and the goal is to reach the top of the hill on the right side, which is at position 0.5. (The car can technically go beyond 0.5, up to 0.6, which is also considered.) The environment will send -1 as a reward everytime step until the goal position (0.5) is reached. The environment will terminate the episode. The done variable will be equal to True if the car reaches the 0.5 position or the number of steps taken reaches 200.

### Implementing a Q-learning agent from scratch

In this section, we will start implementing our intelligent agent step-by-step. We will be implementing the famous Q-learning algorithm using the NumPy library and the MountainCar-V0 environment from the OpenAI Gym library.

```python
#!/usr/bin/env python
import gym
env = gym.make("Qbert-v0")
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

for episode in range(MAX_NUM_EPISODES):
    obs = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action) 
    obs = next_state
    if done is True:
        print("\n Episode #{} ended in {} steps.".format(episode, step+1))
        break
```

This code is a good starting point (aka boilerplate!) for developing our reinforcement learning agent. We will first start by changing the environment name from Qbert-v0 to MountainCar-v0 . Notice in the preceding script that we are setting MAX_STEPS_PER_EPISODE . This is the number of steps or actions that the agent can take before the episode ends. This may be useful in continuing, perpetual, or looping environments, where the environment itself does not end the episode. Here, we set a limit for the agent to avoid infinite loops. However, most of the environments defined in OpenAI Gym have an episode termination condition and once either of them is satisfied, the done variable returned by the env.step(...) function will be set to True. We saw in the previous section that for the Mountain Car problem we are interested in, theenvironment will terminate the episode if the car reaches the goal position (0.5) or if the number of steps taken reaches 200. Therefore, we can further simplify the boilerplate code to look like the following for the Mountain Car environment:

```python
#!/usr/bin/env python
import gym
env = gym.make("MountainCar-v0")
MAX_NUM_EPISODES = 5000

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = env.reset()
    total_reward = 0.0
    step = 0
    while not done:
        env.render()
        action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        environment and receive the next_state, reward and whether done or not
        total_reward += reward
        step += 1
        obs = next_state
        
    print("\n Episode #{} ended in {} steps. total_reward={}".format(episode,
    
step+1, total_reward))
env.close()
```

If you run the preceding script, you will see the Mountain Car environment come up in a new window and the car moving left and right randomly for 1,000 episodes. You will also see the episode number, steps taken, and the total reward obtained printed at the end of every episode, as shown in the following screenshot:

![index](/images_gif/as/a.png)

The sample output should look similar to the following screenshot:

![index](/images_gif/as/1.gif)

You should recall from our previous section that the agent gets a reward of -1 for each step and that the MountainCar-v0 environment will terminate the episode after 200 steps; this is why you the agent may sometimes get a total reward of -200! After all, the agent is taking random actions without thinking or learning from its previous actions. Ideally, we would want the agent to figure out how to reach the top of the mountain (near the flag, close to, at, or beyond position 0.5) with the minimum number of steps. Don't worry - we will build such an intelligent agent by the end of this chapter! Let's move on by having a look at what Q-learning section.

### A simple and complete Q-Learner implementation for solving the Mountain Car problem

In this section, we will put together the whole code into a single Python script to initialize the environment, launch the agent's training process, get the trained policy, test the performance of the agent, and also record how it acts in the environment!

```python
import gym
import numpy as np

MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200 #  This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
NUM_DISCRETE_BINS = 30  # Number of bins to Discretize each observation dim


class Q_Learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS  # Number of bins to Discretize each observation dim
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        # Create a multi-dimensional array (aka. Table) to represent the
        # Q-values
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1,
                           self.action_shape))  # (51 x 51 x 3)
        self.alpha = ALPHA  # Learning rate
        self.gamma = GAMMA  # Discount factor
        self.epsilon = 1.0

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        # Epsilon-Greedy action selection
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        else:  # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[discretized_next_obs])
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error

def train(agent, env):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                     total_reward, best_reward, agent.epsilon))
    # Return the trained policy
    return np.argmax(agent.Q, axis=2)


def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    agent = Q_Learner(env)
    learned_policy = train(agent, env)
    # Use the Gym Monitor wrapper to evalaute the agent and record video
    gym_monitor_path = "./gym_monitor_output"
    env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()
```

If you run the python script called Q_learner_MountainCar.py, the tutorial starts. and you can test later.

![index](/images_gif/as/1.gif) ![index](/images_gif/as/2.gif)

