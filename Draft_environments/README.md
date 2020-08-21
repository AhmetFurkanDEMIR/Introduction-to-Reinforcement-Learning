# `Draft environments`

* Important !!! : Install the modules in requirements.txt before starting this step.

* As I mentioned before, we will use the gym module while creating the environments.

```python
import gym
env = gym.make("MountainCar-v0") # Create a MountainCar environment
env.reset()
for _ in range(2000): # Run for 2000 steps
env.render()
env.step(env.action_space.sample()) # Send a random action
```

* When you run this code, the environment will be created and your agent will make random moves.


### Complete install of OpenAI Gym learning environments

Not all environments are usable with the minimal installation. To be able to use most or all the environments available in the Gym, we will go through the installation of the dependencies and build OpenAI Gym from the latest source code on the master branch. To get started, we will need to install the required system packages first.

* Instructions for Ubuntu-Pardus

Let's install the system packages needed by running the following command
on the Terminal/console:

```Linux
~$sudo apt-get update
~$sudo apt-get install -y build-essential cmake python-dev python-numpy python-opengl libboost-all-dev zlib1g-dev libsdl2-dev libav-tools xorg-dev libjpeg-dev swig
```

This command will install the prerequisite system packages. Note that the -y flag will automatically say yes to confirm the installation of the package, without asking you to confirm manually. If you want to review the packages that are going to be installed for some reason, you may run the command without the flag.

* Completing the OpenAI Gym setup

Let's update our version of pip first:

```Linux
~$python3 -m pip install --upgrade pip
```

Then, let's download the source code of OpenAI Gym from the GitHub
repository into our home folder:

```Linux
~$git clone https://github.com/openai/gym.git
~$cd gym
```

By running the command below you can have all the environments of OpenAI gym.

```Linux
~$pip3 install -e '.[all]'
```

Intelligent Agents and Learning Environments.

| Environment               |  Installation command                |
|---------------------------|--------------------------------------|
|Atari                      | pip33 install -e '.[atari]'          |
|Box2D                      | pip3 install -e '.[box2d]'           |
|Classic control            | pip3 install -e '.[classic_control]' |
MuJoCo (requireslicense)    | pip3 install -e '.[mujoco]'          |
Robotics (requires license) | pip3 install -e '.[robotics]'        |

You can access the environment we just downloaded by running the code(test_box2d.py) snippet below.

```python
#!/usr/bin/env python
import gym
env = gym.make('BipedalWalker-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
```

![index](/images_gif/as/ezgif.com-video-to-gif.gif)

* Proceed to the NVIDIA website and install the NVIDIA driver, CUDA and cuDNN software.

* Scroll down to the PyTorch website and install the Python PyTorch-GPU version.

### Exploring the Gym and its Features

Now that you have a working setup, we will start exploring the various
features and options provided by the Gym toolkit. This chapter will walk you through some of the commonly used environments, the tasks they solve, and what it would take for your agent to master a task. In this chapter, we will explore the following topics

#### Exploring the list of environments and nomenclature :

Let's start by picking an environment and understanding the Gym interface.
You may already be familiar with the basic function calls to create a Gym
environment from the previous chapters, where we used them to test our
installations. Here, we will formally go through them.
We can now use the gym.make method to create an environment from the
available list of environments. You may be asking how to find the list of Gym environments available on your system. We will create a small utility script to generate the list of environments so that you can refer to it later when you need to.

**list_gym_envs.py**

```python
#!/usr/bin/env python
from gym import envs
env_names = [spec.id for spec in envs.registry.all()]
for name in sorted(env_names):
    print(name)
```

This script will print the names of all the environments available through
your Gym installation, sorted alphabetically. You can run this script using the following command to see the names of the environments installed and available in your system:

```Linux
~$python3 list_gym_envs.py
```

The environments installed on your system will be displayed on the console.

![index](/images_gif/Screenshot_2020-08-21_10-50-34.png)

I have 859 media installed.

##### Nomenclature

The presence of the word ram in the environment name means that the
observation returned by the environment is the contents of the Random
Access Memory (RAM) of the Atari console on which the game was
designed to run. The presence of the word deterministic in the environment names means that the actions sent to the environment by the agent are performed repeatedly for a deterministic/fixed duration of four frames, and then the resulting state is returned. The presence of the word NoFrameskip means that the actions sent to the environment by the agent are performed once and the resulting state is returned immediately, without skipping any frames in-between. By default, if deterministic and NoFrameskip are not included in the environment name, the action sent to the environment is repeatedly performed for a duration of n frames, where n is uniformly sampled from {2,3,4}. The letter v followed by a number in the environment name represents the version of the environment. This is to make sure that any change to the environment implementation is reflected in its name so that the results obtained by an algorithm/agent in an environment are comparable to the results obtained by another algorithm/agent without any discrepancies. Let's understand this nomenclature by looking at the Atari Alien environment. The various options available are listed with a description as follows:


| Version name           |  Description                                |
|------------------------|---------------------------------------------|
| Alien-ram-v0           | Observation is the RAM contents of the Atari machine with a total size of 128 bytes and the action sent to the environment is repeatedly performed for aduration of n frames, where n is uniformly sampled from {2,3,4}.  |
| Alien-ram-v4           | Observation is the RAM contents of the Atari machine with a total size of 128 bytes and the action sent to the environment is repeatedly performed for a duration of n frames, where n is uniformly sampled from {2,3,4}. There's some modification in the environment compared to v0.
| Alien-ramNoFrameskip-v0 | Observation is the RAM contents of the Atari machine with a total size of 128 bytes and the action sent to the environment is applied, and the resulting state is returned immediately without skipping any frames. |
| Alien-v4          | Observation is an RGB image of the screen represented as an array of shape (210, 160, 3) and the action sent to the environment is repeatedly performed for a duration of n frames, where n is uniformly sampled from {2,3,4}. There's some modification in the environment compared to v0. |
| AlienNoFrameskip-v4   | Observation is an RGB image of the screen represented as an array of shape (210, 160, 3) and the action sent to the environment is applied, and the resulting state is returned immediately without skipping any frames. any frames. There's some modification in the environment compared to v0.

This summary should help you understand the nomenclature of the environments, and it applies to all environments in general. The RAM may be specific to the Atari environments, but you now have an idea of what to expect when you see several related environment names.

#### Exploring the Gym environments

To make it easy for us to visualize what an environment looks like or what its task is, we will make use of a simple script that can launch any environment and step through it with some randomly sampled actions.

We will use the python script file run_gym_env.py.

```python
#!/usr/bin/env python
import gym
import sys

def run_gym_env(argv):
    env = gym.make(argv[1]) 
    env.reset()
    for _ in range(int(argv[2])):
        env.render()
        env.step(env.action_space.sample())
    env.close()
    
if __name__ == "__main__":
run_gym_env(sys.argv)
```
The name of the environment to be worked on and how many cycles to run are taken as an argument. We will use Alien-ram-v0 as environment and our agent will make 2000 moves.

```Linux
~$python3 run_gym_env.py Alien-ram-v0 2000
```

![index](/images_gif/Screenshot_2020-08-21_11-14-17.png)

After the environment is started and our agent makes 2000 moves, the environment is closed.

#### Understanding the Gym interface

After we import gym, we make an environment using the following line of code:

```python
import gym
env = gym.make("ENVIRONMENT_NAME")
```

Here, ENVIRONMENT_NAME is the name of the environment we want, chosen from
the list of the environments we found installed on our system.

We get that first observation from the environment by calling env.reset() . Let's store the observation in a variable named obs using the following line of code:

```python
obs = env.reset()
```

Now, the agent has received the observation (the end of the first arrow). It's
time for the agent to take an action and send the action to the environment to
see what happens. In essence, this is what the algorithms we develop for the agents should figure out! We'll be developing various state-of-the-art algorithms to develop agents in the next and subsequent chapters. Let's continue our journey towards understanding the Gym interface. Once the action to be taken is decided, we send it to the environment (second arrow in the diagram) using the env.step() method, which will return four values in this order: next_state , reward , done , and info :

* The next_state is the resulting state of the environment after the action was taken in the previous state.
* The reward (third arrow in the diagram) is returned by the environment.
* The done variable is a Boolean (true or false), which gets a value of true if the episode has terminated/finished (therefore, it is time to reset the environment) and false otherwise. This will be useful for the agent to know when an episode has ended or when the environment is going to be reset to some initial state.
* The info variable returned is an optional variable, which some environments may return with some additional information. Usually, this is not used by the agent to make its decision on which action to take.

Here is a consolidated summary of the four values returned by a Gym
environment's step() method, together with their types and a concise
description about them:

| Returned value    | Type     |  Description                  |
|-------------------|----------|--------------------------------------|
|next_state(or      | Object   | Observation returned by the environment. The object could be next_statethe RGB pixel data from the screen/camera, RAM contents, join angles and join velocities of a robot, and so on, depending on the environment. |
| reward            | Float   | Reward for the previous action that was sent to the environment. The range of the Float value varies with each environment, but irrespective of the environment, a higher reward is always better and the goal of the agent should be to maximize the total reward. |
| done              | Boolean  | Indicates whether the environment is going to be reset in the next step. When the Boolean value is true, it most likely means that the episode has ended (due to loss of life of the agent, timeout, or some other episode termination criteria). |
| info              | Dict     | Some additional information that can optionally be sent out by an environment as a dictionary of arbitrary key-value pairs. The agent we develop should not rely on any of the information in this dictionary for taking action. It may be used (if available) for debugging purposes. |

Let's put all the pieces together and look at them in one place:

```python
import gym
env = gym.make("ENVIRONMENT_NAME")
obs = env.reset() # The first arrow in the picture
# Inner loop (roll out)
action = agent.choose_action(obs) # The second arrow in the picture
next_state, reward, done, info = env.step(action) # The third arrow (and more)
obs = next_state
# Repeat Inner loop (roll out)
```

I hope you got a good understanding of one cycle of the interaction between the environment and the agent. This process will repeat until we decide to terminate the cycle after a certain number of episodes or steps have passed. Let's now have a look at a complete example with the inner loop running for MAX_STEPS_PER_EPISODE and the outer loop running for MAX_NUM_EPISODES in a Qbert-v0 environment:

```python
import gym
env = gym.make("Qbert-v0")
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

for episode in range(MAX_NUM_EPISODES):
    obs = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        env.render()
        action = env.action_space.sample()
        obs = next_state
        if done is True:
        print("\n Episode #{} ended in {} steps.".format(episode, step+1))
        break
```

```linux
~$Python3 rl_gym_boilerplate_code.py
```

When you run this script, you will notice a Qbert screen pop up and Qbert
taking random actions and getting a score, as shown here:

![index](/images_gif/as/ezgif.com-video-to-gif(1).gif)

## Spaces in the Gym

We can see that each environment in the Gym is different. Every game environment under the Atari category is also different from the others. For example, in the case of the VideoPinball-v0 environment, the goal is to keep bouncing a ball with two paddles to collect points based on where the ball hits, and to make sure that the ball never falls below the paddles, whereas in the case of Alien-v0 , which is another Atari game environment, the goal is to move through a maze (the rooms in a ship) collecting dots, which are equivalent to destroying the eggs of the alien. Aliens can be killed by collecting a pulsar dot and the reward/score increases when that happens. Do you see the variations in the games/environments? How do we know what types of actions are valid in a game? In the VideoPinball environment, naturally, the actions are to move the paddles up or down, whereas in the Alien environment, the actions are to command the player to move left, right, up, or down. Note that there is no "move left" or "move right" action in the case of VideoPinball. When we look at other categories of environment, the variations are even greater. For example, in the case of continuous control environments such as recently release robotics environments with the fetch robot arms, the action is to vary the continuous valued join positions and joint velocities to achieve the task. The same discussion can be had with respect to the values of the observations from the environment. We already saw the different observation object types in the case of Atari (RAM versus RGB images). This is the motivation for why the spaces (as in mathematics) for the observation and actions are defined for each environment. They are listed in this table, with a brief description of each:

| type          | Description               |  Usage Example             |
|---------------|---------------------------|----------------------------|
| Box           | A box in the space (an n-dimensional box) where each coordinate is bounded to lie in the interval defined by [low,high]. Values will be an array of n numbers. The shape defines the n for the space.          | gym.spaces.Box(low=-100, high=100, shape= (2,)) |
| Discrete      | Discrete, integer- value space in the interval [0,n-1]. The argument for Discrete() defines n. | gym.spaces.Discrete(4)  |
| Dict          | A dictionary of sample space to create arbitrarily complex space. In the example, a Dict space is created, which consists of two discrete spaces for positions and velocities in three dimensions. | gym.spaces.Dict({"position": gym.spaces.Discrete(3), "velocity": gym.spaces.Discrete(3)}) |
| MultiBinary            | n-dimensional binary space. The argument to MultiBinary() definesn.  | gym.spaces.MultiBinary(5) |
| MultiDiscrete          | Multi-dimensional discrete space. | gym.spaces.MultiDiscrete([-10,10], [0,1]) |
| Tuple                  | A product of simpler spaces. | gym.spaces.Tuple((gym.spaces.Discrete(2), spaces.Discrete(2))) |

### Summary

In this chapter, we explored the list of Gym environments available on your system, which you installed in the previous chapter, and then understood the naming conventions, or nomenclature, of the environments. We then revisited the agent-environment interaction (the RL loop) diagram and understood how the Gym environment provides the interfaces corresponding to each of the arrows we saw in the image. We then looked at a consolidated summary of the four values returned by the Gym environment's step() method in a tabulated, easy-to-understand format to reinforce your understanding of what they mean! We also explored in detail the various types of spaces used in the Gym for the observation and action spaces, and we used a script to print out what spaces are used by an environment to understand the Gym environment interfaces better. In our next chapter, we will consolidate all our learning so far to develop our first artificially intelligent agent! Excited?! Flip the page to the next chapter now!

