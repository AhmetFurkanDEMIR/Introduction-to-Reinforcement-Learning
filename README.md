# `Reinforcement Learning`

![a](/images_gif/as/4.gif)

Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.

Reinforcement learning differs from supervised learning in not needing labelled input/output pairs be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).

![index](/images_gif/mathworks-reinforcement-learning-fig1-543.jpg)

The environment is typically stated in the form of a Markov decision process (MDP), because many reinforcement learning algorithms for this context utilize dynamic programming techniques. The main difference between the classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the MDP and they target large MDPs where exact methods become infeasible.

![index](/images_gif/dn-7.9.png)

### Our learning map

* [What is an intelligent agent?](###-what-is-an-intelligent-agent?)
* [Learning environments](###-Learning-environments)
* [What is OpenAI Gym?](###-What-is-OpenAI-Gym?)
* [Understanding the features ofOpenAI Gym](###-Understanding-the-features-of-OpenAI-Gym)
    * Simple environment interface
    * Comparability and reproducibility]
    * Ability to monitor progress
* [What can you do with the OpenAIGym toolkit?](###-What-can-you-do-with-the-OpenAI-Gym-toolkit?)
* [Creating your first OpenAI Gym environment](###-Creating-your-first-OpenAI-Gym-environment)
* [Creating and visualizing a newGym environment](###-Creating-and-visualizing-a-new-Gym-environment)
* [Summary1](###-Summary1)
* [Practical reinforcement learning](###-Practical-reinforcement-learning)
    * Agent
    * Rewards
    * Environment
    * State
    * Value function
        * State-value function]
        * Action-value function
    * Policy
* [Summary2](###-Summary2)
* [Reference](###-Reference)

### What is an intelligent agent?

A major goal of artificial intelligence is to build intelligent agents.Perceiving their environment, understanding, reasoning, and learning to plan, and making decisions, and acting upon them are essential characteristics of intelligent agents. We will begin our first chapter by understanding what an intelligent agent is, from the basic definition of agents, to adding intelligence on top of that. An agent is an entity that acts based on the observation (perception) of its environment. Humans and robots are examples of agents with physical forms.

Software agents are computer programs that are capable of making decision sand taking actions through interaction with their environment. A software agent can be embodied in a physical form, such as a robot. Autonomous agents are entities that make decisions autonomously and take actions based on their understanding of and reasoning about their observations of their environment. An intelligent agent is an autonomous entity that can learn andimprove basedd on its interactions with its environment. An intelligent agent is capable off analyzing its own behavior and performance using its observations. In this book, we will develop intelligent agents to solve sequential decision-making problems that can be solved using a sequence of (independent)decisions/actions in a (loosely) Markovian environment, where feedbackin thee form of reward signals is available (through percepts), at least insome environmentall conditions.

### Learning environments

A learning environment is an integral component of a system where anintelligent agent can be trained to develop intelligent systems. The learningenvironment defines the problem or the task for the agent to complete.A problem or task in which the outcome depends on a sequence of decisionsmade or actions taken is a sequential decision-making problem. Here aresome of the varieties of learning environments:Fully observable versus partially observableDeterministic versus stochasticEpisodic versus sequentialStatic versus dynamicDiscrete versus continuousDiscrete state space versus continuous state spaceDiscrete action space versus continuous action spaceIn 

### What is OpenAI Gym?

OpenAI Gym is an open source toolkit that provides a diverse collection of tasks, called environments, with a common interface for developing and testing your intelligent agent algorithms. The toolkit introduces a standardApplication Programming Interface (API) for interfacing with environments designed for reinforcement learning. Each environment has aversion attached to it, which ensures meaningful comparisons andre producible results with the evolving algorithms and the environments themselves. The Gym toolkit, through its various environments, provides an episodic setting for reinforcement learning, where an agent's experience is broken down into a series of episodes. In each episode, the initial state of the agents randomly sampled from a distribution, and the interaction between the agent and the environment proceeds until the environment reaches a terminal state. Some of the basic environments available in the OpenAI Gym library are shown in the following screenshot:
(We will use some of these environments in our examples.)

![index](/images_gif/b0f8b8c7-aca7-456b-a63b-9e57c4e203fb.png)

### Understanding the features of OpenAI Gym

In this section, we will take a look at the key features that have made the OpenAI Gym toolkit very popular in the reinforcement learning community and led to it becoming widely adopted.

* Simple environment interface :
OpenAI Gym provides a simple and common Python interface to environments. Specifically, it takes an action as input and provides observation, reward, done and an optional info object, based on the action as the output at each step. If this does not make perfect sense to you yet, do not worry. We will go over the interface again in a more detailed manner to help you understand. This paragraph is just to give you an overview of the interface to make it clear how simple it is. This provides great flexibility for users as they can design and develop their agent algorithms based on any paradigm they like, and not be constrained to use any particular paradigm because of this simple and convenient interface.

* Comparability and reproducibility :
We intuitively feel that we should be able to compare the performance of an agent or an algorithm in a particular task to the performance of another agent or algorithm in the same task. For example, if an agent gets a score of 1,000on average in the Atari game of Space Invaders, we should be able to tell that this agent is performing worse than an agent that scores 5000 on average in the Space Invaders game in the same amount of training time. But what happens if the scoring system for the game is slightly changed? Or if the environment interface was modified to include additional information about the game states that will provide an advantage to the second agent? This would make the score-to-score comparison unfair, right?To handle such changes in the environment, OpenAI Gym uses strict versioning for environments. The toolkit guarantees that if there is any change to an environment, it will be accompanied by a different version number.Therefore, if the original version of the Atari Space Invaders game environment was named SpaceInvaders-v0 and there were some changes made to the environment to provide more information about the game states, then the environment's name would be changed to SpaceInvaders-v1. This simple versioning system makes sure we are always comparing performance measured on the exact same environment setup. This way, the results obtained are comparable and reproducible.

* Ability to monitor progress :
All the environments available as part of the Gym toolkit are equipped with a monitor. This monitor logs every time step of the simulation and every reset of the environment. What this means is that the environment automatically keeps track of how our agent is learning and adapting with every step. You can even configure the monitor to automatically record videos of the game while your agent is learning to play. How cool is that?

### What can you do with the OpenAI Gym toolkit?

The Gym toolkit provides a standardized way of defining the interface for environments developed for problems that can be solved using reinforcement learning. If you are familiar with or have heard of the ImageNet LargeScale Visual Recognition Challenge (ILSVRC), you may realize how muchof an impact a standard bench marking platform can have on accelerating research and development. For those of you who are not familiar with ILSVRC, here is a brief summary: it is a competition where the participating teams evaluate the supervised learning algorithms they have developed for the given dataset and compete to achieve higher accuracy with several visual recognition tasks. This common platform, coupled with the success of deep neural network-based algorithms popularized by AlexNet (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf),paved the way for the deep learning era we are in at the moment. In a similar way, the Gym toolkit provides a common platform to benchmark reinforcement learning algorithms and encourages researchers and engineers to develop algorithms that can achieve higher rewards for several challenging tasks. In short, the Gym toolkit is to reinforcement learning what ILSVRC is to supervised learning.

### Creating your first OpenAI Gym environment

Create a new environment for your reinforcement learning exercises, so you can avoid module conflicts and other errors.

* How will we create the new environment? and how do we get to it? : https://github.com/AhmetFurkanDEMIR/Create-a-workspace-for-machine-learning

### Creating and visualizing a new Gym environment

* Important !!! : Install the modules in requirements.txt before starting this step.

In just a minute or two, you have created an instance of an OpenAI Gym environment to get started! Let's open a new Python promptand import the gym module:

```python
import gym
```

Once the gym module is imported, we can use the gym.make method to create ournew environment like this:

```python
env = gym.make('CartPole-v0')
env.reset()
env.render()
```

This will bring up a window like this :

![a](/images_gif/1*LnQ5sRu-tJmlvRWmDsdSvw.gif)

Hooray!

### Summary1

Congrats on completing the first chapter! Hope you had fun creating your own environment. In this chapter, you learned what OpenAI Gym is all about, what features it provides, and what you can do with the toolkit.  You now have a very good idea about OpenAI Gym.

### Practical reinforcement learning

Now that you have an intuitive understanding of what AI really means and the various classes of algorithm that drive its development, we will now focus on the practical aspects of building a reinforcement learning machine. Here are the core concepts that you need to be aware of to develop reinforcement learning systems:

* Agent : 
In the reinforcement learning world, a machine is run or instructed by a(software) agent. The agent is the part of the machine that possesses intelligence and makes decisions on what to do next. You will come across the term"agent" several times as we dive deeper into reinforcement learning.Reinforcement learning is based on the reward hypothesis, which states that any goal can be described by the maximization of the expected cumulative reward. So, what is this reward exactly? That's what we'll discuss next.

* Rewards : 
A reward, denoted by , is usually a scalar quantity that is provided as feedback to the agent to drive its learning. The goal of the agent is to maximize the sum of the reward, and this signal indicates how well the agent is doing at time step . The following examples of reward signals for different tasks may help you get a more intuitive understanding:For the Atari games we discussed before, or any computer games in general, the reward signal can be +1 for every increase in score and -1for every decrease in score.For stock trading, the reward signal can be +1 for each dollar gained and-1 for each dollar lost by the agent.For driving a car in simulation, the reward signal can be +1 for every mile driven and -100 for every collision.Sometimes, the reward signal can be sparse. For example, for a game of chess or Go, the reward signal could be +1 if the agent wins the game and -1 if the agent loses the game. The reward is sparse because the agent receives the reward signal only after it completes one full game,not knowing how good each move it made was.

* Environment :
In the first chapter, we looked into the different environments provided by theOpenAI Gym toolkit. You might have been wondering why they were called environments instead of problems, or tasks, or something else. Now that you have progressed to this chapter, does it ring a bell in your head?The environment is the platform that represents the problem or task that weare interested in, and with which the agent interacts. The following diagram shows the general reinforcement learning paradigm at the highest level of abstraction: ![index](/images_gif/Screenshot_2020-08-20_17-03-16.png) At each time step, denoted by , the agent receives an observation from the environment and then executes an action , for which it receives a scalar reward back from the environment, along with the next observation ,and then this process repeats until a terminal state is reached. What is an observation and what is a state? Let's look into that next.

* State :
As the agent interacts with an environment, the process results in a sequence of observations (teta[i]), actions (A[i]), and rewards (R[i]), as described previously. At some time step "t", what the agent knows so far is the sequence of teta[i], A[i], and R[i] that it observed until time step . It intuitively makes sense to call this the history:
H[t] = {teta[1], A[1], R[1]}, {teta[2], A[2], R[2]}, ..., {teta[t], A[t], R[t]}
What happens next at time step "t+1" depends on the history. Formally, the information used to determine what happens next is called the state. Because it depends on the history up until that time step, it can be denoted as follows:
S[t] = f(H[t]), Here, f() denotes some function. There is one subtle piece of information that is important for you to understand before we proceed. Let's have another look at the general representation of a reinforcement learning system:

![index](/images_gif/aaaaaa.png)

Now, you will notice that the two main entities in the system, the agent and the environment, each has its own representation of the state. The environment state, sometimes denoted by , S[t]^e is the environment's own(private) representation, which the environment uses to pick the next
observation and reward. This state is not usually visible/available to the agent. Likewise, the agent has its own internal representation of the state, sometimes denoted by S[t]^a, which is the information used by the agent to baseits actions on. Because this representation is internal to the agent, it is up tothe agent to use any function to represent it. Typically, it is some function based on the history that the agent has observed so far. On a related note,a Markov stateis a representation of the state using all the useful informationfrom the history. By definition, using the Markov property, a state  is Markov or Markovian if, and only if, P[S[t+1] | S[t]] = P[S[t+1] | S[1], S[2], ..., S[t]] , which meansthat the future is independent of the past given the present. In other words, such a state is a sufficient statistic of the future. Once the state is known, the history can be thrown away. Usually, the environment state, S[t]^e, and thehistory, H[t], satisfy the Markov property. In some cases, the environment may make its internal environmental state directly visible to the agent. Such environments are called fully observable environments. In cases where the agent cannot directly observe the environment state, the agent must construct its own state representation from what it observes. Such environments are called partially observable environments. For example, an agent playing poker can only observe the public cards and not the cards the other players possess. Therefore, it is apartially observed environment. Similarly, an autonomous car with just acamera does not know its absolute location in its environment, which makesthe environment only partially observable. In the next sections, we will learn about some of the key components of anagent.

* Model : 
A model is an agent's representation of the environment. It is similar to the mental models we have about people and things around us. An agent uses its model of the environment to predict what will happen next. There are twokey pieces to it: 
F : The state transition model/probability, R : The reward model. 
The state transition model  is a probability distribution or a function thatpredicts the probability of ending up in a state  in the next time step  given the state  and the action  at time step. Mathematically, it is expressed as follows:
P[S][S']^a = P[S[t+1] = S' | S[t] = S, A[t] = A]
The agent uses the reward model "R" to predict the immediate next reward thatit would get if it were to take action "A" while in state "S" at time step "t". This expectation of the reward at the next time step "t+1" can be mathematicallyexpressed as follows:
R[S]^A = E[R[t+1] | S[t] = S, A[t] = A]

* Value function : 
A value function represents the agent's prediction of future rewards. There are two types of value function: state-value function and action-valuefunction.

    * State-value function : 
    A state-value function is a function that represents the agent's estimate of howgood it is to be in a "S" state at time step "t". It is denoted by V(S) and is usually just called the value function. It represents the agent's prediction of the future reward it would get if it were to end up in state  at time step "t". Mathematically, it can be represented as follows:
    V[pi](S) = E[R[t+1] + gama*R[t+2] + (gama^2)*R[t+3] + ... | S[t] = S]
    What this expression means is that the value of state  under policy  is the expected sum of the discounted future rewards, where  is the discount factorand is a real number in the range [0,1]. Practically, the discount factor is typically set to be in the range of [0.95,0.99]. The other new term is, whichis the policy of the agent.
    
    * Action-value function : 
    The action-value function is a function that represents the agent's estimate ofhow good it is to take action a[t] in state S[t]. It is denoted by Q(S[t], A[t]). It is relatedto the state-value function by the following equation:
    Q(S, A) = E[r + gama*V[S[t+1]]]

* Policy : 
The policy denoted by "pi" prescribes what action is to be taken given the state. It can be seen as a function that maps states to actions. There are two major types of policy: deterministic policies and stochastic policies. A deterministic policy prescribes one action for a given state, that is, there isonly one action, "a", given "s". Mathematically, it means pi(S) = A. A stochastic policy prescribes an action distribution given a state  at timestep , that is, there are multiple actions with a probability value for each action. Mathematically, it means pi(A|S) = P[A[t] = A | S[t] = S]. Agents following different policies may exhibit different behaviors in thesame environment.

### Summary2
So far we have studied reinforcement learning and the OpenAI gym module.
Now is the time to move on to concrete examples.

* **Draft_environments**

![index](/images_gif/as/ezgif.com-resize.gif)

* **Mountain_Car**

![index](/images_gif/as/1.gif) ![index](/images_gif/as/2.gif)

* **deep_Q_learner**

![index](/images_gif/as/3.gif)

Scroll through the figure, sequentially. Lets start then :)

### Reference

* https://www.amazon.com/Hands-Intelligent-Agents-OpenAI-reinforcement-ebook/dp/B07CSLCYDY
* https://www.manning.com/books/deep-reinforcement-learning-in-action
* https://gym.openai.com/



