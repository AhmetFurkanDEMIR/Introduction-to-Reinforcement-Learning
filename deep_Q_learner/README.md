# Implementing an Intelligent Agent for Optimal Control using Deep Q- Learning

In this section, we will be able to train and test according to the arguments we have given with the python script file called deep_Q_learner.py.

## Running the code

The [deep_Q_learner.py](./deep_Q_learner.py) is the main script that takes care of training and testing depending on the script's arguments. The table below summarizes the arguments that the script supports and what they mean. Note that, most of the agent and environment related configuration parameters are in the [parameters.json](parameters.json) file and only the few parameters that are more useful when launching the training/testing scripts are made available through the command line arguments.

| Argument                 | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `--params-file`          | Path to the JSON parameters file. Default=`parameters.json`  |
| `--env`                  | Name/ID of the Atari environment available in OpenAI Gym. Default=`SeaquestNoFrameskip-v4` |
| `--gpu-id`               | ID of the GPU device to be used. Default=`0`                 |
| `--render`               | Render the environment to screen. Off by default             |
| `--test`                 | Run the script in test mode. Learning will be disabled. Off by default |
| `--record`               | Enable recording (Video & stats) of the agent's performance  |
| `--recording-output-dir` | Directory to store the recordings (video & stats). Default=`./trained_models/results` |




- ### Training

  Make sure the `rl_gym_book` conda environment with the necessary packages installed is activated. Assuming that you cloned the code as per the instructions to `~/HOIAWOG/`,  you can launch the Agent training script from the `~/HOIAWOG/ch6` directory using the following command:

  `python deep_Q_learner.py --env RiverraidNoFrameskip-v4 --gpu-id 0` 

   The above command will start training the agent for the Riverraid Atari game (`RiverraidNoFrameskip-v4`) . If a saved agent "brain" (trained model) is available for the chosen environment, the training script will upload that brain to the agent and continue training the agent to improve further.

  The training will run until `max_training_steps` is reached, which is specified in the [parameters.json](./parameters.json) file. There are several other parameters that can be configured using the [parameters.json](./parameters.json)  and it is recommended to adjust them based on the capabilities of the hardware you are running on. You can set `use_cuda` to `false` if you are running on a machine without a GPU.

  The log files are written to the directory pointed with the `summary_file_path_prefix` parameter (the default is `logs/DQL_*`). When the training script is running, you can monitor the learning progress of the agent visually using Tensorboard. From the `~/HOIAWOG/ch6` directory, you can launch Tensorboard with the following command: `tensorboard --log_dir=./logs/`. You can then visit the web URL printed on the console (the default one is: http://localhost:6006) to monitor the progress.


- ### Testing

  `python deep_Q_learner.py --env RiverraidNoFrameskip-v4 --test --render  --record`
  
  The above command should launch the Deep Q Learning Agent in test mode and render the environment states while also recording the performances. You can find the stats and the recording in the [trained_models/results](trained_models/results) directory after the script finishes running. Sample output for the agent trained on `RiverraidNoFrameskip-v4` for a few thousand episode is shown below:
  
    ![asd](trained_models/results/RiverraidNoFrameskip-v4.gif)
    
    
* You can train and test by entering other environments into the env variable of the data as an argument.
Sample : `python deep_Q_learner.py --env SpaceInvaders-v0 --gpu-id 0` , `python deep_Q_learner.py --env SpaceInvaders-v0 --test --render --record`
