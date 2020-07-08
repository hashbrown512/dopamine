import numpy as np
import os
from os import path
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import run_experiment, gym_lib
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
import pandas as pd

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def gen_config(update_horizon, min_replay_history, update_period, target_update_period, learning_rate, num_iterations, training_steps,
               evaluation_steps):
    config = """
    # chosen achieve reasonable performance.
    import dopamine.agents.rainbow.rainbow_agent
    import dopamine.discrete_domains.gym_lib
    import dopamine.discrete_domains.run_experiment
    import dopamine.replay_memory.prioritized_replay_buffer
    import gin.tf.external_configurables

    RainbowAgent.observation_shape = %gym_lib.ABR_OBSERVATION_SHAPE
    RainbowAgent.observation_dtype = %gym_lib.ABR_OBSERVATION_DTYPE
    RainbowAgent.stack_size = %gym_lib.ABR_STACK_SIZE
    RainbowAgent.network = @gym_lib.ABRRainbowNetwork
    RainbowAgent.num_atoms = 51
    RainbowAgent.vmax = 10.
    RainbowAgent.gamma = 0.99
    RainbowAgent.update_horizon = {}
    RainbowAgent.min_replay_history = {}
    RainbowAgent.update_period = {}
    RainbowAgent.target_update_period = {}
    RainbowAgent.epsilon_fn = @dqn_agent.identity_epsilon
    RainbowAgent.replay_scheme = 'prioritized'
    # RainbowAgent.tf_device = '/gpu:0'  # use '/cpu:*' for non-GPU version
    RainbowAgent.tf_device = '/cpu:*'
    RainbowAgent.optimizer = @tf.train.AdamOptimizer()

    tf.train.AdamOptimizer.learning_rate = {}
    tf.train.AdamOptimizer.epsilon = 0.0003125

    # TODO: figure out how to make the environment with gym make
    create_gym_environment.environment_name = 'ABR'
    create_gym_environment.version = 'v1'
    create_agent.agent_name = 'rainbow'
    Runner.create_environment_fn = @gym_lib.create_gym_environment
    Runner.num_iterations = {}
    Runner.training_steps = {}
    Runner.evaluation_steps = {}
    Runner.max_steps_per_episode = 500

    WrappedPrioritizedReplayBuffer.replay_capacity = 50000
    WrappedPrioritizedReplayBuffer.batch_size = 128
    """.format(update_horizon, min_replay_history, update_period, target_update_period, learning_rate, num_iterations, training_steps,
               evaluation_steps)
    return config


# min_replay_histories = [500, 5000, 20000]
# update_periods = [1, 2, 4]
# target_update_periods = [500, 1000]
# num_training_steps = 2000
# evaluation_steps = 1000
# num_iterations = 2

# DIR = "test/"
# create_folder_if_not_exists(DIR)

# min_replay_histories = [500]
# update_periods = [1, 2]
# target_update_periods = [500]
# num_training_steps = 2000
# evaluation_steps = 1000
# num_iterations = 2

grid_dir = 'grid_5M/'
grid_file = grid_dir + 'results.csv'
create_folder_if_not_exists(grid_dir)

# Run 1
# update_horizons = [1,3]
# min_replay_histories = [20000]
# update_periods = [1, 4]
# target_update_periods = [50]
# learning_rate = 0.09
# num_training_steps = 5000000
# evaluation_steps = 500000
# num_iterations = 1

# Run 2
update_horizons = [1,3]
min_replay_histories = [20000]
update_periods = [1, 4]
target_update_periods = [100]
learning_rate = 0.09
num_training_steps = 5000000
evaluation_steps = 500000
num_iterations = 1

# Run 3
# update_horizons = [1,3]
# min_replay_histories = [20000]
# update_periods = [1,4]
# target_update_periods = [4000]
# learning_rate = 0.09
# num_training_steps = 5000000
# evaluation_steps = 500000
# num_iterations = 1

# # Run 4
# update_horizons = [1,3]
# min_replay_histories = [20000]
# update_periods = [1, 4]
# target_update_periods = [8000]
# learning_rate = 0.09
# num_training_steps = 5000000
# evaluation_steps = 500000
# num_iterations = 1

# Run 5
# update_horizons = [1,3]
# min_replay_histories = [20000]
# update_periods = [1, 4]
# target_update_periods = [16000]
# learning_rate = 0.09
# num_training_steps = 5000000
# evaluation_steps = 500000
# num_iterations = 1



# update_horizons = [1]
# min_replay_histories = [500, 5000]
# update_periods = [1]
# target_update_periods = [50, 100]
# learning_rate = 0.09
# num_training_steps = 1000
# evaluation_steps = 1000
# num_iterations = 2

for uh in update_horizons:
    for mrh in min_replay_histories:
        for up in update_periods:
            for tup in target_update_periods:
                kwargs = {"min_replay_history": mrh, "update_period": up,
                          "training_steps": num_training_steps,
                          "evaluation_steps": evaluation_steps, "target_update_period": tup,
                          "update_horizon": uh,
                          "learning_rate": learning_rate,
                          "num_iterations": num_iterations}
                config = gen_config(**kwargs)
                gin.parse_config(config, skip_unknown=False)
                LOG_PATH = grid_dir + str(mrh) + "_" + str(up) + '_' + str(tup) + '_' + str(uh)


                def create_agent(sess, environment, summary_writer=None):
                    return rainbow_agent.RainbowAgent(sess, num_actions=environment.action_space.n)

                # TODO: rerunning may require changing the create agent function to load the agent
                rainbow_runner = run_experiment.TrainRunner(LOG_PATH, create_agent,
                                                            create_environment_fn=gym_lib.create_gym_environment)
                rainbow_runner.run_experiment()
                data = colab_utils.read_experiment(
                    LOG_PATH, verbose=True, summary_keys=['train_episode_returns', 'train_average_return'])
                final_eval = data.loc[data['iteration'] == num_iterations - 1]['train_episode_returns'][data['iteration'] == num_iterations - 1]
                kwargs["average_episode_reward"] = final_eval

                if path.exists(grid_file):
                    df = pd.read_csv(grid_file, index_col=0)
                    df = df.append(kwargs, ignore_index=True)
                    print(df)
                else:
                    df = pd.DataFrame(columns = ["average_episode_reward", "min_replay_history",  "update_period",
                                      "training_steps",
                                      "evaluation_steps", "target_update_period",
                                      "num_iterations"])
                    df = df.append(kwargs, ignore_index=True)
                print(df)
                df.to_csv(grid_file)