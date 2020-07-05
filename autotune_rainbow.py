import numpy as np
import os
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import run_experiment, gym_lib
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
import optuna

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

def run(mrh, up, tup, uh, lr):
    DIR = "auto_2/"
    create_folder_if_not_exists(DIR)

    num_training_steps = 100000
    evaluation_steps = 200000
    num_iterations = 10
    # num_training_steps = 1000
    # evaluation_steps = 1000
    # num_iterations = 3

    kwargs = {"min_replay_history": mrh, "update_period": up,
              "training_steps": num_training_steps,
              "evaluation_steps": evaluation_steps, "target_update_period": tup,
              "update_horizon": uh,
              "learning_rate": lr,
              "num_iterations": num_iterations}
    config = gen_config(**kwargs)
    gin.parse_config(config, skip_unknown=False)
    LOG_PATH = DIR + str(mrh) + "_" + str(up) + '_' + str(tup) + str(uh)

    def create_agent(sess, environment, summary_writer=None):
        return rainbow_agent.RainbowAgent(sess, num_actions=environment.action_space.n)

    rainbow_runner = run_experiment.TrainRunner(LOG_PATH, create_agent,
                                                create_environment_fn=gym_lib.create_gym_environment)
    rainbow_runner.run_experiment()
    data = colab_utils.read_experiment(
        LOG_PATH, verbose=True, summary_keys=['train_episode_returns', 'train_average_return'])
    final_eval = data.loc[data['iteration'] == num_iterations - 1]['train_episode_returns'][data['iteration'] == num_iterations - 1]
    return final_eval

def objective(trial):
    min_replay_histories = trial.suggest_int('min_replay_histories', 100, 40000)
    update_periods = trial.suggest_int('update_periods', 1, 8)
    target_update_periods = trial.suggest_int('target_update_periods', 50, 20000)
    update_horizon = trial.suggest_int('update_horizon', 1, 8)
    learning_rate = trial.suggest_uniform('learning_rate', 0.01, 0.15)

    # Must minimize, recording negative of the score to maximize positive score
    return run(min_replay_histories, update_periods, target_update_periods, update_horizon, learning_rate)

if __name__ == '__main__':
    # optuna create-study --study-name "autotune_2" --direction "maximize" --storage "sqlite:///autotune_2.db"
    # hyper_opt_autotuner()
    study = optuna.load_study(study_name='autotune_2', storage='sqlite:///autotune_2.db')
    # study.optimize(hyper_opt_autotuner, n_trials=250)
    study.optimize(objective, n_trials=12)

# study = optuna.load_study(study_name='autotune_1', storage='sqlite:///autotune_1.db')
# df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
# csv = df.to_csv(index=False)
# print(csv)