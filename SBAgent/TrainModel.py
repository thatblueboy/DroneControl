import sys
sys.path.append("..")

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from envs.utils.EnvBuilder import EnvBuilder


parser = argparse.ArgumentParser()
parser.add_argument("configFileName", help="Name of the environment config file.", type=str)
parser.add_argument("outputModelName", help="(base|finetuned + )Name to save the model with.", type=str)
parser.add_argument("-s", "--steps", default=2_000_000, help="Number of timesteps to train for", type=int)
parser.add_argument("-d", "--dynamic", default=False, help="Use Dynamic Obstacles", type=bool)
args = parser.parse_args()

configFileName = args.configFileName
modelName = args.outputModelName
n_steps = args.steps
dynamic = args.dynamic

assert modelName.startswith('base') or modelName.startswith('finetuned'), "Model name must include base/finetuned"

if dynamic:
    env_file_path = os.path.join('..', 'configs', 'specifics', 'dynamicObs', configFileName)
else:
    env_file_path = os.path.join('..', 'configs', 'specifics', configFileName)
env = EnvBuilder.buildEnvFromConfig(env_file_path, gui=False)

eval_callback = EvalCallback(env, best_model_save_path=os.path.join('models', modelName), 
                                log_path=os.path.join('sbEvalLogs', modelName), 
                                eval_freq=100_000, deterministic=True, render=False)

agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join('logs', modelName))
agent.learn(n_steps, callback=eval_callback, tb_log_name="train_logs")
agent.save(os.path.join('models', modelName, 'final_model'))

