 
# from stable_baselines3.common.cmd_util import make_atari_env
import gymnasium as gym
import os
import random as rd
import numpy as np
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
#from supply_chain.Batch_inventory_management import LeanSupplyEnv1, LeanSupplyEnv2, AgileSupplyEnv3
from supply_chain.inventory_management import  LeanSupplyEnv0, LeanSupplyEnv01, AgileSupplyEnv0
from gymnasium.envs.registration import register,make
import time
register(id='StoSupplyEnv-v1',
    entry_point='supply_chain.inventory_management:LeanSupplyEnv0'
)

register(id='StoSupplyEnv-v2',
    entry_point='supply_chain.inventory_management:LeanSupplyEnv01'
 )
register(id='StoSupplyEnv-v3',
    entry_point='supply_chain.inventory_management:AgileSupplyEnv0'
)


models_dir = f"lr=0.003_Scontinuous_models/{int(time.time())}/"
logdir = f"lr=0.003_Scontinuous_logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

###--------------------------------------------------------####################################
env_name3 = 'StoSupplyEnv-v3'
vec_env3 = make(env_name3)
# Parallel environments



# There already exists an environment generator
# that will make and wrap atari environments correctly
# env = make_atari_env('DemonAttackNoFrameskip-v4', num_env=8, seed=0)
model = PPO('MlpPolicy', vec_env3, learning_rate = 0.003, verbose=1,tensorboard_log=logdir)
model.learn(total_timesteps=1000000, tb_log_name=f'PPO_lr=0.003_first_phase_stoc1')
model.save(f"{models_dir}/PPO_model3")




obs,_ = vec_env3.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _,info, = vec_env3.step(action)
    # env.render()
# Close the processes
vec_env3.close()
del model
del vec_env3
# The number of environments must be identical when changing environments
env_name2 = 'StoSupplyEnv-v2'
env2 = make(env_name2)
# change env
model = PPO.load(f"{models_dir}/PPO_model3")
model.set_env(env2)
model.learn(total_timesteps=1000000, reset_num_timesteps=False, tb_log_name=f'PPO_lr=0.003_first_phase_stoc01')
model.save(f"{models_dir}/PPO_model2")
obs,_ = env2.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _,info = env2.step(action)
    # env.render()
env2.close()
del model
del env2

env_name1 = 'StoSupplyEnv-v1'
env1 = make(env_name1)
# change env
model = PPO.load(f"{models_dir}/PPO_model2")
model.set_env(env1)
model.learn(total_timesteps=1000000, reset_num_timesteps=False, tb_log_name=f'PPO_lr=0.003_first_phase_stoc0')
model.save(f"{models_dir}/PPO_model1")
obs,_ = env1.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _,info = env1.step(action)
    # env.render()
env1.close()
del model
del env1

###--------------------------------------------------------####################################
env_name23 = 'StoSupplyEnv-v3'
vec_env23 = make(env_name23)
# Parallel environments



model = PPO.load(f"{models_dir}/PPO_model1")
model.set_env(vec_env23)
# There already exists an environment generator
# that will make and wrap atari environments correctly
# env = make_atari_env('DemonAttackNoFrameskip-v4', num_env=8, seed=0)
# model = PPO('MlpPolicy', vec_env23, verbose=1,tensorboard_log=logdir)
model.learn(total_timesteps=1000000,reset_num_timesteps=False, tb_log_name=f'PPO_lr=0.003_second_phase_stoc1')
model.save(f"{models_dir}/PPO_model23")




obs,_ = vec_env23.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _,info, = vec_env23.step(action)
    # env.render()
# Close the processes
vec_env23.close()
del model
del vec_env23
# The number of environments must be identical when changing environments
env_name22 = 'StoSupplyEnv-v2'
env22 = make(env_name22)
# change env
model = PPO.load(f"{models_dir}/PPO_model23")
model.set_env(env22)
model.learn(total_timesteps=1000000, reset_num_timesteps=False, tb_log_name=f'PPO_lr=0.003_second_phase_stoc01')
model.save(f"{models_dir}/PPO_model22")
obs,_ = env22.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _,info = env22.step(action)
    # env.render()
env22.close()
del model
del env22

env_name21 = 'StoSupplyEnv-v1'
env21 = make(env_name21)
# change env
model = PPO.load(f"{models_dir}/PPO_model22")
model.set_env(env21)
model.learn(total_timesteps=1000000, reset_num_timesteps=False, tb_log_name=f'PPO_lr=0.003_second_phase_stoc0')
model.save(f"{models_dir}/PPO_model21")
obs,_ = env21.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _,info = env21.step(action)
    # env.render()
env21.close()
del model
del env21

###--------------------------------------------------------####################################
env_name33 = 'StoSupplyEnv-v3'
vec_env33 = make(env_name33)
# Parallel environments
model = PPO.load(f"{models_dir}/PPO_model21")
model.set_env(vec_env33)
# There already exists an environment generator
# that will make and wrap atari environments correctly
# env = make_atari_env('DemonAttackNoFrameskip-v4', num_env=8, seed=0)
# model = PPO.load(f"{models_dir}/PPO_model21")

model.learn(total_timesteps=1000000,reset_num_timesteps=False, tb_log_name=f'PPO_lr=0.003_third_phase_stoc1')
model.save(f"{models_dir}/PPO_model33")




obs,_ = vec_env33.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _,info, = vec_env33.step(action)
    # env.render()
# Close the processes
vec_env33.close()
del model
del vec_env33
# The number of environments must be identical when changing environments
env_name32 = 'StoSupplyEnv-v2'
env32 = make(env_name32)
# change env
model = PPO.load(f"{models_dir}/PPO_model33")
model.set_env(env32)
model.learn(total_timesteps=1000000, reset_num_timesteps=False, tb_log_name=f'PPO_lr=0.003_third_phase_stoc01')
model.save(f"{models_dir}/PPO_model32")
obs,_ = env32.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _,info = env32.step(action)
    # env.render()
env32.close()
del model
del env32

env_name31 = 'StoSupplyEnv-v1'
env31 = make(env_name31)
# change env
model = PPO.load(f"{models_dir}/PPO_model32")
model.set_env(env31)
model.learn(total_timesteps=1000000, reset_num_timesteps=False, tb_log_name=f'PPO_lr=0.003_third_phase_stoc0')
model.save(f"{models_dir}/PPO_model31")
obs,_ = env31.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _,info = env31.step(action)
    # env.render()
env31.close()
del model
del env31

