# -*- coding: utf-8 -*-


import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import cloudpickle
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import numpy as np
from supply_chain.inventory_management import  LeanSupplyEnv0, LeanSupplyEnv01, AgileSupplyEnv0
from supply_chain.Batch_inventory_management import LeanSupplyEnv1, LeanSupplyEnv2, AgileSupplyEnv3
from gymnasium.envs.registration import register,make
import time

register(id='StoSupplyEnv-v1',
	entry_point='supply_chain.inventory_management:LeanSupplyEnv0'
)


register(    
    id="StoSupplyEnv-v2", 
    entry_point="supply_chain.inventory_management:LeanSupplyEnv01"
)


register(
   
    id="StoSupplyEnv-v3",
    
    entry_point="supply_chain.inventory_management:AgileSupplyEnv0"
    
)

register(id='BatSupplyEnv-v1',
	entry_point='supply_chain.Batch_inventory_management:LeanSupplyEnv1'
)


register(id='BatSupplyEnv-v2',
 	entry_point='supply_chain.Batch_inventory_management:LeanSupplyEnv2'
 )
 
register(id='BatSupplyEnv-v3',
	entry_point='supply_chain.Batch_inventory_management:AgileSupplyEnv3'
)






def create_directories(models_dir, logdir):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

def main():
    config = {
        "PPO": [ "BatSupplyEnv-v1"]
    }


   
    models_dir = f"./12modelPolicy/{int(time.time())}/"
    logdir = f"./12logPolicy/{int(time.time())}/"

    create_directories(models_dir, logdir)

    for algorithm_name, environments in config.items():
        for environment_name in environments:
            env = make(environment_name)
            log_name = f"{environment_name}_{algorithm_name}"
            
            if algorithm_name == "PPO":
                TIMESTEPS = 100
                total_iterations = 100
                model = PPO("MlpPolicy", env, learning_rate=0.003, verbose=1, tensorboard_log=logdir)
            elif algorithm_name == "RecurrentPPO":
                TIMESTEPS = 230
                total_iterations = 230
                model = RecurrentPPO("MlpLstmPolicy", env, learning_rate=0.003, verbose=1, tensorboard_log=logdir)
            else:
                print(f"Unknown algorithm: {algorithm_name}")
                continue

            iters = 0
            for _ in range(total_iterations):
                iters += 1
                print(f"Training {algorithm_name} on {environment_name}, log_name={log_name}")
                model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name)
                model.save(f"{models_dir}/{iters}")

if __name__ == "__main__":
    main()



