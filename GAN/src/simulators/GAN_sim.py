import numpy as np
import pickle
import os, sys
import pathlib
import jax
import pandas as pd
import optax
import jax.numpy as jnp
import sklearn


from jax import grad, jit, vmap
import time

params_dir = '/Users/mxd6118/Desktop/GAN/src/simulators/'
#sim_dir = os.path.dirname(dir_dir)
sys.path.insert(0,'/Users/mxd6118/Desktop/DiffSim')


from simulators.WF_sim import simulate_waveforms, init_params
from jax import random
from jax.example_libraries import optimizers as jax_opt

def get_data(data, name_data):
        
        with open(f'{name_data}.pickle','wb') as f:
            pickle.dump(data, f)
            f.close()

class Producer(object):

    def __init__(self, config):
    
        self.config = config
        
        self.key = random.PRNGKey(int(time.time()))
        self.key, self.subkey = random.split(self.key)
        
        number_of_events = self.config.run.minibatch_size  #int(input('How many events to be produced?'))
        
        params_path = os.path.join(params_dir ,"trained_params.pickle")
                
         #self.load_state(params_path)
        
        self.dataloader = self.build_dataloader(number_of_events)
        
        self.data_set = next(self.dataloader.iterate())
        
        choice = input('Do u want to load previus (supervised trained) wieghts?')
        
        if choice == 'yes':
            self.sim_params = self.load_state(params_path)
        else:
            self.sim_params = init_params(self.subkey,self.data_set)
        


    def build_dataloader(self,number_of_events):
        from src.utils.dataloaders.krypton_DATES_CUSTOM_DROPOUT import krypton
        # Load the sipm database:
        sipm_db = pd.read_pickle("/Users/mxd6118/Desktop/DiffSim/database/new_sipm.pkl")
    
        dl = krypton(
            batch_size  = number_of_events,
            db          = sipm_db,
            path        = "/Users/mxd6118/Desktop/DiffSim/kdst",
            run         = 8530,
            shuffle = False,
            drop = 0,
            z_slice = 0,
            )
            
        return dl
        

    def arrays(self,monitor_data,params):
        
        self.key, subkey = jax.random.split(self.key)
        
        # First, run the monitor data through the simulator:
        noise = batch_gen_noise_constant(monitor_data['S2Si'],subkey)
        simulated_pmts, simulated_sipms = simulate_waveforms(monitor_data, params, noise, subkey)
        
        return simulated_pmts, simulated_sipms, params
        
    def load_state(self,file):
        with open(file,"rb") as f:
            params = pickle.load(f)
        return params

    def build_random_batch(self, number_of_events):
    
        batch =[]
        for i in range(0,number_of_events):
            one = np.hstack((np.random.uniform(low = -150, high = 150),
                             np.random.uniform(low = -150, high = 150),
                             np.random.uniform(low = 20,   high = 500),0.0415575))

            two = np.vstack((one,np.zeros(4)))
    
            batch.append(two)

        return np.array(batch)


