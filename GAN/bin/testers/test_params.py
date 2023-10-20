import numpy as np
import pickle
import os, sys
import pathlib
import jax
import pandas as pd
import optax
import jax.numpy as jnp
import sklearn
import matplotlib.pyplot as plt
import glob

from jax.example_libraries import stax
from jax import grad, jit, vmap
import time

#src_dir = os.path.dirname(dir_dir) + "/src/"
sys.path.insert(0,'/Users/mxd6118/Desktop/GAN/src/simulators/')
from WF_sim import simulate_waveforms, init_params

sys.path.insert(0,'/Users/mxd6118/Desktop/DiffSim')
from Plots import *

from jax import random

def load_state(file):
    with open(file,"rb") as f:
        params = pickle.load(f)
    return params


@jit
def gen_noise_constant(sipm_waveforms,subkey):

    noise_constant = random.uniform(subkey,shape = sipm_waveforms.shape,minval = -0.5, maxval = 1)
    #noise_constant = random.poisson(subkey,shape = sipm_waveforms.shape,lam = 0.05)
    #noise_constant = random.beta(subkey,shape = sipm_waveforms.shape,a = 2, b = 5)

    return noise_constant

xy_gen_noise_constant = jit(vmap(gen_noise_constant,in_axes=[0,None]))
event_gen_noise_constant = jit(vmap(xy_gen_noise_constant,in_axes=[0,None]))
batch_gen_noise_constant = jit(vmap(event_gen_noise_constant,in_axes=[0,None]))


class test_params():

    def __init__(self):
        print('Many parameters found:')
        [print(file_path) for file_path in glob.glob('/Users/mxd6118/Desktop/GAN/bin/*.pickle')]
        print('/Users/mxd6118/Desktop/GAN/src/simulators/trained_params.pickle') #to include the supervised params
        path = input('Which parameters do you wish to use?')
        

        params = load_state(path)
        
        #print(params,flush = True)
        n_events = input('How many events have to be simulated?')
        
        dataloader = self.build_dataloader(int(n_events))
        
        batch_real = next(dataloader.iterate())
        
        sim_pmt, sim_sipm = self.simulate(batch_real,params)
        
        data = {"real_batch" : batch_real,
                "fake_sipms" : sim_sipm}
                
        self.get_data(data,f"data_tested_{path.split('/')[-1].split('.')[0]}")

    def build_dataloader(self, number_of_events):

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

    def simulate(self,monitor_data,params):
    
        key = random.PRNGKey(int(time.time()))
        
        key, subkey = jax.random.split(key)
        
        noise = batch_gen_noise_constant(monitor_data['S2Si'],subkey)
       
        # First, run the monitor data through the simulator:
        simulated_pmts, simulated_sipms = simulate_waveforms(monitor_data['energy_deposits'], params, noise, subkey)

        return simulated_pmts, simulated_sipms
    
    def get_data(self, data, name_data):

        with open(f'testers/{name_data}.pickle','wb') as f:
            pickle.dump(data, f)
            f.close()
    
test_params()
