import sys, os
import pathlib
import time
import numpy as np

import sklearn
import pickle
from jax.example_libraries import stax
import jax

# For database reads:
import pandas as pd
from jax import random


# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

hydra.output_subdir = None

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from jax import random
import jax.numpy as numpy
import jax.tree_util as tree_util

# from jax.config import config
# config.update('jax_disable_jit', True)


from tensorboardX import SummaryWriter


# Add the local folder to the import path:
src_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(src_dir) + "/src/"

print(src_dir)
sys.path.insert(0,src_dir)

from simulators.WF_sim import simulate_waveforms

from config import Config
#from config import MPI_AVAILABLE, NAME

# if MPI_AVAILABLE:
#     import horovod.tensorflow as hvd
#from utils.checkpoint import init_checkpointer


class GAN(object):
    
    def __init__(self,config):
        
        self.config = config
        
        prod = self.build_producer(self.config)

        self.batch = prod.data_set
        sim_params = prod.sim_params
        
        
        #print(self.batch['Labels'].shape, flush = True)
        
        self.key = random.PRNGKey(int(time.time()))
        
        self.key, self.subkey = random.split(self.key)
        
        _parameters, dis_apply = self.init_params(sim_params,self.subkey)
        
        #print(self.out_size, flush = True)
        
        self.trainer = self.build_trainer(self.batch, dis_apply, simulate_waveforms, _parameters,self.subkey)
        
    def build_producer(self, config):
        from simulators.GAN_sim import Producer
        
        return Producer(self.config)
        
    def init_params(self, sim_params,subkey):
    
        dis_init, dis_apply = stax.serial(
            stax.Flatten,
            stax.Dense(128),stax.Sigmoid,
            stax.Dense(16), stax.Sigmoid,
            stax.Dense(2),stax.Softmax
        )
        
        dis_out_size, dis_network_params = dis_init(subkey,(1,47,47))
        
        parameters = {
        'D_parameters': dis_network_params,
        'S_parameters': sim_params,
        }
        
        return parameters, dis_apply
    
    def build_trainer(self, batch, dis_fn, sim_fn, params,subkey):
        from trainers.Trainer import GAN_trainer
        # Shouldn't reach this portion unless training.
        trainer = GAN_trainer(batch, dis_fn, sim_fn, params,subkey)
        
        return trainer
        
    def get_data(self, data, name_data):

        with open(f'{name_data}.pickle','wb') as f:
            pickle.dump(data, f)
            f.close()
   
    def train(self):
    
        print('Beginning Training', flush = True)
        c = 0
        self.key = jax.random.PRNGKey(int(time.time()))
        self.key, subkey = jax.random.split(self.key)

        while c <= self.config.run.iterations:

            metrics = {}
            start = time.time()

            metrics["io_time"] = time.time() - start

            train_metrics, opt_state = self.trainer.train_iteration(self.batch, c)
            
            #print('train metrics',train_metrics.keys(),flush = True)

            # print(model_parameters.keys())
            # print(model_parameters['diffusion'])
            
            metrics.update(train_metrics)

            metrics['time'] = time.time() - start
            
            #metrics['accuracy'] = acc

            #self.summary(metrics, self.global_step)

            if c % 1 == 0:
                print(f"step = {c}, loss = {metrics['loss/loss']:.3f}, time = {metrics['time']:.3f}",flush = True)

            c += 1
                      
        self.get_data(self.trainer.get_params(opt_state),'D_S_parameters')

@hydra.main(config_path="../src/config", config_name="config")
def main(cfg : Config) -> None:

    # Prepare directories:
    #work_dir = pathlib.Path(cfg.save_path)
    #work_dir.mkdir(parents=True, exist_ok=True)
    #log_dir = pathlib.Path(cfg.save_path + "/log/")
    #log_dir.mkdir(parents=True, exist_ok=True)

    # cd in to the job directory since we disabled that with hydra:
    # os.chdir(cfg.hydra.run.dir)
    e = GAN(cfg)
    e.train()
    #signal.signal(signal.SIGINT, e.interupt_handler)


if __name__ == "__main__":
    import sys

    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra.run.dir=.', 'hydra/job_logging=disabled', 'hydra/hydra_logging=disabled']
        print(sys.argv)
    main()
