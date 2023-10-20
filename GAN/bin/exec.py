import sys, os
import pathlib
import time
import numpy as np
import inquirer

import sklearn
import pickle
from jax.example_libraries import stax
import jax

# For database reads:
import pandas as pd
from jax import random
import matplotlib.pyplot as plt

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
        
        self._parameters, self.dis_apply = self.get_params(sim_params,self.subkey)
        
        self.save_weights = self.config.run.save_weigths
        
        #print(_parameters, flush = True)
        
    def build_producer(self, config):
        from simulators.GAN_sim import Producer
        
        return Producer(self.config)
        
    def get_params(self, sim_params,subkey):
    
        dis_init, dis_apply = stax.serial(
            stax.Flatten,
            stax.Dense(256),stax.Sigmoid,
            stax.Dense(64), stax.Sigmoid,
            stax.Dense(16), stax.Sigmoid,
            stax.Dense(2),stax.Softmax
        )
        
        dis_out_size, dis_network_params = dis_init(subkey,(1,47,47,550))
        
        D_params = {'D_network_params': dis_network_params}
        
        parameters = sim_params | D_params
        
        return parameters, dis_apply
    
    def build_trainer(self, batch, dis_fn, sim_fn, params,subkey):
        
        from trainers.Trainer import GAN_trainer
        
        trainer = GAN_trainer(batch, dis_fn, sim_fn, params,subkey)
        
        return trainer
        
    def build_asymmetric_trainer(self, batch, dis_fn, sim_fn, params,subkey):
        from trainers.trainer_sim import GAN_trainer_sim
        from trainers.trainer_dis import GAN_trainer_dis
        
        
        trainer_sim = GAN_trainer_sim(batch, dis_fn, sim_fn, params,subkey)
        trainer_dis = GAN_trainer_dis(batch, dis_fn, sim_fn, params,subkey)
        
        return trainer_sim, trainer_dis
        
    def get_data(self, data, name_data):

        with open(f'{name_data}.pickle','wb') as f:
            pickle.dump(data, f)
            f.close()
            
    def plots(self,array,name):
        plt.clf()
        plt.plot(range(0,len(array)),array)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(f'loss_epochs_{name}.png')
        #plt.show()
        
    def plots_both(self,sim,dis):

        plt.plot(range(0,len(sim)),sim,label= 'Sim Loss')
        plt.plot(range(0,len(dis)),dis,label= 'Dis Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'loss_epochs_both.png')
        #plt.show()
   
    def train(self):
    
        print('Training phase entered, brace yourself')
        self.trainer = self.build_trainer(self.batch, self.dis_apply, simulate_waveforms, self._parameters,self.subkey)
    
        print('Beginning Training', flush = True)
        c = 0
        self.key = jax.random.PRNGKey(int(time.time()))
        self.key, subkey = jax.random.split(self.key)
        loss =[]
        

        while c <= self.config.run.iterations:

            metrics = {}
            start = time.time()
            metrics["io_time"] = time.time() - start
            train_metrics, opt_state = self.trainer.train_iteration(self.batch, c)
            metrics.update(train_metrics)
            loss.append(metrics['loss/loss'])
            metrics['time'] = time.time() - start
                        
            
            print(f"step = {c}, loss = {metrics['loss/loss']:.3f}, time = {metrics['time']:.3f}",flush = True)
            _params = self.trainer.get_params(opt_state)
            
            for key in _params.keys():
                if 'network' not in key :
                    if 'range' not in key:
                        print(f'Parameter {key}: {_params[key]}',flush = True)
            c += 1
        if self.save_weights:
            self.get_data(self.trainer.get_params(opt_state),f'D_S_parameters')
        self.plots(loss,'total')
        
    def train_asymmetric(self):
    
        print('Asymmetric training entered, really brace yourself')
        
        self.trainer_sim, self.trainer_dis = self.build_asymmetric_trainer(self.batch, self.dis_apply, simulate_waveforms, self._parameters,self.subkey)
        
        _parameters_sim  = {x:self._parameters[x] for x in self._parameters.keys() if x != "D_network_params"}

        c = 0
        self.key = jax.random.PRNGKey(int(time.time()))
        self.key, subkey = jax.random.split(self.key)
        loss_sim =[]
        loss_dis =[]
        
        
        while c <= self.config.run.iterations:

            metrics = {}
            start = time.time()
            metrics["io_time"] = time.time() - start
            
            
            train_metrics, opt_state_sim = self.trainer_sim.train_iteration(self.batch,c)
            _params = self.trainer_sim.get_params(opt_state_sim)
                
           
            #train_metrics, opt_state_dis = self.trainer_dis.train_iteration(self.batch,c)
            #_params = self.trainer_dis.get_params(opt_state_dis)
               
            metrics.update(train_metrics)
            
            if metrics['loss'][1] == 'sim':
                loss_sim.append(metrics['loss'][0])
            else:
                loss_dis.append(metrics['loss'][0])
            
            metrics['time'] = time.time() - start
                        
            print(f"step = {c}, loss_{metrics['loss'][1]} = {metrics['loss'][0]:.5f}, time = {metrics['time']:.3f}",flush = True)
            
            for key in _params.keys():
                if 'network' not in key :
                    if 'range' not in key:
                        print(f'Parameter {key}: {_params[key]}',flush = True)
            
            if self.save_weights:
                self.get_data(self.trainer.get_params(opt_state),f'D_S_parameters_{c}_asymmetric')
            c += 1
            
        #self.get_data(self.trainer.get_params(opt_state),f'D_S_parameters_asymmetric')
        self.plots(loss_sim,'sim')
        self.plots(loss_dis,'dis')
        
        self.plots_both(loss_sim,loss_dis)
        

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
    questions = [inquirer.List('mode',
                message="Which training mode",
                choices=['Coupled', 'Uncoupled'],),]
    answers = inquirer.prompt(questions)
    print(answers,flush=True)
    if answers['mode'] == "Coupled":
        print('Coupled training selected')
        e.train()
    else:
        e.train_asymmetric()
    #signal.signal(signal.SIGINT, e.interupt_handler)


if __name__ == "__main__":
    import sys

    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra.run.dir=.', 'hydra/job_logging=disabled', 'hydra/hydra_logging=disabled']
        print(sys.argv)
    main()
