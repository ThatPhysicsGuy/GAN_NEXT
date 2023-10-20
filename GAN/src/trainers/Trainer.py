from jax import grad, jit, vmap
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers as jax_opt
from jax import random
import time
import numpy as np
import sklearn
from jax import device_put


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8 # Small value to avoid division by zero
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)  # Clip values to prevent NaNs
    loss = -(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))
    return jnp.mean(loss)
    
def gen_loss(y_true,y_pred):
    epsilon = 1e-8 # Small value to avoid division by zero
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)  # Clip values to prevent NaNs
    loss = -(y_true) * jnp.log(1 - y_pred)
    return jnp.mean(loss)
    
## ADD ACCURACY FUNCTION ##
        
@jit
def gen_noise_constant(sipm_waveforms,subkey):

    noise_constant = random.uniform(subkey,shape = sipm_waveforms.shape,minval = -0.5, maxval = 1)
    #noise_constant = random.poisson(subkey,shape = sipm_waveforms.shape,lam = 0.05)
    #noise_constant = random.beta(subkey,shape = sipm_waveforms.shape,a = 2, b = 5)

    return noise_constant
xy_gen_noise_constant = jit(vmap(gen_noise_constant,in_axes=[0,None]))
event_gen_noise_constant = jit(vmap(xy_gen_noise_constant,in_axes=[0,None]))
batch_gen_noise_constant = jit(vmap(event_gen_noise_constant,in_axes=[0,None]))


class GAN_trainer():
    
    def __init__(self,batch, fn_dis, fn_sim, parameters, subkey):
        
        self.dis_apply = fn_dis
        self.sim_wf = fn_sim
        
        print('Created Trainer',flush = True)
        
        @jit
        def forward_pass(batch, parameters, key):
        
            simulated_pmts, simulated_sipms = self.sim_wf(batch['energy_deposits'], parameters, self.noise, key)
            
            GAN_batch = self.Chanteclair(batch['S2Si'],simulated_sipms)
            
            print(GAN_batch['Labels'].shape,flush = True)
           
            fake_labels = self.dis_apply(parameters['D_network_params'],GAN_batch['Train'])
        
            loss_dis = binary_cross_entropy(GAN_batch['Labels'],fake_labels)
            
            loss_sim = gen_loss(GAN_batch['Labels'],fake_labels)
            
            loss = loss_dis + loss_sim
            
            return loss
            
        self.noise = batch_gen_noise_constant(batch['S2Si'],subkey)
        
        self.gradient_fn = jax.value_and_grad(forward_pass, argnums=1,has_aux=False)
        
        opt_init, opt_update, get_params = jax_opt.adamax(1e-2)
            
        self.opt_state = opt_init(parameters)

        self.opt_update = opt_update
        self.get_params = get_params
    
    ## FIX THIS FOR BOTH LOADED AND NEW PARAMS ###
    
    def parameters(self):
        
        p = self.get_params(self.opt_state)
        # Deliberately slice things up here:

        parameters = {}
        parameters["D_network_params"] = p["D_network_params"]
        parameters["diffusion/x"] = p["diffusion"][0]
        parameters["diffusion/y"] = p["diffusion"][1]
        parameters["diffusion/z"] = p["diffusion"][2]
        parameters["lifetime"] = p["lifetime"]
        parameters["el_spread"] = p["el_spread"]
        parameters["pmt_dynamic_range"] = p["pmt_dynamic_range"]
        parameters["sipm_dynamic_range"] = p["sipm_dynamic_range"]
        
        return parameters

        
    
    def train_iteration(self, batch, c):
        
        metrics = {}

        parameters = self.get_params(self.opt_state)
        
        #print(self.parameters,flush = True)
    
        self.key = random.PRNGKey(int(time.time()))

        self.key, subkey = jax.random.split(self.key)

        loss, gradients = self.gradient_fn(batch, parameters, subkey)
                
        self.opt_state = self.opt_update(c, gradients, self.opt_state)

        # # Combine losses
        
        metrics['loss/loss'] = loss

        metrics.update(self.parameters())

        return metrics, self.opt_state
        
    
    def Chanteclair(self,real,fake):
    
        train_batch_filtered = {}
        train_batch_filtered['S2Si'] = []
        train_batch_filtered['SIPM_FAKE'] =[]
        
        for n in range(real.shape[0]):
    
            train_batch_filtered['SIPM_FAKE'].append(real[n])
                                                            
            train_batch_filtered['S2Si'].append(fake[n])
            

        l = len(train_batch_filtered['S2Si'])

        train_batch_filtered['train'] = jnp.vstack((train_batch_filtered['S2Si'],
                                                   train_batch_filtered['SIPM_FAKE']))

        labels = []
        
        ## [real, fake ] 1 hot ####

        for c in range(0,2*l):
            if c < l:
                labels.append(jnp.array((0.95,0.05)))
            else:
                labels.append(jnp.array((0.05,0.95)))

        train_batch_filtered['Labels'] = jnp.array(labels)
        
        train, labels = sklearn.utils.shuffle(train_batch_filtered['train'],
                                              train_batch_filtered['Labels'])


        batch = {'Train': train, 'Labels' :labels}

        return batch
    
