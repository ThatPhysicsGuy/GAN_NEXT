import sys, os
import jax
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.example_libraries import stax

import jax.tree_util as tree_util # for applying learning rate to gradients

# This is a global variable defining the sipm locations:
sipms_1D = jnp.arange(-235, 235, 10.) + 5
n_sipms = sipms_1D.shape[0]
sipm_locations_x = jnp.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms))
sipm_locations_y = jnp.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms)).transpose()

sipm_locations = jnp.stack([sipm_locations_y, sipm_locations_x], -1)
#print(sipm_locations)
# JAX Is pure functional programming.  So, we're gonna build up to one function here:


# Build the whole thing as a function we can JIT:
#@jit
def simulate_waveforms(energy_depositions, parameters, noise, key):

    # Energy depositions comes in with shape [batch, max_deps_per_event, 4]
    # where 4 represenets (x,y,z,E)

    # Split the key for generation:
    key, subkey = random.split(key)
    generated_electrons, valid_electrons = generate_electrons_batch(energy_depositions, subkey)

    # The generated electrons are per energy deposition, so it has shape
    # [batch, max_deps_per_event, max_electrons_per_dep, 3], 3 for x/y/z
    # Split the key for diffusion
    key, subkey = random.split(key)
    diffused = diffuse_electrons_batch(generated_electrons, parameters["diffusion"],subkey)

    # Here, diffused electrons is of the shape [batch_size, ]

    lifetime = compute_lifetime(diffused, parameters["lifetime"], valid_electrons)

    # Split off the XY and Z:

    diffused_xy = diffused[:,:,:,0:2]
    diffused_z  = diffused[:,:,:,2:]
    
   
    # Compute the PMT response to each energy deposition:
    pmt_response = batch_pmt_nn_apply(parameters["pmt_network"], diffused_xy)**2
    # print("pmt_response.shape: ", pmt_response.shape )
    # Normalize the PMT response per PMT:
    pmt_response = pmt_response * (parameters["pmt_dynamic_range"])
    
    #sigmas_pmt = batch_get_sigma_pmt(generated_electrons[:,:,0,2])
    #sigmas_pmt = sigmas_pmt * 0.1/parameters["diffusion"][2]
    #sigmas_pmt = jnp.where(sigmas_pmt == 0.0,0.01,sigmas_pmt)
    sigma = 0.2 * 0.3/parameters["diffusion"][2]
    # Compute the waveforms and sum over the individual depositions:
    pmt_waveforms = batch_build_pmt_waveforms(pmt_response, diffused_z, lifetime, sigma).sum(axis=1)

    # The sipm respone network outputs a value between 0 and 1 for each xy location
    # It is multiplied by an overall scale factor.
    # Square it to ensure positive:
    sipm_response = batch_sipm_nn_apply(parameters["sipm_network"], diffused_xy)
   
    sigmas_sipm = batch_get_sigma_sipm(generated_electrons[:,:,0,2])
    
    # This function takes the sipm light production and maps it onto waveforms.
    # So the weights are applied here, too.
    # The parameter el_spread indicates how far in xy each electron's light
    # spreads, modeled as a gaussian.  The output is not yet summed over electrons.
    sipm_waveforms = batch_build_sipm_waveforms(sipm_response, diffused_xy, diffused_z, lifetime, sigmas_sipm, parameters["el_spread"])
    # Sum over all electrons
    
    
    sipm_waveforms = (sipm_waveforms).sum(axis=1) #usually amplitude < 6 units
    
    #sipm_waveforms = batch_noise_percentage(sipm_waveforms,subkey)
    sipm_waveforms = batch_noise_constant(sipm_waveforms,noise, subkey)
    # This parameter accounts for sipm-to-sipm variations
    sipm_waveforms = sipm_waveforms * (parameters["sipm_dynamic_range"])
    # print("sipm_waveforms.shape: ", sipm_waveforms.shape )

    return pmt_waveforms, sipm_waveforms

# Another critical function is the parameter initialization:
def init_params(key, example_input):

    # The ijnput here needs to be a single electron deposition
    # but we splice that off here, to make it easier.

    # Index into the batch dimension and the energy deposit in the batch:
    pmt_input, n_valid = generate_electrons(example_input["energy_deposits"][0][0], key)
    # Diffuse them:
    pmt_input = diffuse_electrons(pmt_input, jnp.asarray([0.4,0.4,0.3]),key)

    input_shape = pmt_input[:,0:2].shape

    pmt_output_size,  pmt_network_params  =  pmt_nn_init(key, input_shape)
    sipm_output_size, sipm_network_params =  sipm_nn_init(key, input_shape)


    parameters = {
        "diffusion"     : jnp.asarray([0.4,0.4,0.3]),
        "lifetime"      : 5000*jnp.ones(1),
        "pmt_network"   : pmt_network_params,
        "sipm_network"  : sipm_network_params,
        "el_spread"     : 6*jnp.ones(1),
        "pmt_dynamic_range" : jnp.ones(12),
        #"scale_pmt_sigma"  : jnp.ones(1),
        #"powers"         : jnp.asarray([0.6,0.5,0.5]),
        #"betas"          : jnp.asarray([30.,0.,0.])
        #"el_amplification" : 12*jnp.ones(1),
        "sipm_dynamic_range" : jnp.ones([1,47,47,1]),
        #"waveform_sigma" : 0.2 * jnp.ones(1),
    }

    return parameters


@jit
def generate_electrons(energy_and_position, key):
    '''
    Generate a sequence of electrons given an energy amount.
    A random number of electrons are generated, but a static-sized array
    is returned.  This ensures jit'ing is possible later.
    The number of electrons in the array that is valid is also returned
    '''

    energy   = energy_and_position[-1]
    position = energy_and_position[0:3]

    # For each energy, compute n:
    n = energy * 1000.*1000. / 22.4
    sigmas = jnp.sqrt(n * 0.15)
    # Generate a sample for each energy:
    n_electrons = (sigmas*random.normal(key) + n).astype(jnp.int32)

    shape = (2000, 3)
    el_arr = jnp.broadcast_to(position, shape)

    return el_arr, n_electrons

# Vmap and jit these out to batches:
generate_electrons_event = jit(vmap(generate_electrons, in_axes=[0, None]))
generate_electrons_batch = jit(vmap(generate_electrons_event, in_axes=[0, None]))


#def f(z,powers):
    #alpha = jnp.where(z<150,powers[0],powers[1])
    #alpha = jnp.where(z>350,powers[2],alpha)
    #return z ** alpha
    
#def g(z,betas):
    #beta = jnp.where(z<150,betas[0],betas[1])
    #beta = jnp.where(z>350,betas[2],beta)
    #return beta

@jit
def diffuse_electrons(electrons, diffusion_scale, key):
    '''
    Apply diffusion to the electrons in a single energy deposition.
    '''

    # Then input should be an array of shape [max_electrons, 3]
    # only some of these are valid, but that's ok, it's not
    # that wasteful to diffuse them all.

    # Get the z position (we only need this as a scalar!):
    z = electrons[-1,-1]
    n_samples = electrons.shape[0]
    
    # Sample from a normal distribution:
    kicks = jax.random.normal(key, (electrons.shape))
    #scalez wrt the location using different powers
    #scaled_z = f(z,powers)
    # add a constant spread depending on the z location
    #beta = g(z,betas)
    
    # Scale the kicks by the diffusion scale and sqrt(z)
    return electrons + (diffusion_scale**2)* jnp.sqrt(z) * kicks


# Likewise, jit and vmap these out:
diffuse_electrons_event = jit(vmap(diffuse_electrons, in_axes=[0,None, None]))
diffuse_electrons_batch = vmap(jit(diffuse_electrons_event), in_axes=[0,None, None])


# The lifetime gets finnicky if I put these two steps into one function.
# I don't know why yet.  But this works.  It also applies 0 weights to invalid electrons.


@jit
def s_compute_probability(_diffused_electrons, _lifetime):
    _z = _diffused_electrons[:,-1]
    probability = jnp.exp(- _z / _lifetime)

    return probability

@jit
def s_compute_mask(_diffused_electrons, _n_valid):
    # This technique is "stupid" but it jit's and vmap's, so it's not THAT stupid
    arange = jnp.arange(len(_diffused_electrons))
    mask = arange < _n_valid
    mask = mask.astype("float32")
    return mask

e_compute_probability = jit(vmap(s_compute_probability, in_axes=[0,None]))
compute_probability = jit(vmap(e_compute_probability, in_axes=[0,None]))

e_compute_mask = jit(vmap(s_compute_mask))
compute_mask = jit(vmap(e_compute_mask))

@jit
def compute_lifetime(electrons_batch, _lifetime, n_valid_batch):
    return compute_mask(electrons_batch, n_valid_batch) * compute_probability(electrons_batch, _lifetime)


# Define the PMT network and it's worker functions:
pmt_nn_init, pmt_nn_apply = stax.serial(
    # stax.Dense(28), stax.Tanh,
    #stax.Dense(28), stax.Tanh,
    stax.Dense(28), stax.Sigmoid,
    stax.Dense(12), stax.Sigmoid,
    stax.Dense(1)
)

event_pmt_nn_apply = jit(vmap(jit(pmt_nn_apply), in_axes=[None, 0]))
batch_pmt_nn_apply = jit(vmap(event_pmt_nn_apply, in_axes=[None, 0]))

# Define the SiPM network and it's worker functions:
sipm_nn_init, sipm_nn_apply = stax.serial(
    stax.Dense(28),stax.Sigmoid,
    stax.Dense(256),stax.Sigmoid,
    stax.Dense(32),stax.Sigmoid,
    #stax.Dense(512), stax.Tanh,
    stax.Dense(1)
)
event_sipm_nn_apply = jit(vmap(jit(sipm_nn_apply), in_axes=[None, 0]))
batch_sipm_nn_apply = jit(vmap(event_sipm_nn_apply, in_axes=[None, 0]))


#@jit
#def get_sigma_pmt(_z_positions):
    #sigma = jnp.asarray(6*jnp.power(_z_positions,-0.42),dtype = 'float32')
    #return  jnp.where(sigma == jnp.inf, 0.001, sigma)
    
#e_get_sigma_pmt = jit(vmap(get_sigma_pmt, in_axes=[0]))
#batch_get_sigma_pmt = jit(vmap(e_get_sigma_pmt, in_axes=[0]))

@jit
def get_sigma_sipm(_z_positions):
    sigma = jnp.asarray(jnp.sqrt(10*jnp.power(_z_positions,-0.56)),dtype = 'float32')
    return  jnp.where(sigma == jnp.inf, 0.001, sigma)
    
    
e_get_sigma_sipm = jit(vmap(get_sigma_sipm, in_axes=[0]))
batch_get_sigma_sipm = jit(vmap(e_get_sigma_sipm, in_axes=[0]))

# Functions to build waveforms based on weights and responses:
@jit
def build_pmt_waveforms(_sensor_response, _z_positions, _weights, _bin_sigma):
    '''
    Compute the PMT response to electrons on the EL region
    '''
    # This is basically a constant:
    _n_ticks=550

    n_electrons = _z_positions.shape[0]
    # Build a range for the exponential ijnput:
    starts = jnp.zeros(shape=(n_electrons)) # + 0.5
    stops  = jnp.ones(shape=(n_electrons)) * (_n_ticks) # + 0.5
    
    # Reshape z positions for broadcasting:
    _z_positions = _z_positions.reshape((-1,1))

    exp_input = jnp.linspace(start=starts, stop=stops, num=_n_ticks, axis=-1)

    exp_values = jnp.exp( - 0.5 * (exp_input - _z_positions)**2.  / (2. * _bin_sigma))

    # Normalize the values:
    exp_values = exp_values.transpose() * (0.39894228040/jnp.sqrt(_bin_sigma))

    # Scale by the weights:
    exp_values = exp_values * _weights

    # print("pmt exp_values.shape: ", exp_values.shape)
    # print("pmt _sensor_response.shape: ", _sensor_response.shape)
    waveforms = jnp.matmul(exp_values, _sensor_response)
    # print("pmt waveforms.shape: ", waveforms.shape)
    return waveforms.transpose()

e_build_pmt_waveforms = jit(vmap(build_pmt_waveforms, in_axes=[0,0,0,None]))
batch_build_pmt_waveforms = jit(vmap(e_build_pmt_waveforms, in_axes=[0,0,0,None]))


@jit
def apply_noise_constant(sipm_waveforms,noise,subkey):

    #noise_constant = random.uniform(subkey,shape = sipm_waveforms.shape,minval = -0.5, maxval = 1)
    #noise_constant = random.poisson(subkey,shape = sipm_waveforms.shape,lam = 0.05)
    #noise_constant = random.beta(subkey,shape = sipm_waveforms.shape,a = 2, b = 5)
    
    noised = sipm_waveforms + noise
    noised = jnp.where(noised < 1,0,noised)
    return noised


xy_noise_constant = jit(vmap(apply_noise_constant,in_axes=[0,0,None]))
event_noise_constant = jit(vmap(xy_noise_constant,in_axes=[0,0,None]))
batch_noise_constant = jit(vmap(event_noise_constant,in_axes=[0,0,None]))


# Functions to build waveforms based on weights and responses:
# @jit
def build_sipm_waveforms(_sensor_response, _xy_positions, _z_positions, _weights, _bin_sigma, el_spread):
    '''
    Compute the SIPM response to electrons on the EL region
    '''
    # This is basically a constant:
    _n_ticks=550

    # print("_sensor_response: ", _sensor_response)
    # print("_xy_positions: ", _xy_positions)
    # print("_z_positions: ", _z_positions)

    # print(jnp.min(_sensor_response))
    # print(jnp.max(_sensor_response))

    # exit()

    n_electrons = _z_positions.shape[0]
    

    # Build a range for the exponential ijnput:
    starts = jnp.zeros(shape=(n_electrons)) # + 0.5
    stops  = jnp.ones(shape=(n_electrons)) * (_n_ticks) # + 0.5

    # Reshape z positions for broadcasting:
    _z_positions = _z_positions.reshape((-1,1))

    # print("_sensor_response: ", _sensor_response.shape)
    # print("_xy_positions: ", _xy_positions.shape)
    # print("_z_positions: ", _z_positions.shape)
    # print("_weights: ", _weights.shape)
    # print(sipm_locations.shape)
    _xy_reshaped = _xy_positions.reshape((_xy_positions.shape[0], 1,1,_xy_positions.shape[-1]))
    subtracted_values = _xy_reshaped - sipm_locations
    # print(subtracted_values.shape)

    r_squared = (subtracted_values**2).sum(-1)
    # print(r_squared[0])
    # print("r_squared.shape: ", r_squared.shape)
    # print(jnp.min(r_squared[0]))

    # print(el_spread)
    # sipm_spread_response = jnp.exp(-0.5*(r_squared/el_spread)**2)  / (el_spread * 2.5066)
    sipm_spread_response = jnp.exp(-0.225 * (r_squared/(el_spread)**2) ) / (el_spread * 2.5066)
    # print(sipm_spread_response[0])
    # print(jnp.max(sipm_spread_response[0]))
    # print(jnp.argmax(sipm_spread_response[0]))
    # print(jnp.sum(sipm_spread_response[0]))
    # print(jnp.max(sipm_spread_response[1]))
    # print(jnp.argmax(sipm_spread_response[1]))
    # exit()

    # print("sipm_spread_response.shape: ", sipm_spread_response.shape)

    _sensor_response = sipm_spread_response * _sensor_response.reshape((-1,1,1))

    # Run the subtracted values through a gaussian response:

    # r_squared = subtracted_values**2
    # r_squared = 1./(r_squared.sum(axis=-1) + 1.0) # This extra addition is the "standoff distance"
    # print("r_squared.shape: ", r_squared.shape)
    # print("_sensor_response.shape: ", _sensor_response.shape)

    # Multiple the total light (sensor response) by the 1/r^2 scaling:
    # _sensor_response = _sensor_response.reshape((-1,1,1)) * r_squared

    # We put the subtracted differences through a 1/r^2 response

    exp_input = jnp.linspace(start=starts, stop=stops, num=_n_ticks, axis=-1)

    exp_values = jnp.exp( - 0.5 * (exp_input - _z_positions)**2.  / (2. * _bin_sigma))

    # Normalize the values:
    exp_values = exp_values.transpose() * (0.39894228040/jnp.sqrt(_bin_sigma))

    # Scale by the weights:
    exp_values = exp_values * _weights

    # print("exp_values.shape: ", exp_values.shape)

    # print("sipm _sensor_response.shape: ", _sensor_response.shape)
    # To do the matmul, we have to flatten the _sensor_response briefly
    _sensor_response_flat = _sensor_response.reshape((-1, 47*47))
    # print("sipm _sensor_response_flat.shape: ", _sensor_response_flat.shape)
    # print("sipm exp_values.shape: ", exp_values.shape)
    waveforms = jnp.matmul(exp_values, _sensor_response_flat)
    # print("sipm waveforms.shape: ", waveforms.shape)
    # And, unflatten:
    waveforms = waveforms.reshape((-1, 47, 47))
    return waveforms.transpose((1,2,0))


e_build_sipm_waveforms = vmap(build_sipm_waveforms, in_axes=[0,0,0,0,0,None])
batch_build_sipm_waveforms = vmap(e_build_sipm_waveforms, in_axes=[0,0,0,0,0, None])


def load_state(file):
    with open(file,"rb") as f:
        params = pickle.load(f)
    return params
