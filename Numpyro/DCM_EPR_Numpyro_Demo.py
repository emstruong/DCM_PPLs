#!/usr/bin/env python
# coding: utf-8

# # ODE  model of NeuroImaging with Bayesian Inference in Numpyro
# 
# :::
# 
# :post: April 12, 2024
# :tags: ODE model in Nympyro
# :category: Intermediate, Demo
# :author: Meysam HASHEMI, INS, AMU, Marseille.
# :acknowledgment: Nina BLADY, Cyprien DAUTREVAUX and Matthieu GILSON, and Marmadule WOODMAN. 
# 
# :::




import os
import sys
import time
import errno
import timeit
import pathlib

import numpy as np
import arviz as az
import matplotlib.pyplot as plt





import jax 
import jax.numpy as jnp
from jax import grad, vmap, lax, random
from jax.experimental.ode import odeint





import numpyro as npr
from numpyro import sample, plate, handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value




npr.set_platform("cpu")


print ("-"*60)
print ("-"*60)
print('Deependencies:')


print(f"Numpy version: {np.__version__}")
print(f"JAX version: {jax.__version__}")
print(f"Numpyro version: {npr.__version__}")
print(f"Arviz version: {az.__version__}")




import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action="ignore", category=FutureWarning)




az.style.use("arviz-darkgrid")
colors_l = ["#A4C3D9", "#7B9DBF", "#52779F", "#2A537E"] 




cwd = os.getcwd()
main_path = str(pathlib.Path.cwd().parent)
sys.path.append(main_path) # Path to import the model and solver

output_dir= cwd + '/output_numpyro/'


try:
    os.makedirs(output_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


### Simulator


from ForwardModel.ERPmodel_JAX import DCM_ERPmodel, odeint_euler, odeint_huen, odeint_rk4
from Helper.ERPhelper import *



from jax import random
rng_key = random.PRNGKey(0)




tend = 200.0
dt = 0.1
t0 = 0.0
ts = np.arange(t0, tend + dt, dt)
nt = ts.shape[0]




ns = 9
x_init=np.zeros((ns))




theta_true = np.array([0.42, 0.76, 0.15, 0.16, 12.13, 5.77, 27.87, 7.77, 1.63, 3.94])
n_params = theta_true.shape[0]




my_var_names = ['g_1', 'g_2', 'g_3', 'g_4', 'delta', 'tau_i', 'h_i', 'tau_e', 'h_e', 'u']


print ("-"*60)
print ("-"*60)
print('Running 1 simuation using different integrator of JAX odeint:')


# Run the model



start_time = time.time()

xs_euler = odeint_euler(DCM_ERPmodel, x_init, ts, theta_true)

print("similations took using Euler odeint (sec):" , (time.time() - start_time))



start_time = time.time()

xs_huen = odeint_rk4(DCM_ERPmodel, x_init, ts, theta_true)

print("similations took using Heun odeint (sec):" , (time.time() - start_time))




start_time = time.time()

xs_rk4 = odeint_rk4(DCM_ERPmodel, x_init, ts, theta_true)

print("similations took using RK4 odeint (sec):" , (time.time() - start_time))





plt.figure(figsize=(6,4))
plt.plot(ts, xs_euler[:,8],'--', color='g', lw=4, label='JAX odeint Euler');
plt.plot(ts, xs_huen[:,8],'--', color='b', lw=3, label='JAX odeint Heun');
plt.plot(ts, xs_rk4[:,8],'--', color='r', lw=2, label='JAX odeint RK4');
plt.legend(fontsize=10, frameon=False, loc='upper right')
plt.ylabel('Voltage [mV]', fontsize=14); 
plt.xlabel('Time [ms]', fontsize=14); 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout();
plt.savefig(os.path.join((output_dir),"Simulators.png"), dpi=300)


# So, we use Euler integration, But don't worry about computional time! we put JAX's JIT on Odeint to make it more faster!



@jax.jit
def ERP_JAXOdeintSimuator(x_init, ts, params):

    xs_rk4 = odeint_euler(DCM_ERPmodel, x_init, ts, params)    
    x_py=xs_rk4[:,8]
    
    return x_py


# The initial compilation takes a bit of time, but after that, it flies through the air!



start_time = time.time()

xpy_jax=ERP_JAXOdeintSimuator(x_init, ts, theta_true)

print("similations with compiling took (sec):" , (time.time() - start_time))




start_time = time.time()

xpy_jax=ERP_JAXOdeintSimuator(x_init, ts, theta_true)

print("similations using JAX's JIT took (sec):" , (time.time() - start_time))


# ## Synthetic Observation

# We assume that we only have accessto the activity of pyramidfal neurons, and for the sake of sppeding the computational time, we downsample the simuations.



#observation noise
sigma_true = 0.1 




xpy_jax = ERP_JAXOdeintSimuator(x_init, ts, theta_true)
x_noise = np.random.normal(loc=0, scale=sigma_true, size=xpy_jax.shape)
x_py = xpy_jax + x_noise



#downsampling
ds=10



ts_obs=ts[::ds]
xpy_obs=x_py[::ds]
nt_obs=int(x_py[::ds].shape[0])





data= { 'nt_obs': nt_obs, 'ds': ds, 'ts': ts, 'ts_obs': ts_obs, 'dt': dt, 'x_init': x_init, 'obs_err': sigma_true, 'xpy_obs': xpy_obs }




plot_obsrvation(ts, xpy_jax, ts_obs, xpy_obs)
plt.savefig(os.path.join((output_dir),"Observation.png"), dpi=300)


# ## Prior



shape=[18.16, 29.9, 29.14, 30.77, 22.87, 34.67, 20.44, 33.02, 24.17, 23.62]
scale=[0.03, 0.02, 0.005, 0.007, 0.51, 0.23, 0.96, 0.16, 0.07, 0.13]
rate = 1. / np.array(scale)




prior_specs = dict(shape=shape, rate=rate)




def model(data, prior_specs):
    #Data
    dt = data['dt']
    ts = data['ts']
    ds = data['ds']
    nt_obs = data['nt_obs']
    x_init = data['x_init']
    obs_err= data['obs_err']
    xpy_obs = data['xpy_obs']

    # Prior               
    g_1 = npr.sample('g_1', dist.Gamma(prior_specs['shape'][0], prior_specs['rate'][0]))
    g_2 = npr.sample('g_2', dist.Gamma(prior_specs['shape'][1], prior_specs['rate'][1]))
    g_3 = npr.sample('g_3', dist.Gamma(prior_specs['shape'][2], prior_specs['rate'][2]))
    g_4 = npr.sample('g_4', dist.Gamma(prior_specs['shape'][3], prior_specs['rate'][3]))
    delta = npr.sample('delta', dist.Gamma(prior_specs['shape'][4], prior_specs['rate'][4]))
    tau_i = npr.sample('tau_i', dist.Gamma(prior_specs['shape'][5], prior_specs['rate'][5]))
    h_i = npr.sample('h_i', dist.Gamma(prior_specs['shape'][6], prior_specs['rate'][6]))
    tau_e = npr.sample('tau_e', dist.Gamma(prior_specs['shape'][7], prior_specs['rate'][7]))
    h_e = npr.sample('h_e', dist.Gamma(prior_specs['shape'][8], prior_specs['rate'][8]))
    u = npr.sample('u', dist.Gamma(prior_specs['shape'][9], prior_specs['rate'][9]))
     
    #Parameters    
    params_samples=[g_1, g_2, g_3, g_4, delta, tau_i,  h_i, tau_e, h_e, u]
    
    #Forward model
    xpy_hat=ERP_JAXOdeintSimuator(x_init, ts, params_samples)[::ds]
    
    # Likelihood
    with plate('data', size=nt_obs):
        xpy_model = npr.deterministic('xpy_model', xpy_hat)
        npr.sample('xpy_obs', dist.Normal(xpy_model, sigma_true), obs=xpy_obs)



# ### Prior predictive check

# In[38]:


n_ = 100
prior_predictive = Predictive(model, num_samples=n_)
prior_predictions = prior_predictive(rng_key, data, prior_specs)




title='Prior Predictive Check'
plot_priorcheck(ts_obs, xpy_obs, prior_predictions, n_, title)
plt.savefig(os.path.join((output_dir),"PriorPredictiveCheck.png"), dpi=300)


# ## NUTS sampling 

#  Due to large dimentionality of problem and the nonlinear relation between parameeters, the multimodality is omnipresence in this case. In the follwing , we run 4 NUTS chains with default configurations that operates across diverse problems, but not necessarliy leads to convergence. Then we tune the algorithmic parameetrs for better convergence, however, resulting in multimodality. Finnaly, we propose the weighted stacking the chains as a solution to deal with this challnge. 

# NOTES: By default set-up, the chains may converge or not. In particular, it may be seen that some samples hits the max tree-depth, which then lead to no convergence. The convergence can be checked by monitoring the \hat R, close to 1 as a rule of thumb, also the Rank Plot, and Effective Sample Size (EES) which we will see in the following.

# Now we run the chains at the tail of prior to get convergence for all chains.



n_warmup, n_samples, n_chains= 200, 200, 4




tails_5th_percentile=tails_percentile(my_var_names, prior_predictions, 0.05)    
init_to_low_prob = init_to_value(values=tails_5th_percentile)




# NUTS set up
kernel = NUTS(model, max_tree_depth=12,  dense_mass=False, adapt_step_size=True, init_strategy=init_to_low_prob)
mcmc= MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains, chain_method='parallel')



print ("-"*60)
print ("-"*60)
print('Running the chains:')


#RUN NUTS
start_time = time.time()

mcmc.run(rng_key, data, prior_specs, extra_fields=('potential_energy', 'num_steps', 'diverging'))



print('Terminated the sampling.')
print ("-"*60)
print ("-"*60)

print(" All Chains using NUTS' Numpyro took (sec):" , (time.time() - start_time))


# The values of r_hat ~1 show the convergence. This convergence leads to a large effective sample size.




mcmc.print_summary(exclude_deterministic=True)





lp = -mcmc.get_extra_fields()['potential_energy']
print('Expected log joint density: {:.2f}'.format(np.mean(lp)))





title='Converged chains'
plot_lp_chains(lp, n_chains, title)
plt.savefig(os.path.join((output_dir),"Lp__.png"), dpi=300)


# ### Posterior 


# Get posterior samples
posterior_samples = mcmc.get_samples(group_by_chain=True)
pooled_posterior_samples = mcmc.get_samples()


# vizualize with arviz


az_obj = az.from_numpyro(mcmc)


# showing the posterior samples of all chains


axes = az.plot_trace(
    az_obj,
    var_names=my_var_names,
    compact=True,
    kind="trace",
    backend_kwargs={"figsize": (6, 14), "layout": "constrained"},)

for ax, true_val in zip(axes[:, 0], theta_true):
    ax.axvline(x=true_val, color='red', linestyle='--')  
for ax, true_val in zip(axes[:, 1], theta_true):
    ax.axhline(y=true_val, color='red', linestyle='--')
    
plt.gcf().suptitle("Converged NUTS", fontsize=16)
plt.tight_layout();
plt.savefig(os.path.join((output_dir),"ConvergedChains.png"), dpi=300)



chains_pooled = az_obj.posterior[my_var_names].to_array().values.reshape(n_params, -1)
params_map_pooled=calcula_map(chains_pooled)




title="Pooled Posteriors"
plot_posterior_pooled(my_var_names, theta_true, prior_predictions, chains_pooled, title)
plt.savefig(os.path.join((output_dir),"PooledPosterior.png"), dpi=300)


# ### Fit and Posterior predictive check 



plot_fitted(data, az_obj.posterior)
plt.savefig(os.path.join((output_dir),"Fitteddata.png"), dpi=300)





pooled_posterior_predictive = Predictive(model=model, posterior_samples=pooled_posterior_samples)
rng_key, rng_subkey = random.split(key=rng_key)
pooled_posterior_predictive_samples = pooled_posterior_predictive(rng_subkey, data, prior_specs)

ppc_=pooled_posterior_predictive_samples['xpy_model']
xpy_per05_pooled=np.quantile(ppc_, 0.05, axis=0)
xpy_per95_pooled=np.quantile(ppc_, 0.95, axis=0)




title='Posterior Predictive Check'
plot_posteriorcheck(data, xpy_per05_pooled, xpy_per95_pooled, title)
plt.savefig(os.path.join((output_dir),"PosteriorPredictiveCheck.png"), dpi=300)


print ("-"*60)
print ("-"*60)
print('The end!')
print ("-"*60)
print ("-"*60)

