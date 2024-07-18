import numpyro as npr
import numpyro.distributions as dist
import jax 

import os
import pathlib
import sys

cwd = os.getcwd()
main_path = str(pathlib.Path.cwd().parent)
sys.path.append(main_path) # Path to import the model and solver


from ForwardModel.ERPmodel_JAX import DCM_ERPmodel, odeint_euler


@jax.jit
def ERP_JAXOdeintSimulator(x_init, ts, params):

    xs_rk4 = odeint_euler(DCM_ERPmodel, x_init, ts, params)    
    x_py=xs_rk4[:,8]
    
    return x_py


def make_model(data, prior_specs, model_id=0) :
    
    if model_id == 0 :
        
        def model(data, prior_specs, model_id=0):
            #Data
            dt = data['dt']
            ts = data['ts']
            ds = data['ds']
            nt_obs = data['nt_obs']
            x_init = data['x_init']
            obs_err= data['obs_err']
            obs = data['xpy_obs']

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
            xpy_hat=ERP_JAXOdeintSimulator(x_init, ts, params_samples)[::ds]

            # Likelihood
            with npr.plate('data', size=nt_obs):
                xpy_model = npr.deterministic('xpy_model', xpy_hat)
                npr.sample('xpy_obs', dist.Normal(xpy_model, obs_err), obs=obs)
                xpy_ppc = npr.sample('xpy_ppc', dist.Normal(xpy_model, obs_err))
                
    elif model_id == 1 :
        
        def model(data, prior_specs, model_id=0):
            #Data
            dt = data['dt']
            ts = data['ts']
            ds = data['ds']
            nt_obs = data['nt_obs']
            x_init = data['x_init']
            obs_err= data['obs_err']
            obs = data['xpy_obs']

            # Prior   
            g_1 = 0
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
            xpy_hat=ERP_JAXOdeintSimulator(x_init, ts, params_samples)[::ds]

            # Likelihood
            with npr.plate('data', size=nt_obs):
                xpy_model = npr.deterministic('xpy_model', xpy_hat)
                npr.sample('xpy_obs', dist.Normal(xpy_model, obs_err), obs=obs)
                xpy_ppc = npr.sample('xpy_ppc', dist.Normal(xpy_model, obs_err))
                
    elif model_id == 2 :
    
        def model(data, prior_specs, model_id=0):
            #Data
            dt = data['dt']
            ts = data['ts']
            ds = data['ds']
            nt_obs = data['nt_obs']
            x_init = data['x_init']
            obs_err= data['obs_err']
            obs = data['xpy_obs']

            # Prior   
            g_1 = npr.sample('g_1', dist.Gamma(prior_specs['shape'][0], prior_specs['rate'][0]))
            g_2 = 0
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
            xpy_hat=ERP_JAXOdeintSimulator(x_init, ts, params_samples)[::ds]

            # Likelihood
            with npr.plate('data', size=nt_obs):
                xpy_model = npr.deterministic('xpy_model', xpy_hat)
                npr.sample('xpy_obs', dist.Normal(xpy_model, obs_err), obs=obs)
                xpy_ppc = npr.sample('xpy_ppc', dist.Normal(xpy_model, obs_err))
                
    elif model_id == 3 :
    
        def model(data, prior_specs, model_id=0):
            #Data
            dt = data['dt']
            ts = data['ts']
            ds = data['ds']
            nt_obs = data['nt_obs']
            x_init = data['x_init']
            obs_err= data['obs_err']
            obs = data['xpy_obs']

            # Prior   
            g_1 = npr.sample('g_1', dist.Gamma(prior_specs['shape'][0], prior_specs['rate'][0]))
            g_2 = npr.sample('g_2', dist.Gamma(prior_specs['shape'][1], prior_specs['rate'][1]))
            g_3 = 0
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
            xpy_hat=ERP_JAXOdeintSimulator(x_init, ts, params_samples)[::ds]

            # Likelihood
            with npr.plate('data', size=nt_obs):
                xpy_model = npr.deterministic('xpy_model', xpy_hat)
                npr.sample('xpy_obs', dist.Normal(xpy_model, obs_err), obs=obs)
                xpy_ppc = npr.sample('xpy_ppc', dist.Normal(xpy_model, obs_err))
                
    elif model_id == 4 :
    
        def model(data, prior_specs, model_id=0):
            #Data
            dt = data['dt']
            ts = data['ts']
            ds = data['ds']
            nt_obs = data['nt_obs']
            x_init = data['x_init']
            obs_err= data['obs_err']
            obs = data['xpy_obs']

            # Prior   
            g_1 = npr.sample('g_1', dist.Gamma(prior_specs['shape'][0], prior_specs['rate'][0]))
            g_2 = npr.sample('g_2', dist.Gamma(prior_specs['shape'][1], prior_specs['rate'][1]))
            g_3 = npr.sample('g_3', dist.Gamma(prior_specs['shape'][2], prior_specs['rate'][2]))
            g_4 = 0
            delta = npr.sample('delta', dist.Gamma(prior_specs['shape'][4], prior_specs['rate'][4]))
            tau_i = npr.sample('tau_i', dist.Gamma(prior_specs['shape'][5], prior_specs['rate'][5]))
            h_i = npr.sample('h_i', dist.Gamma(prior_specs['shape'][6], prior_specs['rate'][6]))
            tau_e = npr.sample('tau_e', dist.Gamma(prior_specs['shape'][7], prior_specs['rate'][7]))
            h_e = npr.sample('h_e', dist.Gamma(prior_specs['shape'][8], prior_specs['rate'][8]))
            u = npr.sample('u', dist.Gamma(prior_specs['shape'][9], prior_specs['rate'][9]))

            #Parameters    
            params_samples=[g_1, g_2, g_3, g_4, delta, tau_i,  h_i, tau_e, h_e, u]

            #Forward model
            xpy_hat=ERP_JAXOdeintSimulator(x_init, ts, params_samples)[::ds]

            # Likelihood
            with npr.plate('data', size=nt_obs):
                xpy_model = npr.deterministic('xpy_model', xpy_hat)
                npr.sample('xpy_obs', dist.Normal(xpy_model, obs_err), obs=obs)
                xpy_ppc = npr.sample('xpy_ppc', dist.Normal(xpy_model, obs_err))
                
                
    return model
