import numpy as np 
import pymc as pm
import pytensor
import numpyro as npr
import jax

from ForwardModel.ERPmodel_JAX import DCM_ERPmodel, odeint_euler

c=-0.56


def Sigmodal(x1, x2, delta, c):
    S=(1./(1.+np.exp(c*(x1-(delta*x2))))) - 0.5
    return S


@jax.jit
def ERP_JAXOdeintSimulator(x_init, ts, params):
    xs_rk4 = odeint_euler(DCM_ERPmodel, x_init, ts, params)    
    x_py=xs_rk4[:,8]
    return x_py



def ode_update_function(x0, x1, x2, x3, x4, x5, x6, x7, x8, 
                        g_1, g_2, g_3, g_4, delta, tau_i, h_i, tau_e, h_e, u,
						dt ):

    dx0=x3
    dx1=x4
    dx2=x5
    dx6=x7
    dx3=(1./tau_e)*(h_e*(g_1*(Sigmodal(x8, x4-x5, delta, c))+u)-(x0/tau_e)-2*x3)
    dx4=(1./tau_e)*(h_e*(g_2*(Sigmodal(x0, x3, delta, c)))-(x1/tau_e)-2*x4)
    dx5=(1./tau_i)*(h_i*(g_4*(Sigmodal(x6, x7, delta, c)))-(x2/tau_i)-2*x5)
    dx7=(1./tau_e)*(h_e*(g_3*(Sigmodal(x8, x4-x5, delta, c)))-(x6/tau_e)-2*x7)
    dx8=x4-x5

    x0_new = x0 + dt * dx0 
    x1_new = x1 + dt * dx1 
    x2_new = x2 + dt * dx2 
    x3_new = x3 + dt * dx3 
    x4_new = x4 + dt * dx4 
    x5_new = x5 + dt * dx5 
    x6_new = x6 + dt * dx6 
    x7_new = x7 + dt * dx7 
    x8_new = x8 + dt * dx8 

    return x0_new, x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new



def build_model_pymc(data, prior_specs, integration_specs):

    # PyMC model
    with pm.Model() as model:
        # Priors
        g_1 = pm.Gamma('g_1', alpha=prior_specs['shape'][0], beta=prior_specs['rate'][0])
        g_2 = pm.Gamma('g_2', alpha=prior_specs['shape'][1], beta=prior_specs['rate'][1])
        g_3 = pm.Gamma('g_3', alpha=prior_specs['shape'][2], beta=prior_specs['rate'][2])
        g_4 = pm.Gamma('g_4', alpha=prior_specs['shape'][3], beta=prior_specs['rate'][3])
        delta = pm.Gamma('delta', alpha=prior_specs['shape'][4], beta=prior_specs['rate'][4])
        tau_i = pm.Gamma('tau_i', alpha=prior_specs['shape'][5], beta=prior_specs['rate'][5])
        h_i = pm.Gamma('h_i', alpha=prior_specs['shape'][6], beta=prior_specs['rate'][6])
        tau_e = pm.Gamma('tau_e', alpha=prior_specs['shape'][7], beta=prior_specs['rate'][7])
        h_e = pm.Gamma('h_e', alpha=prior_specs['shape'][8], beta=prior_specs['rate'][8])
        u = pm.Gamma('u', alpha=prior_specs['shape'][9], beta=prior_specs['rate'][9])
                        
        result, updates = pytensor.scan(
                fn=ode_update_function,  
                outputs_info=integration_specs['init_state'],  
                non_sequences=[g_1, g_2, g_3, g_4, delta, tau_i, h_i, tau_e, h_e, u, integration_specs['dt']],  
                n_steps=integration_specs['n_steps'])

        final_result = pm.math.stack([result[8]], axis=1)
        
        xpy_model = pm.Deterministic('xpy_model', final_result[::integration_specs['ds']])

        # Likelihood function
        pm.Normal('xpy_obs', mu=xpy_model, sigma=np.unique(data['obs_noise']), observed=data[['xpy_obs']].values)
        
    return model



def build_model_numpyro(data, prior_specs, integration_specs):
    #Data
    nt_obs = data['nt_obs']
    xpy_obs = data['xpy_obs']
    obs_err = data['obs_noise']

    # Prior               
    g_1 = npr.sample('g_1', npr.distributions.Gamma(prior_specs['shape'][0], prior_specs['rate'][0]))
    g_2 = npr.sample('g_2', npr.distributions.Gamma(prior_specs['shape'][1], prior_specs['rate'][1]))
    g_3 = npr.sample('g_3', npr.distributions.Gamma(prior_specs['shape'][2], prior_specs['rate'][2]))
    g_4 = npr.sample('g_4', npr.distributions.Gamma(prior_specs['shape'][3], prior_specs['rate'][3]))
    delta = npr.sample('delta', npr.distributions.Gamma(prior_specs['shape'][4], prior_specs['rate'][4]))
    tau_i = npr.sample('tau_i', npr.distributions.Gamma(prior_specs['shape'][5], prior_specs['rate'][5]))
    h_i = npr.sample('h_i', npr.distributions.Gamma(prior_specs['shape'][6], prior_specs['rate'][6]))
    tau_e = npr.sample('tau_e', npr.distributions.Gamma(prior_specs['shape'][7], prior_specs['rate'][7]))
    h_e = npr.sample('h_e', npr.distributions.Gamma(prior_specs['shape'][8], prior_specs['rate'][8]))
    u = npr.sample('u', npr.distributions.Gamma(prior_specs['shape'][9], prior_specs['rate'][9]))
     
    #Parameters    
    params_samples=[g_1, g_2, g_3, g_4, delta, tau_i,  h_i, tau_e, h_e, u]
    
    #Forward model
    xpy_hat=ERP_JAXOdeintSimulator(integration_specs['init_state'], 
                                  integration_specs['ts'], 
                                  params_samples)[::integration_specs['ds']]
    
    # Likelihood
    with npr.plate('data', size=nt_obs):
        xpy_model = npr.deterministic('xpy_model', xpy_hat)
        npr.sample('xpy_obs', npr.distributions.Normal(xpy_model, obs_err), obs=xpy_obs)


