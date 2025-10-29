import pymc as pm
import pytensor.tensor as pt
import pytensor.tensor.extra_ops as pte
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

#Functions
def Ez(z, Om, w, wDM):
    opz = 1 + z
    return pt.sqrt(Om * opz ** (3 * (1 + wDM)) + (1 - Om) * opz ** (3 * (1 + w)))

def dCs(zs, Om, w, wDM):
    dz = zs[1:] - zs[:-1]
    fz = 1 / Ez(zs, Om, w, wDM)
    I = 0.5 * dz * (fz[:-1] + fz[1:])
    return pt.concatenate([pt.as_tensor([0.0]), pt.cumsum(I)])

def dLs(zs, dCs):
    return dCs * (1 + zs)

def pt_interp(x, xs, ys):
    x = pt.as_tensor(x)
    xs = pt.as_tensor(xs)
    ys = pt.as_tensor(ys)
    ind = pte.searchsorted(xs, x)
    ind = pt.clip(ind, 1, xs.shape[0] - 1)
    r = (x - xs[ind - 1]) / (xs[ind] - xs[ind - 1])
    return r * ys[ind] + (1 - r) * ys[ind - 1]

def fracHz(z, Om=0.3, w=-1.0):
    """Proxy for fractional uncertainty σH/H from log-derivatives wrt parameters."""
    opz = 1 + z
    denom = Om + (1 - Om) * opz ** (3*w)
    num_Om = 1 - opz ** (3*w)
    num_w  = (1 - Om) * 3 * opz ** (3*w) * np.log(opz)
    return np.sqrt(num_Om**2 + num_w**2) / (2 * denom)

#Finding the current seed
current_state = np.random.get_state()
current_seed = current_state[1][0]
print(f"Current random seed: {current_seed}")


#Mock_gen
Nobs = 100
z_true = np.random.beta(3, 9, Nobs) * 10
DL_true = cosmo.luminosity_distance(z_true).to(u.Gpc).value
sigma_DL = 0.07
DL_obs = np.random.normal(DL_true, sigma_DL)

mu_p_true = 1.17
sigma_p_true = 0.1
M_source_true = np.random.normal(mu_p_true, sigma_p_true, size=Nobs)
Mz_true = (1 + z_true) * M_source_true
Mz_obs = Mz_true #no noise

mu_q_true = 0.5
sigma_q_true = 0.1
q_true = np.clip(np.random.normal(mu_q_true, sigma_q_true, size=Nobs), 1e-4, 1-1e-4)#Generating only [0,1]
sigma_q_obs = 0.07
q_obs = np.clip(np.random.normal(q_true, sigma_q_obs), 1e-4, 1-1e-4)

m1_true = M_source_true * (1 + q_true) ** (1/5) / q_true ** (3/5)
m2_true = M_source_true * (1 + q_true) ** (1/5) * q_true ** (2/5)
#(m1+m2)/2 = 1.4
c0_true = 4.8
c1_true = -5.0
Lambda1_true = c0_true + c1_true * (m1_true - 1.4)
Lambda2_true = c0_true + c1_true * (m2_true - 1.4)
Lambda_til_true = (16.0 / 13.0) * ((m1_true + 12 * m2_true) * m1_true**4 * Lambda1_true + (m2_true + 12 * m1_true) * m2_true**4 * Lambda2_true) / (m1_true + m2_true) ** 5
sigma_lambda = 0.07
Lambda_obs = np.random.normal(Lambda_til_true, sigma_lambda)

zinterp = np.linspace(0, 10, 1000)

#Pivot redshift
z_grid = z_true
frac_vals = fracHz(z_grid)
z_pivot = float(z_grid[np.argmin(frac_vals)])
print("Using optimal z_pivot =", z_pivot) #better if ~(0.45-0.5)

with pm.Model() as model:

    #Cosmology priors
    H_pivot = pm.Uniform("H_pivot", 0.01, 2.0)
    Om = pm.Uniform("Om", 0.1, 0.5)
    w = pm.Uniform("w", -2.5, -0.3)
    wDM = 0.0
    #H(z_pivot) = H0 * Ez(z_pivot)
    z_piv_t = pt.as_tensor(z_pivot)
    H0 = pm.Deterministic("H0", H_pivot / Ez(z_piv_t, Om, w, wDM))
    dH = pm.Deterministic("dH", 2.99792 / H0)  # Gpc

    #Chirp Mass distribution
    mu_p = pm.Uniform("mu_p", 0.5, 2.0)
    sigma_p = pm.Uniform("sigma_p", 0.0, 0.5)

    #Mass ratio population hyper-parameters
    mu_q = pm.Uniform("mu_q", 0.01, 0.99)
    sigma_q = pm.Uniform("sigma_q", 0.0, 0.5)

    #Eos
    c0 = pm.Uniform("c0", 3.0, 6.0)
    c1 = pm.Uniform("c1", -7.0, -3.0)

    z_unit = pm.Beta("z_unit", 3, 9, shape=Nobs)
    z = pm.Deterministic("z", z_unit * 10)

    dCinterp = dH * dCs(zinterp, Om, w, wDM)
    dLinterp = dLs(zinterp, dCinterp)
    dL = pm.Deterministic("dL", pt_interp(z, zinterp, dLinterp))

    Mc = pm.Deterministic("Mc", Mz_obs / (1 + z))

    q = pm.TruncatedNormal("q", mu=mu_q, sigma=sigma_q, lower=0.0, upper=1.0, shape=Nobs)

    m1 = pm.Deterministic("m1", Mc * (1 + q) ** (1/5) / q ** (3/5))
    m2 = pm.Deterministic("m2", Mc * (1 + q) ** (1/5) * q ** (2/5))

    Lambda1 = pm.Deterministic("Lambda1", c0 + c1 * (m1 - 1.4))
    Lambda2 = pm.Deterministic("Lambda2", c0 + c1 * (m2 - 1.4))

    Lambda_til = pm.Deterministic("Lambda_til",(16.0 / 13.0)*((m1 + 12 * m2) * m1**4 * Lambda1 + (m2 + 12 * m1) * m2**4 * Lambda2)/(m1 + m2) ** 5)
  
    # Likelihoods & priors
    pm.Potential("mcprior", pt.sum(pm.logp(pm.Normal.dist(mu_p, sigma_p), Mc)))
    pm.Potential("mcjac", pt.sum(-pt.log1p(z)))
    pm.Normal("q_likelihood", mu=q, sigma=sigma_q_obs, observed=q_obs)
    pm.Normal("dL_likelihood", mu=dL, sigma=sigma_DL, observed=DL_obs)
    pm.Normal("Lambda_til_likelihood", mu=Lambda_til, sigma=sigma_lambda, observed=Lambda_obs)

    # Initial values
    initvals = {
        "H_pivot": cosmo.H(z_pivot).value / 100,
        "Om": cosmo.Om0,
        "w": -1.0,
        "mu_p": 1.17,
        "sigma_p": 0.1,
        "c0": 4.8,
        "c1": -5.0,
        "mu_q": 0.5,
        "sigma_q": 0.1,
    }

    trace = pm.sample(4000, tune=4000, target_accept=0.95, initvals=initvals, max_treedepth=15)


summary = az.summary(trace, var_names=["H_pivot", "H0", "Om", "w", "mu_p", "sigma_p", "c0", "c1", "mu_q", "sigma_q"])
print(summary)

az.plot_trace(trace, var_names=["H_pivot", "H0", "Om", "w", "mu_p", "sigma_p", "c0", "c1", "mu_q", "sigma_q"],
              lines=[('H_pivot', {}, cosmo.H(z_pivot).value/100),
                     ('H0', {}, cosmo.H0.value/100),
                     ('Om', {}, cosmo.Om0),
                     ('w', {}, -1),
                     ('mu_p', {}, 1.17),
                     ('sigma_p', {}, 0.1),
                     ('c0', {}, 4.8),
                     ('c1', {}, -5.0),
                     ('mu_q', {}, 0.5),
                     ('sigma_q', {}, 0.1)])
plt.tight_layout()
plt.show()

#Very less information about w for 100 obs. Need to increase number of obs. 
#Cosmology more sensitive for zpiv ~ (0.45-0.5).
#Very few results show multipeak for mup and c0.
