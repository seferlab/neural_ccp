# Packages
import jax.numpy as jnp
from jax import grad as fgrad
from jax import jit, vmap, jacfwd, jacrev
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad
import cvxpy as cp
from copy import deepcopy
import scipy
from scipy.stats import norm
np.set_printoptions(suppress=True)
CUDA = torch.cuda.is_available()
mse = nn.MSELoss()

# Global parameters
nx = 3000
ny = 3001
NBATCH = 128
ITERATIONS = 10000
NUMERICAL_METHOD = 'RK45'

# Neurodynaimc Optimization to form ODE systems
def ODE_psi(psi):
    def f(x):
        return -P1@x

    def g(x):
        output_1 = psi*jnp.linalg.norm(Sigma_sqrt@x, 2)-P2@x
        # output_2 = A@x-b
        output = jnp.array([output_1])
        return output

    df = fgrad(f)
    dg = jacrev(g)

    def G(y):
        x, u = y[:nx], y[nx:]
        output_1 = df(x) + u@dg(x)
        output_2 = -g(x)
        return jnp.concatenate([output_1, output_2])

    def projection(y):
        x, u = y[:nx], y[nx:]
        xp, up = np.clip(x, a_min=0, a_max=2), np.clip(u, a_min=0, a_max=None)
        return np.concatenate([xp, up])

    def ODE(t, y):
        x, u = y[:nx], y[nx:]
        xp, up = jnp.clip(x, a_min=0, a_max=2), jnp.clip(u, a_min=0, a_max=None)
        yp = jnp.concatenate([xp, up])

        dy = -G(yp)+yp-y
        return dy
    ODE = jit(ODE)

    def evalutation(y):
        x, u = y[:nx], y[nx:]
        gv = np.array(g(x))
        return f(x)+100*np.linalg.norm(np.clip(gv, a_min=0, a_max=None), 2)
    
    return  f, g, ODE, projection, evalutation
    

# Algorithm 1
T = 10
L_prediction = []
y0 = np.ones((3001, ))
net = FNN(y0)
if CUDA:
    net = net.cuda()
optimizer = Adam(net.parameters(), lr=0.0001, weight_decay=0.01)
NPE_best = 4070995.28

for alpha in np.arange(0.01, 0.50, 0.01):
    psi = norm.ppf(1-alpha)
    f, g, ODE, projection, evalutation = ODE_psi(psi)
    vODE = vmap(ODE)
    
    for i in range(200):
        # ORR mechanism
        pred_curr = predict(net)
        NPE_curr = evalutation(pred_curr)
        if NPE_curr < NPE_best:
            NPE_best = NPE_curr
            y0 = pred_curr

        # GD training
        t = np.random.uniform(0, T, (NBATCH, 1))
        t = torch.tensor(t, dtype=torch.float, requires_grad=True)
        if CUDA:
            t = t.cuda()
        loss = loss_compute(t, net, vODE)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
        # Monitor
        if i%10==0:
            print(f'alpha: {alpha}, Iteration: {i}, Loss: {loss.item() :.0f}, epsilon_current: {NPE_curr :.0f}, epsilon_best: {NPE_best :.2f}, y0:{y0}')
        if torch.isnan(loss):
            net = FNN(y0)
            optimizer = Adam(net.parameters(), lr=0.0001, weight_decay=0.01)
            break
    L_prediction.append(y0)
