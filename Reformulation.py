# Packages
!pip install jax --upgrade
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
from scipy.integrate import solve_ivp
import scipy
from scipy.stats import norm
import timeit
vppf = np.vectorize(norm.ppf)
np.set_printoptions(suppress=True)
CUDA = torch.cuda.is_available()
mse = nn.MSELoss()

# Global parameters
nx = 3000
ny = 3001
NBATCH = 128
ITERATIONS = 10000
NUMERICAL_METHOD = 'RK45'

# Neurodynamic optimization 2008 XIA
# Model a CCO problem by an ODE system
def projection(y):
    x, u = y[:nx], y[nx:]
    xp, up = np.clip(x, a_min=0, a_max=2), np.clip(u, a_min=0, a_max=None)
    return np.concatenate([xp, up])  

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
      
class FNN(nn.Module):
    def __init__(self, z0):
        super(FNN, self).__init__()
        self.z0 = torch.tensor(z0, dtype=torch.float)
        if CUDA:
            self.z0 = self.z0.cuda()

        self.linear1 = nn.Linear(1, 500)
        self.batch_norm1 = nn.BatchNorm1d(100)

        self.linear2 = nn.Linear(300, 300)
        self.batch_norm2 = nn.BatchNorm1d(300)

        self.linear3 = nn.Linear(100, 100)
        self.batch_norm3 = nn.BatchNorm1d(100)

        self.linear4 = nn.Linear(100, 100)
        self.batch_norm4 = nn.BatchNorm1d(100)

        self.linear5 = nn.Linear(100, 100)
        self.batch_norm5 = nn.BatchNorm1d(100)

        self.linear6 = nn.Linear(500, NZ)

    def forward(self, x):
        t = x
        x = torch.tanh(self.linear1(x))
        # x = torch.tanh(self.batch_norm2(self.linear2(x)))
        # x = torch.tanh(self.batch_norm3(self.linear3(x)))
        # x = torch.tanh(self.batch_norm4(self.linear4(x)))
        # x = torch.tanh(self.batch_norm5(self.linear5(x)))
        x = self.linear6(x)
        x = self.z0 + (1 - torch.exp(-(t-0)))*x
        return x

# Reformulate a CCO problem as neural network training
class NN_NOP:
    def __init__(self, z0, time_range, P, ODE, C_epsilon):
        self.z0 = np.array(z0)
        self.time_range = time_range
        self.T = time_range[-1]
        self.P = P
        self.ODE = ODE
        self.C_epsilon = C_epsilon
        self.vODE = vmap(ODE)
        # self.numerical_method()

    def numerical_method(self):
        sol = solve_ivp(self.ODE, self.time_range, self.z0, method=NUMERICAL_METHOD)
        self.y_T_ODE = sol.y[:, -1]
        self.y_T_ODE = self.P(self.y_T_ODE)
        self.y_T_ODE = np.array(self.y_T_ODE)
        self.epsilon_ODE = self.C_epsilon(self.y_T_ODE)
    
    def NN_method(self):
        net = FNN(self.z0)
        if CUDA:
            net = net.cuda()
        optimizer = Adam(net.parameters())

        i = 0
        L_loss = []
        L_epsilon = []
        L_prediction = []
        while True:
            # ORR mechanism
            pred_curr = self.P_multiple(net)[:NZ]
            NPE_curr = self.C_epsilon(pred_curr)
            if i==0:
                NPE_best = NPE_curr
                pred_best = pred_curr
            if NPE_curr < NPE_best:
                NPE_best = NPE_curr
                pred_best = pred_curr
            L_epsilon.append(NPE_best)
            L_prediction.append(pred_best)

            # GD training
            t = np.random.uniform(0, self.T, (NBATCH, 1))
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            if CUDA:
                t = t.cuda()
            loss = self.loss_compute(t, net)
            loss.backward()
            L_loss.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

            # Monitor
            if i%10==0:
                print(f'T: {self.T}, Iteration: {i}, Loss: {loss.item() :.0f}, epsilon_current: {NPE_curr :.0f}, epsilon_best: {NPE_best :.2f}')
            i = i+1
            if i==ITERATIONS:
                break

        return L_loss, L_epsilon, L_y_T

    def loss_compute(self, t, net):
        # nn output xu, torch.tensor (NBATCH, nxu)
        y = net(t)

        # True dxu, torch.tensor (NBATCH, nxu)
        dy = self.vODE(t.cpu().detach().numpy(), y[:, :].cpu().detach().numpy())
        dy = np.array(dy)
        dy = torch.tensor(dy, dtype=torch.float)
        if CUDA:
            dy = dy.cuda()

        # Predicted pdxu, torch.tensor (NBATCH, nxu)
        def jac_fwd(t):
            return torch.func.jacfwd(net)(t)
        pdy = torch.vmap(jac_fwd)(t)
        pdy = pdy.reshape(NBATCH, NZ)

        # Compute loss, torch.tensor a float
        loss = mse(dy, pdy)
        return loss

    def P_multiple(self, net):
        T = self.T*torch.ones((2, 1), dtype=torch.float)
        if CUDA:
            T = T.cuda()
        y = net(T)[0]
        y = y.reshape((-1, )).cpu().detach().numpy()
        output = self.P(y)
        output = np.array(output)
        return output
