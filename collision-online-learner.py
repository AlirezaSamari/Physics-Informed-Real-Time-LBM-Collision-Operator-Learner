"""
Author: Alireza Samari
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveActivation(nn.Module):
    def __init__(self):
        super(AdaptiveActivation, self).__init__()
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor(0.9, device=device))

    def forward(self, x):
        return self.tanh(self.alpha * x)

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.activation = AdaptiveActivation()
        self.fc1 = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(output_size, output_size)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc_residual = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self.fc_residual.weight)
        self.beta = nn.Parameter(torch.tensor(0.9, device=device))

    def forward(self, x):
        beta = torch.clamp(self.beta, min=0.01, max=1.0)
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        residual = self.fc_residual(residual)
        x = beta * x + (1 - beta) * residual
        x = self.activation(x)
        return x

class AdaptiveResNet(nn.Module):
    def __init__(self, input_size, output_size, residual_blocks_neurons):
        super(AdaptiveResNet, self).__init__()
        layers = []
        prev_layer_size = input_size
        for neurons in residual_blocks_neurons:
            layers.append(ResidualBlock(prev_layer_size, neurons))
            prev_layer_size = neurons
        self.output_layer = nn.Linear(prev_layer_size, output_size)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        output = self.output_layer(output)
        return output

model = AdaptiveResNet(input_size=9, output_size=9, residual_blocks_neurons=[50])
model.to(device)

# Simulation Parameters

maxIter = 50000
nx, ny = 100, 100
cs = torch.sqrt(torch.tensor(1/3, device=device))
nulb = 0.01
tau = 0.5 + nulb / (cs**2)
rho0 = 5.0
u0 = 0.01
Re = (u0 * ny) / nulb
fin = torch.zeros((9, nx, ny), dtype=torch.float32, device=device)

# Lattice constants as tensors
v = torch.tensor([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
], dtype=torch.int64, device=device)

t = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
                 dtype=torch.float32, device=device)

opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64, device=device)

# Solid boundary
def create_solid(nx, ny, device):
    solid = torch.zeros((nx, ny), dtype=torch.bool, device=device)
    solid[:, 0] = True
    solid[-1, :] = True
    solid[0, :] = True
    return solid

solid_mask = create_solid(nx, ny, device)

# Macroscopic variables
def macroscopic(fin):
    rho = fin.sum(dim=0)
    u = torch.zeros((2, nx, ny), dtype=torch.float32, device=device)
    for i in range(9):
        u[0] += v[i, 0].float() * fin[i]
        u[1] += v[i, 1].float() * fin[i]
    u = u / rho
    return rho, u

# Equilibrium
def equilibrium(rho, u):
    usqr = (3/2) * (u[0]**2 + u[1]**2)
    feq = torch.zeros((9, nx, ny), dtype=torch.float32, device=device)
    for i in range(9):
        uv = 3 * (v[i, 0].float() * u[0] + v[i, 1].float() * u[1])
        feq[i] = rho * t[i] * (1 + uv + 0.5 * uv**2 - usqr)
    return feq


# Physics-Informed Loss (Boltzmann Equation)
def boltzmann_loss(fin_hat, fin_prev, feq, v, tau):
    # Advection term
    advection_term = torch.zeros_like(fin_hat)
    for i in range(9):
        advection_term[i] = (torch.roll(fin_hat[i], shifts=(v[i, 0].item(), v[i, 1].item()), dims=(0, 1)) - fin_hat[i])

    # Collision term
    collision_term = (feq - fin_hat) / tau
    

    loss = torch.mean(advection_term ** 2) + torch.mean(collision_term ** 2)
    return loss

# Initialize fin
for i in range(9):
    fin[i] = t[i] * rho0

rho = torch.ones((nx, ny), dtype=torch.float32, device=device) * rho0
u = torch.zeros((2, nx, ny), dtype=torch.float32, device=device)

# Create meshgrid
x = torch.arange(0, nx)
y = torch.arange(0, ny)
X, Y = torch.meshgrid(x, y, indexing='ij')



optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

# Simulation and Training Loop
training_step = 10
epochs = 100
for time in range(maxIter + 1):
    # Compute macroscopic variables
    rho, u = macroscopic(fin)
    u[0, :, -1] = u0
    # Compute equilibrium distribution
    feq = equilibrium(rho, u)
    fout = fin - (fin - feq) / tau
    fin_prev = fin.clone()

    # Streaming step
    for i in range(9):
        fin[i] = torch.roll(fout[i], shifts=(v[i, 0].item(), v[i, 1].item()), dims=(0, 1))
        fin[i][solid_mask] = torch.roll(fout[opp[i].item()], shifts=(0, 0), dims=(0, 1))[solid_mask]
        fin[i][:, -1] = torch.roll(fout[opp[i].item()], shifts=(0, 0), dims=(0, 1))[:, -1] \
                       - 6 * rho0 * t[opp[i].item()] * v[opp[i].item(), 0].float() * u0

    fin_tensor = fin_prev.reshape(-1, 9)
    
    if time % training_step == 0:
        for epoch in range(epochs):

            fout_hat = model(fin_tensor.reshape(-1, 9)).view_as(fin)
            fout_tensor = fout.reshape(-1, 9).detach()

            # Compute physics-informed loss
            loss = boltzmann_loss(fout_hat, fin_prev, feq, v, tau) + torch.mean((fout_hat.view(-1, 9) - fout_tensor) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()

    if time % 1000 == 0:
        print(f'Iteration = {time}, Loss = {loss.item():.6f}')
        _, u_nn = macroscopic(fout_hat)
        u_nn[0, :, -1] = u0
        u_nn_mag = torch.sqrt(u_nn[0]**2 + u_nn[1]**2) / u0
        u_mag = torch.sqrt(u[0]**2 + u[1]**2) / u0

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        cf1 = axes[0].contourf(X.cpu().numpy(), Y.cpu().numpy(),
                               u_nn_mag.cpu().detach().numpy(), cmap=cm.jet, levels=200)
        fig.colorbar(cf1, ax=axes[0], label=r'$\frac{|u|_{GNN}}{u0}$')
        axes[0].set_title('Online Collision Learner (NN)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        cf2 = axes[1].contourf(X.cpu().numpy(), Y.cpu().numpy(),
                               u_mag.cpu().numpy(), cmap=cm.jet, levels=200)
        fig.colorbar(cf2, ax=axes[1], label=r'$\frac{|u|}{u0}$')
        axes[1].set_title('LBM')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')

        plt.suptitle(f'Iteration = {time}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
