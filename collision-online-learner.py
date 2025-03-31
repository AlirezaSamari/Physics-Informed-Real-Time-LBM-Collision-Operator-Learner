"""
Author: Alireza Samari
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#Model
class AdaptiveActivation(nn.Module):
    def __init__(self):
        super(AdaptiveActivation, self).__init__()
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        return self.tanh(self.alpha * x)

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.activation = AdaptiveActivation()
        self.fc1 = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(output_size, output_size)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc_residual = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self.fc_residual.weight)
        self.beta = nn.Parameter(torch.tensor(0.9))

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
        super().__init__()
        layers = []
        prev_layer_size = input_size
        for neurons in residual_blocks_neurons:
            residual_block = ResidualBlock(prev_layer_size, neurons)
            layers.append(residual_block)
            prev_layer_size = neurons
        self.output_layer = nn.Linear(prev_layer_size, output_size)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        output = self.output_layer(output)
        return output

model = AdaptiveResNet(input_size=9, output_size=9, residual_blocks_neurons=[64])
model = model.to(device)
# model.load_state_dict(torch.load('model.pth'))

# Setup for simulation
maxIter = 50000
nx, ny = 100, 100
cs = np.sqrt(1/3)
nulb = 0.01
tau = 0.5 + nulb/cs**2
rho0 = 5
u0 = 0.01
Re = (u0 * ny)/nulb
fin = np.zeros((9, nx, ny))

# Lattice constants
#    6   2   5
#    3   0   1
#    7   4   8
v = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
])
t = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Solid boundaries function
def solid(nx, ny):
    solid = np.zeros((nx, ny), dtype=bool)
    solid[:, 0] = True
    solid[-1, :] = True
    solid[0, :] = True
    return solid

# Macroscopic variables
def macroscopic(fin):
    rho = np.sum(fin, axis=0)
    u = np.zeros((2, nx, ny))
    for i in range(9):
        u[0, :, :] += v[i, 0] * fin[i, :, :]
        u[1, :, :] += v[i, 1] * fin[i, :, :]
    u /= rho
    return rho, u

# Equilibrium distribution
def equilibrium(rho, u):
    usqr = (3/2) * (u[0]**2 + u[1]**2)
    feq = np.zeros((9, nx, ny))
    for i in range(9):
        uv = 3 * (v[i, 0] * u[0, :, :] + v[i, 1] * u[1, :, :])
        feq[i, :, :] = rho * t[i] * (1 + uv + 0.5 * uv**2 - usqr)
    return feq

# Simulation initialization
solid = solid(nx, ny)
for i in range(9):
    fin[i, :, :] = t[i] * rho0
rho = np.ones((nx, ny)) * rho0
u = np.zeros((2, nx, ny))
x = np.arange(0, nx, 1)
y = np.arange(0, ny, 1)
X, Y = np.meshgrid(x, y)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.8)

num_training_points = 10000
epochs = 10

# Simulation & Training loop
frames_lbm = []
frames_nn = []
u_nn = np.zeros((2, nx, ny))


for time in range(maxIter + 1):
    rho, u = macroscopic(fin)
    u[0, :, -1] = u0
    feq = equilibrium(rho, u)

    # Collision step
    fout = fin - (fin - feq) / tau

    # Training
    if time % 10 == 0:
        for epoch in range(epochs):
            fin_tensor = torch.tensor(fin.copy().reshape(-1, 9), dtype=torch.float32, requires_grad=True, device=device)
            random_indices = torch.randperm(fin_tensor.size(0))[:num_training_points]
            fin_batch = fin_tensor[random_indices]

            fout_hat = model(fin_batch)
            fout_tensor = torch.tensor(fout.reshape(-1, 9), dtype=torch.float32, device=device)[random_indices]
            loss = torch.mean((fout_hat - fout_tensor)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    scheduler.step()
    # Streaming step
    for i in range(9):
        fin[i, :, :] = np.roll(np.roll(fout[i, :, :], v[i, 0], axis=0), v[i, 1], axis=1)
        fin[i, solid] = fout[opp[i], solid]
        fin[i, :, -1] = fout[opp[i], :, -1] - 6 * rho0 * t[opp[i]] * v[opp[i], 0] * u0

    # Collect frames
    if time % 10 == 0:
        u_mag = np.sqrt(u[0]**2 + u[1]**2)/u0
        frames_lbm.append(u_mag.T)

        # Forward pass
        fout_hat_plot = model(fin_tensor).view(fin.shape).detach().cpu().numpy()
        fin_hat_plot = np.zeros((9, nx, ny))
        for i in range(9):
            fin_hat_plot[i, :, :] = np.roll(np.roll(fout_hat_plot[i, :, :], v[i, 0], axis=0), v[i, 1], axis=1)
            fin_hat_plot[i, solid] = fout_hat_plot[opp[i], solid]
            fin_hat_plot[i, :, -1] = fout_hat_plot[opp[i], :, -1] - 6 * rho0 * t[opp[i]] * v[opp[i], 0] * u0
        _, u_nn = macroscopic(fin_hat_plot)
        u_nn[0, :, -1] = u0

        u_nn_mag = np.sqrt(u_nn[0]**2 + u_nn[1]**2)/u0
        frames_nn.append(u_nn_mag.T)




# Animation
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
cax1 = axes[0].imshow(np.zeros((100, 100)), cmap=cm.jet, vmin=0, vmax=1)
cax2 = axes[1].imshow(np.zeros((100, 100)), cmap=cm.jet, vmin=0, vmax=1)
axes[0].set_title("LBM")
axes[1].set_title("Collision Online-Learner")
fig.colorbar(cax1, ax=axes[0])
fig.colorbar(cax2, ax=axes[1])

def update(frame):
    cax1.set_data(frames_lbm[frame])
    cax2.set_data(frames_nn[frame])
    return cax1, cax2

ani = FuncAnimation(fig, update, frames=len(frames_lbm), interval=50, blit=True)
ani.save("lbm_nn_animation.mp4", writer="ffmpeg", fps=120)
plt.show()
