import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

# -------------- PINNs --------------
def create_A(N):
    A = np.zeros((N, N))
    for i in range(1, N-1):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1
    return torch.tensor(A, dtype=torch.float32)

N = 50
T = 0.5
steps = 50 # To match dt=0.01 and 50 steps
x = np.linspace(0, 1, N)
t = np.linspace(0, T, steps)

u0 = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
A = create_A(N)
u = [torch.tensor(u0, dtype=torch.float32)]
dt = t[1] - t[0]

for i in range(steps-1):
    u_next = u[-1].clone() + dt * (A @ u[-1])
    u_next[0] = 0.0
    u_next[-1] = np.sin(2 * np.pi * t[i+1])
    u.append(u_next)

u_true = torch.stack(u)
t_tensor = torch.tensor(t, dtype=torch.float32)

class HybridODEFunc(nn.Module):
    def __init__(self, dim, A, lam=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )
        self.A = A
        self.lam = lam

    def forward(self, t, u):
        du = self.net(u) + self.lam * (self.A @ u)
        du_boundary = du.clone()
        du_boundary[..., 0] = 0.0
        du_boundary[..., -1] = 2 * np.pi * torch.cos(2 * np.pi * t)
        return du_boundary

model_hybrid = HybridODEFunc(N, A)
optimizer = torch.optim.Adam(model_hybrid.parameters(), lr=0.001)

for epoch in range(250):
    optimizer.zero_grad()
    u0_tensor = u_true[0]
    u_pred = odeint(model_hybrid, u0_tensor, t_tensor)
    loss = torch.mean((u_pred - u_true)**2)
    loss.backward()
    optimizer.step()

u_pred_hybrid = odeint(model_hybrid, u_true[0], t_tensor).detach().numpy()

# -------------- ANALYTICAL --------------
def analytical_solution(x, t, alpha, modes, N_fourier=50):
    u = np.zeros_like(x)
    for n, A_mode in modes:
        lam = alpha * (n * np.pi)**2
        u += A_mode * np.exp(-lam * t) * np.sin(n * np.pi * x)
    
    u += x * np.sin(2 * np.pi * t)
    
    omega = 2 * np.pi
    for n in range(1, N_fourier + 1):
        lam = alpha * (n * np.pi)**2
        B_n = 2 * (-1)**(n+1) / (n * np.pi)
        C_n = -omega * B_n
        I_n = (C_n / (lam**2 + omega**2)) * (lam * np.cos(omega * t) + omega * np.sin(omega * t) - lam * np.exp(-lam * t))
        u += I_n * np.sin(n * np.pi * x)
    return u

u_anal = analytical_solution(x, T, alpha=1.0, modes=[(1, 1.0), (3, 0.5)])

np.savez('torch_anal_out.npz', x=x, u_pinn=u_pred_hybrid[-1], u_anal=u_anal)
