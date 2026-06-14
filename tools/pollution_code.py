import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import h5py
import os
from PIL import Image
# ------------------------
# Heaviside function
# ------------------------
def heaviside(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(x.dtype)

# ------------------------
# Upwind iteration for one timestep (with inner iteration)
# ------------------------
def upwind_step_iterative(C, u, v, w, alpha, dx, dy, dz, dt, S, 
                          max_iter=20, conv_tol=1e-6):
    """
    Conservative 3D upwind (implicit, Jacobi-like) one-step update:
      b*C^{m+1} = (b - a_center)*C^{m} 
                  - a_im1*C_{i-1}^{m} - a_ip1*C_{i+1}^{m}
                  - a_jm1*C_{j-1}^{m} - a_jp1*C_{j+1}^{m}
                  - a_km1*C_{k-1}^{m} - a_kp1*C_{k+1}^{m}
                  + (1/dt)*C^n + S
    """
    C_new = C.clone()

    # ---------- face velocities with building blocking ----------
    # dims: (z, y, x) -> roll by +1 means take neighbor on -dim side
    u_face = 0.5 * (torch.roll(u, 1, 2) + u) * (1 - heaviside(torch.maximum(torch.roll(alpha,1,2), alpha)))
    v_face = 0.5 * (torch.roll(v, 1, 1) + v) * (1 - heaviside(torch.maximum(torch.roll(alpha,1,1), alpha)))
    w_face = 0.5 * (torch.roll(w, 1, 0) + w) * (1 - heaviside(torch.maximum(torch.roll(alpha,1,0), alpha)))

    # i-1/2, i+1/2; j-1/2, j+1/2; k-1/2, k+1/2
    u_imh, u_iph = u_face, torch.roll(u_face, -1, 2)
    v_jmh, v_jph = v_face, torch.roll(v_face, -1, 1)
    w_kmh, w_kph = w_face, torch.roll(w_face, -1, 0)

    Hu_imh, Hu_iph = heaviside(u_imh), heaviside(u_iph)
    Hv_jmh, Hv_jph = heaviside(v_jmh), heaviside(v_jph)
    Hw_kmh, Hw_kph = heaviside(w_kmh), heaviside(w_kph)

    # ---------- neighbor coefficients (upwind inflow) ----------
    a_im1 = - Hu_imh * u_imh / dx
    a_ip1 =   (1 - Hu_iph) * u_iph / dx
    a_jm1 = - Hv_jmh * v_jmh / dy
    a_jp1 =   (1 - Hv_jph) * v_jph / dy
    a_km1 = - Hw_kmh * w_kmh / dz
    a_kp1 =   (1 - Hw_kph) * w_kph / dz

    # ---------- center coefficient a_{i,j,k} (from slide) ----------
    a_center = (1.0/dt
                + Hu_iph * (u_iph/dx) - (1 - Hu_imh) * (u_imh/dx)
                + Hv_jph * (v_jph/dy) - (1 - Hv_jmh) * (v_jmh/dy)
                + Hw_kph * (w_kph/dz) - (1 - Hw_kmh) * (w_kmh/dz))

    # ---------- diagonal b (stabilizing, diagonal dominance) ----------
    b = (1.0/dt
            + torch.maximum(torch.abs(u_imh), torch.abs(u_iph))/dx
            + torch.maximum(torch.abs(v_jmh), torch.abs(v_jph))/dy
            + torch.maximum(torch.abs(w_kmh), torch.abs(w_kph))/dz)

    for it in range(max_iter):
        C_old = C_new.clone()
        # ---------- Jacobi-style update ----------
        rhs0 = (1.0/dt) * C + S
        rhs  = (rhs0
                + (b - a_center) * C_old
                - a_im1 * torch.roll(C_old,  1, 2)
                - a_ip1 * torch.roll(C_old, -1, 2)
                - a_jm1 * torch.roll(C_old,  1, 1)
                - a_jp1 * torch.roll(C_old, -1, 1)
                - a_km1 * torch.roll(C_old,  1, 0)
                - a_kp1 * torch.roll(C_old, -1, 0)
               )
        C_new = rhs / b
        # ---------- (optional) residual for early stop ----------
        denom = torch.maximum(C_old.abs().max(), torch.tensor(1e-12, device=C.device))
        res = (C_new - C_old).abs().max() / denom
        if res < conv_tol:
            print(it)
            break

    return C_new


# ------------------------
# Visualization helper
# ------------------------
def plot_slice(C, step, dt):
    nz = 5
    os.makedirs('result', exist_ok=True)
    # 提取切片并归一化到 0~255
    slice_data = C[nz, :, :].cpu().numpy()
    slice_data = np.clip(slice_data,0,3)
    slice_uint8 = (slice_data * 255/3).astype(np.uint8)
    img = Image.fromarray(slice_uint8).convert("L").resize((slice_uint8.shape[1], slice_uint8.shape[0]))
    img.save(f"result/{step:04d}.png")

def plot_slice_h(C, step, dt):
    os.makedirs('result_h', exist_ok=True)
    ny = 512
    slice_data = C[:, ny, :].cpu().numpy()
    # slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
    slice_data = np.clip(slice_data,0,3)
    slice_uint8 = (slice_data * 255/3).astype(np.uint8)
    img = Image.fromarray(slice_uint8).convert("L").resize((slice_uint8.shape[1], slice_uint8.shape[0]))
    img.save(f"result_h/{step:04d}.png")

def plot_slice_u(C, step, dt):
    # nz = C.shape[0] // 2
    nz = 4
    os.makedirs('result_u',exist_ok=True)
    # plt.figure(figsize=(5,5))
    plt.imshow(C[nz,:,:].cpu().numpy(), origin="lower", aspect="auto",
               cmap="jet", vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"z=mid slice, step={step}, t={step*dt:.2f}")
    plt.savefig(f"result_u/{step:04d}.png")
    plt.close()

# ------------------------
# Run a simple test
# ------------------------
nx, ny, nz = 1024, 1024, 64
dx = dy = dz = 1.0
dt = 0.5

device = "cuda"  # or "cuda" if GPU available

# initial condition: Gaussian blob
x = torch.arange(nx, device=device)
y = torch.arange(ny, device=device)
z = torch.arange(nz, device=device)
Z,Y,X = torch.meshgrid(z,y,x, indexing="ij")
S = torch.exp(-((X-20.0)**2 + (Y-ny/2)**2 + (Z-nz/2)**2)/100.0)
S = torch.zeros((nz,ny,nx), device=device)
S[2:8, ny//2-300:ny//2+300, 50-10:50+10] = 1.0
C0 = S      # no source

C = C0.clone()

sigma = torch.from_numpy(np.load('data/SCALED_dataset/south_kensington/sigma.npy')[0,0]).to(device)/1e8

os.makedirs('result',exist_ok=True)
# time loop
nsteps = 4000
for step in tqdm(range(nsteps)):
    filename = f'data/SCALED_dataset/south_kensington/{step:06d}.h5'
    with h5py.File(filename, 'r') as f:
        data = f['uvw']
        data = np.array(data)
    data = torch.from_numpy(data).to(torch.float32).to(device)
    u = -data[0]
    v = data[1]
    w = data[2]
    C = upwind_step_iterative(C, u, v, w, sigma, dx, dy, dz, dt, S*0.1, max_iter=200, conv_tol=1e-7)
    C[sigma.bool()] = 0
    C_save = C.clone().cpu().numpy()
    new_filename = f'data/SCALED_dataset/south_kensington_conservation/{step:06d}.h5'
    with h5py.File(new_filename, "w") as f_out:
        f_out.create_dataset("C", data=C_save, compression="gzip")