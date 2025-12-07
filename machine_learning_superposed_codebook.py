import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

J = 6   # 用户数
K = 4   # 资源数
M = 4   # 每用户码字数

base_codebook = torch.zeros(J, K, M, dtype=torch.cfloat)

base_codebook[0] = torch.tensor([
    [-0.3318+0.6262j, -0.8304+0.4252j,  0.8304-0.4252j,  0.3318-0.6262j],
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
    [ 0.7055+0.0000j, -0.3601+0.0000j,  0.3601+0.0000j, -0.7055+0.0000j],
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
], dtype=torch.cfloat)

base_codebook[1] = torch.tensor([
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
    [ 0.7055+0.0000j, -0.3601+0.0000j,  0.3601+0.0000j, -0.7055+0.0000j],
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
    [-0.3318+0.6262j, -0.8304+0.4252j,  0.8304-0.4252j,  0.3318-0.6262j],
], dtype=torch.cfloat)

base_codebook[2] = torch.tensor([
    [ 0.3601+0.0000j,  0.7055+0.0000j, -0.7055+0.0000j, -0.3601+0.0000j],
    [-0.4202-0.8350j,  0.5933+0.3548j, -0.5933-0.3548j,  0.4202+0.8350j],
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
], dtype=torch.cfloat)

base_codebook[3] = torch.tensor([
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
    [-0.3318+0.6262j, -0.8304+0.4252j,  0.8304-0.4252j,  0.3318-0.6262j],
    [-0.4202-0.8350j,  0.5933+0.3548j, -0.5933-0.3548j,  0.4202+0.8350j],
], dtype=torch.cfloat)

base_codebook[4] = torch.tensor([
    [-0.4202-0.8350j,  0.5933+0.3548j, -0.5933-0.3548j,  0.4202+0.8350j],
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
    [ 0.3601+0.0000j,  0.7055+0.0000j, -0.7055+0.0000j, -0.3601+0.0000j],
], dtype=torch.cfloat)

base_codebook[5] = torch.tensor([
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
    [-0.3318+0.6262j, -0.8304+0.4252j,  0.8304-0.4252j,  0.3318-0.6262j],
    [-0.4202-0.8350j,  0.5933+0.3548j, -0.5933-0.3548j,  0.4202+0.8350j],
    [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
], dtype=torch.cfloat)

base_codebook = base_codebook.to(device)

# 稀疏结构 mask：非零位置为 1
mask = (base_codebook.abs() > 1e-9).float()

with torch.no_grad():
    target_power = torch.zeros(J, device=device)
    for j in range(J):
        nz = mask[j] > 0
        if nz.any():
            p = (base_codebook[j][nz].real**2 +
                 base_codebook[j][nz].imag**2).mean()
            target_power[j] = p
print("Target power per user:", target_power.detach().cpu().numpy())


def build_index_table(J, M, device):
    grids = torch.meshgrid(
        *[torch.arange(M, device=device) for _ in range(J)],
        indexing='ij'
    )
    idx_table = torch.stack([g.reshape(-1) for g in grids], dim=-1)  # (M^J, J)
    return idx_table


def build_super_constellation(C, index_table):
    """
    C: (J, K, M)
    index_table: (num_points, J)，每列是该用户的符号索引 0..M-1
    返回 super_points: (num_points, K)
    """
    J, K, M = C.shape
    num_points = index_table.size(0)
    super_points = torch.zeros(num_points, K, dtype=torch.cfloat, device=C.device)

    for j in range(J):
        C_j = C[j]                     # (K, M)
        idx_j = index_table[:, j]      # (num_points,)
        selected = C_j[:, idx_j]       # (K, num_points)
        super_points += selected.T     # -> (num_points, K)

    return super_points


index_table = build_index_table(J, M, device)


def med_mpd_super(points: torch.Tensor, eps: float = 1e-8):
    """
    points: (N, K) 复数张量，对应叠加码本的 N 个 4 维点
    """
    N, K = points.shape
    x = points.view(N, K)

    # ---------- MED ----------
    diff = x.unsqueeze(1) - x.unsqueeze(0)       # (N, N, K)
    abs_diff = diff.abs()
    dist = torch.sqrt((abs_diff ** 2).sum(dim=-1) + eps)  # (N, N)

    dist = dist + torch.eye(N, device=points.device) * 1e9
    MED = dist.min()

    # ---------- MPD ----------
    nz_mask = abs_diff > eps
    factors = torch.where(nz_mask, abs_diff, torch.ones_like(abs_diff))
    prod_dist = factors.prod(dim=-1)            # (N, N)
    prod_dist = prod_dist + torch.eye(N, device=points.device) * 1e9
    MPD = prod_dist.min()

    return MED, MPD


def distance_spectrum_loss(points: torch.Tensor, beta: float = 0.5):
    """
    points: (N, K)
    L = E[ exp(-beta * d_ij^2) ], i!=j
    beta 越大越偏重小距离。
    """
    N, K = points.shape
    x = points.view(N, K)

    diff = x.unsqueeze(1) - x.unsqueeze(0)
    d2 = (diff.real**2 + diff.imag**2).sum(dim=-1)  # (N, N)

    mask_neq = ~torch.eye(N, dtype=torch.bool, device=points.device)
    d2 = d2[mask_neq]

    loss = torch.exp(-beta * d2).mean()
    return loss


with torch.no_grad():
    super0 = build_super_constellation(base_codebook, index_table)
    MED0, MPD0 = med_mpd_super(super0)
print(f"Initial super MED = {MED0.item():.6f}, MPD = {MPD0.item():.6f}")


beta = 0.5          # 距离谱 loss 参数
lambda_med = 0.02   # MED 权重
lr = 1e-2
num_steps = 1000

codebook = torch.nn.Parameter(base_codebook.clone())
optimizer = torch.optim.Adam([codebook], lr=lr)


for step in range(num_steps):

    optimizer.zero_grad()

    C = codebook * mask
    super_points = build_super_constellation(C, index_table)
    MED, MPD = med_mpd_super(super_points)
    spec_loss = distance_spectrum_loss(super_points, beta=beta)
    loss = spec_loss - lambda_med * MED

    loss.backward()
    optimizer.step()

    # 投影
    with torch.no_grad():
        codebook.data *= mask
        for j in range(J):
            nz = mask[j] > 0
            if nz.any():
                cur_p = (codebook.data[j][nz].real**2 +
                         codebook.data[j][nz].imag**2).mean()
                scale = torch.sqrt(target_power[j] / (cur_p + 1e-8))
                codebook.data[j][nz] *= scale

    if step % 50 == 0:
        print(f"step {step:4d} | loss {loss.item():.6e} | "
              f"MED {MED.item():.6f} | MPD {MPD.item():.6f} | "
              f"spec_loss {spec_loss.item():.6e}")

with torch.no_grad():
    C_final = (codebook * mask).detach().cpu()        # (J, K, M)
    super_final = build_super_constellation(codebook * mask, index_table).detach().cpu()
    MED_final, MPD_final = med_mpd_super(super_final)

print("\n===== Final super MED / MPD =====")
print(f"Final MED = {MED_final.item():.6f}, MPD = {MPD_final.item():.6f}")

print("\n===== Optimized codebook (per user) =====")
for j in range(J):
    print(f"\nUser {j+1} codebook (4x4):")
    print(C_final[j].numpy())

torch.save(C_final, "optimized_codebook_super.pt")
np.save("optimized_codebook_super.npy", C_final.numpy())
print("\nSaved to optimized_codebook_super.pt / .npy")

C_np = C_final.numpy()
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for j in range(J):
    ax = axes[j]
    user_c = C_np[j]                       # (K, M)
    nonzero = np.abs(user_c) > 1e-6
    points = user_c[nonzero]

    ax.scatter(points.real, points.imag)
    ax.set_title(f"User {j+1}")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.grid(True)
    ax.axis("equal")

plt.tight_layout()
plt.show()

super_np = super_final.numpy()
super_flat = super_np.sum(axis=1)

plt.figure(figsize=(6, 6))
plt.scatter(super_flat.real, super_flat.imag, s=4)
plt.xlabel("Real")
plt.ylabel("Imag")
plt.title("Optimized Super-Constellation (projected)")
plt.grid(True)
plt.axis("equal")
plt.show()
