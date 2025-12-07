import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import math
import random
import torch


M, dv = 4, 2; DELTA, E, V = 120, 12000, 12
RHO_START, RHO_END_FRAC = int(0.30*DELTA), 0.15  # ρ 线性降
EPSILON_MAX = 0.5
UCB_C = 2.0
P_POWER = 2.0
TOP_K_PATHS = 5            # 每轮细化使用的最优路径数（多锚点）
PRUNE_K_BEST = None        # 探索阶段只在 UCB Top-K 中选动作
USE_AMP = True             # 奖励计算阶段使用 AMP
SEED = 4232                  # 随机种子

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

B = 2 * M * dv
step_positions = []
for j in range(M):
    for k in range(dv):
        step_positions.append((j, k, 'real'))
        step_positions.append((j, k, 'imag'))

# 初始化
def init_action_sets():
    return [torch.linspace(-1.0, 1.0, DELTA+1, device=device) for _ in range(B)]

action_sets = init_action_sets()

def rho_for_round(v_idx: int) -> int:
    #第 v 轮的 ρ：线性衰减到 rho_min后不再降
    rho_min = int(RHO_END_FRAC * DELTA)
    if V <= 1:
        return max(rho_min, RHO_START)
    rho = int(RHO_START + (rho_min - RHO_START) * (v_idx - 1) / (V - 1))
    return max(rho, rho_min)

def compute_reward(codebook_real: torch.Tensor, codebook_imag: torch.Tensor):
    #归一化
    total_power = (codebook_real**2 + codebook_imag**2).sum()
    if total_power > 0:
        scale = math.sqrt(M * P_POWER / float(total_power))
        codebook_real = codebook_real * scale
        codebook_imag = codebook_imag * scale

    #向量化计算d_p^2
    with torch.amp.autocast('cuda', enabled=USE_AMP and torch.cuda.is_available()):
        # ri, ii: (dv, M)
        ri = codebook_real
        ii = codebook_imag
        # (dv, M, 1) -(dv, 1, M)= (D, M, M)
        diff_r = ri.unsqueeze(2) - ri.unsqueeze(1)
        diff_i = ii.unsqueeze(2) - ii.unsqueeze(1)
        #模平方
        diff2 = (diff_r * diff_r + diff_i * diff_i).float()
        # 沿 dv 相乘
        pd_mm = diff2.prod(dim=0)

    # 取上三角 i<j
    iu = torch.triu_indices(M, M, offset=1)
    pd = pd_mm[iu[0], iu[1]]
     # 若存在极差码本
    if torch.any(pd <= 0):
        return 0.0, 0.0

    inv_pd = 1.0 / pd
    Psi = inv_pd.mean().item()
    R = 1.0 / Psi if Psi > 0 else 0.0

    MPD = pd.min().item()
    return R, MPD


def pick_action(step, state, epsilon, Q, state_count, state_action_count):

    actions = action_sets[step]
    S = state
    state_count[S] = state_count.get(S, 0) + 1

    # exploit
    if random.random() >= epsilon:
        best_Q = -float('inf')
        cands = []
        for ai in range(len(actions)):
            val = Q.get((S, ai), 0.0)
            if val > best_Q:
                best_Q = val; cands = [ai]
            elif val == best_Q:
                cands.append(ai)
        return random.choice(cands)

    # explore (UCB)
    # 未试过的动作优先
    untried = [ai for ai in range(len(actions)) if state_action_count.get((S, ai), 0) == 0]
    if untried:
        return random.choice(untried)

    total_visits = state_count[S]

    if PRUNE_K_BEST is not None and PRUNE_K_BEST < len(actions):
        ucb_vals = []
        #UCB
        for ai in range(len(actions)):
            Qa = Q.get((S, ai), 0.0)
            Na = state_action_count.get((S, ai), 0)
            ucb = Qa + UCB_C * math.sqrt(max(1e-12, math.log(total_visits) / max(1, Na)))
            ucb_vals.append((ucb, ai))
        ucb_vals.sort(reverse=True, key=lambda x: x[0])
        pool = [ai for _, ai in ucb_vals[:PRUNE_K_BEST]]
    else:
        pool = list(range(len(actions)))

    best_ucb = -float('inf')
    bests = []
    for ai in pool:
        Qa = Q.get((S, ai), 0.0)
        Na = state_action_count.get((S, ai), 0)
        ucb = Qa + UCB_C * math.sqrt(max(1e-12, math.log(total_visits) / max(1, Na)))
        if ucb > best_ucb:
            best_ucb = ucb; bests = [ai]
        elif ucb == best_ucb:
            bests.append(ai)
    return random.choice(bests)

# 训练状态
best_reward_overall = -float('inf')
best_mpd_overall = 0.0
best_codebook_real = None
best_codebook_imag = None

for v in range(1, V+1):
    rho = rho_for_round(v)
    print(f"\nRound {v}/{V}")

    Q = {}
    state_count = {}
    state_action_count = {}
    epsilon = 0.0
    eps_step = 1.0 / E

    # 本轮最优们
    top_paths = []  # 列表元素：(R, MPD, actions_list)

    for episode in range(1, E+1):
        # 初始化空码本
        creal = torch.zeros(dv, M, device=device)
        cimag = torch.zeros(dv, M, device=device)
        state = tuple()
        path = []

        for step in range(B):
            ai = pick_action(step, state, epsilon, Q, state_count, state_action_count)
            state_action_count[(state, ai)] = state_action_count.get((state, ai), 0) + 1
            path.append((state, ai))

            j, k, part = step_positions[step]
            val = action_sets[step][ai]
            if part == 'real':
                creal[k, j] = val
            else:
                cimag[k, j] = val

            state = state + (ai,)

        # 计算奖励/MPD（内部会做归一化）
        R, MPD = compute_reward(creal, cimag)

        # 更新全局最优
        if R > best_reward_overall:
            best_reward_overall = R
            best_mpd_overall = MPD
            best_codebook_real = creal.detach().cpu()
            best_codebook_imag = cimag.detach().cpu()

        # 本轮 Q 表回溯更新
        for st, ai in path:
            prev = Q.get((st, ai), 0.0)
            if R > prev:
                Q[(st, ai)] = R

        # 维护本轮 Top-K 最优路径（按 R 排，若 R 相同按 MPD）
        acts = [ai for (_, ai) in path]
        top_paths.append((R, MPD, acts))
        top_paths.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if len(top_paths) > TOP_K_PATHS:
            top_paths = top_paths[:TOP_K_PATHS]

        # GLIE：逐步增 ε
        epsilon = min(EPSILON_MAX, epsilon + eps_step)

        if episode % max(1, E//5) == 0:
            print(f"  best_R@round={top_paths[0][0]:.5f} | best_MPD@round={top_paths[0][1]:.5f} ", end="\r")

    # 细化动作集合
    if v < V:
        new_sets = []
        tau = int(0.10 * DELTA)  # 边缘阈值
        for step in range(B):
            actions = action_sets[step]
            # 收集本轮 Top-K 最优路径在该步的动作索引
            anchor_idx = [acts[step] for (_, _, acts) in top_paths if len(acts) == B]
            if not anchor_idx:
                # 极少数情况下未收集到，有则回退为全域
                new_sets.append(torch.linspace(-1.0, 1.0, DELTA+1, device=device))
                continue

            # 合并[L,R]
            Ls, Rs = [], []
            force_left, force_right = False, False
            for idx in anchor_idx:
                half = max(1, rho // 2)
                L = max(0, idx - half)
                R = min(DELTA, idx + half)
                Ls.append(L); Rs.append(R)
                # 边缘保护
                if idx < tau: force_left = True
                if idx > (DELTA - tau): force_right = True

            L_global = min(Ls); R_global = max(Rs)
            left_val = float(actions[L_global].item())
            right_val = float(actions[R_global].item())

            # 强制纳入区间端点（-1, +1）
            if force_left: left_val = -1.0
            if force_right: right_val =  1.0
            if right_val < left_val:  # 罕见数值异常保护
                left_val, right_val = min(left_val, right_val), max(left_val, right_val)

            new_sets.append(torch.linspace(left_val, right_val, DELTA+1, device=device))

        action_sets = new_sets
        print(f"\nRound {v} done.")

print("\n========== Training Finished ==========")
print(f"Global Best Reward: {best_reward_overall:.6f}")
print(f"Global Best MPD   : {best_mpd_overall:.6f}")
if best_codebook_real is not None:
    Cc = best_codebook_real + 1j * best_codebook_imag
    print("Best codebook (dv x M complex vectors):")
    for j in range(M):
        point = [complex(Cc[k, j]) for k in range(dv)]
        print(f"  Codeword {j+1}: {point}")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    if best_codebook_real is None or best_codebook_imag is None:
        print("error")
    else:
        C = best_codebook_real.numpy() + 1j * best_codebook_imag.numpy()
        if dv < 2:
            print("db < 2", dv)
        else:
            d1 = C[0, :]
            d2 = C[1, :]
            plt.figure(figsize=(6, 6))
            plt.scatter(np.real(d1), np.imag(d1), marker='o', label='Dimension 1')
            plt.scatter(np.real(d2), np.imag(d2), marker='o', facecolors='none', label='Dimension 2')

            for j in range(M):
                plt.annotate(str(j+1), (np.real(d1[j]), np.imag(d1[j])),
                             xytext=(3, 3), textcoords='offset points', fontsize=9)
                plt.annotate(str(j+1), (np.real(d2[j]), np.imag(d2[j])),
                             xytext=(3, 3), textcoords='offset points', fontsize=9)

            plt.axhline(0, linewidth=0.8)
            plt.axvline(0, linewidth=0.8)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True, linewidth=0.6, alpha=0.5)
            plt.xlabel('In-Phase (Real)')
            plt.ylabel('Quadrature (Imag)')
            plt.title(f'SCMA Mother Codebook (M={M}, dv={dv})')
            plt.legend(loc='best')
            plt.tight_layout()

            out_path = 'codebook_constellation.png'
            plt.savefig(out_path, dpi=200)
            print(f"已保存星座图：{out_path}")
            plt.show()
except Exception as e:
    print("error")