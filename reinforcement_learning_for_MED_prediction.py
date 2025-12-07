import time
import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



M = 4   # 每个用户码字数
J = 6   # 用户数
K = 4   # 码字维度
N_SUP = M ** J

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

N_TRAIN_CODEBOOKS = 200     # 训练集码本数量
N_VAL_CODEBOOKS   = 20      # 验证集码本数量
N_PAIRS_PER_CB    = 512
N_EPOCHS          = 40
LR                = 1e-3
PG_LAMBDA         = 0.1

N_BEST = 5

# 最优模型保存路径
BEST_MODEL_PATH = "med_estimator_best.pt"

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

idx_all = torch.cartesian_prod(*[torch.arange(M) for _ in range(J)])  # (4096,6)
idx_all = idx_all.long()

idx_to_row = {tuple(idx_all[n].tolist()): n for n in range(N_SUP)}
print("Number of superposed codewords:", N_SUP)



def generate_random_codebooks(J, M, K):
    real = torch.randn(J, M, K, dtype=torch.float32)
    imag = torch.randn(J, M, K, dtype=torch.float32)
    return (real + 1j * imag).to(torch.complex64)


def build_superposed_codebook(codebooks, idx_all):

    J_, M_, K_ = codebooks.shape
    N = idx_all.shape[0]
    S = torch.zeros(N, K_, dtype=codebooks.dtype)
    for j in range(J_):
        S += codebooks[j, idx_all[:, j], :]
    return S


def med_exact_gram(S):

    S_dev = S.to(device)
    diag = (S_dev.conj() * S_dev).sum(dim=1).real  # (N,)
    G = S_dev @ S_dev.conj().t()                   # (N,N)
    D2 = diag.unsqueeze(1) + diag.unsqueeze(0) - 2.0 * G.real
    D2.fill_diagonal_(float("inf"))
    med2 = D2.min()
    med = float(torch.sqrt(med2).cpu())
    return med


def sample_pair_with_hamming(idx_all, idx_to_row, h):
    """
    从 superposed codebook 中随机抽一个 pair (i,j)，
    要求 Hamming(m_i, m_j) = h。
    """
    N = idx_all.shape[0]
    J_ = idx_all.shape[1]

    # 随机基索引 i
    i = random.randint(0, N - 1)
    base_pattern = idx_all[i].clone()  # (J_,)

    # 选 h 个位置
    users = random.sample(range(J_), h)

    new_pattern = base_pattern.clone()
    for u in users:
        old_sym = int(new_pattern[u].item())
        # 从 [0..M-1] 中选一个 != old_sym 的新符号
        new_sym = random.randrange(M - 1)
        if new_sym >= old_sym:
            new_sym += 1
        new_pattern[u] = new_sym

    j = idx_to_row[tuple(new_pattern.tolist())]
    return i, j


def compute_pair_features_from_h(S, pairs, h_list):
    """
    给定 S 和 pair 索引 + Hamming 权重，计算每个 pair 的特征：
      - distance
      - hamming_norm = h / J
    """
    S_dev = S.to(device)
    i = pairs[:, 0].to(device)
    j = pairs[:, 1].to(device)

    diff = S_dev[i] - S_dev[j]  # (P,K)
    d2 = (diff.conj() * diff).sum(dim=1).real
    d = torch.sqrt(d2 + 1e-9)   # (P,)

    h_tensor = torch.tensor(h_list, dtype=torch.float32, device=device)
    h_norm = h_tensor / float(J)

    pair_feats = torch.stack([d, h_norm], dim=1)  # (P,2)
    return pair_feats


def compute_global_features(S):
    """
    原始全局特征：每个超码字的范数统计
    返回: (4,) float32 on device
    """
    S_dev = S.to(device)
    norms = torch.sqrt((S_dev.conj() * S_dev).sum(dim=1).real)  # (N_sup,)
    mean = norms.mean()
    std = norms.std()
    minv = norms.min()
    maxv = norms.max()
    g = torch.stack([mean, std, minv, maxv]).float()
    return g  # on device


def create_dataset(n_codebooks):
    """
    生成 n_codebooks 个 (S, med_exact) 样本
    """
    data = []
    for t in range(n_codebooks):
        codebooks = generate_random_codebooks(J, M, K)
        S = build_superposed_codebook(codebooks, idx_all)
        d_exact = med_exact_gram(S)
        data.append({
            "S": S,
            "med_exact": d_exact,
        })
        if (t + 1) % 10 == 0:
            print(f"  Generated {t+1}/{n_codebooks} codebooks...")
    return data


print("\nCreating training dataset...")
train_data = create_dataset(N_TRAIN_CODEBOOKS)
print("\nCreating validation dataset...")
val_data = create_dataset(N_VAL_CODEBOOKS)



class PolicyNet(nn.Module):

    def __init__(self, global_feat_dim=4, hidden=32, n_categories=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_categories),
        )

    def forward(self, global_feats):
        logits = self.mlp(global_feats)
        return logits


class RegressorNet(nn.Module):

    def __init__(self, pair_feat_dim=2, global_feat_dim=4,
                 hidden_global=32, hidden_pair=64, hidden_out=64):
        super().__init__()

        self.global_mlp = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_global),
            nn.ReLU(),
            nn.Linear(hidden_global, hidden_global),
            nn.ReLU(),
        )

        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim + hidden_global, hidden_pair),
            nn.ReLU(),
            nn.Linear(hidden_pair, hidden_pair),
            nn.ReLU(),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_pair + hidden_global, hidden_out),
            nn.ReLU(),
            nn.Linear(hidden_out, 1),
        )

    def forward(self, pair_feats, global_feats):
        P = pair_feats.size(0)
        g = self.global_mlp(global_feats)              # (hg,)
        g_expanded = g.unsqueeze(0).expand(P, -1)      # (P,hg)

        x = torch.cat([pair_feats, g_expanded], dim=1) # (P, 2+hg)
        pair_emb = self.pair_mlp(x)                    # (P,hp)

        z_pairs = pair_emb.sum(dim=0)                  # (hp,)
        z = torch.cat([z_pairs, g], dim=0)             # (hp+hg,)

        out = self.out_mlp(z)                          # (1,)
        return out.squeeze(-1)                         # 标量


policy_net = PolicyNet().to(device)
regressor_net = RegressorNet().to(device)
optimizer = optim.Adam(list(policy_net.parameters()) +
                       list(regressor_net.parameters()), lr=LR)

HAMMING_VALUES = [2, 3, 4]


def run_epoch(data, train_mode=True):
    """
    对 data 中的所有码本跑一轮
    返回: (avg_loss, avg_rel_err)
    """
    if train_mode:
        policy_net.train()
        regressor_net.train()
    else:
        policy_net.eval()
        regressor_net.eval()

    total_loss = 0.0
    total_rel_err = 0.0

    for item in data:
        S = item["S"]
        d_exact = item["med_exact"]

        global_feats = compute_global_features(S)  # (4,)
        global_feats = global_feats.to(device)

        logits = policy_net(global_feats)          # (3,)
        cat_dist = torch.distributions.Categorical(logits=logits)

        pair_i = []
        pair_j = []
        h_list = []
        log_probs = []

        with torch.set_grad_enabled(train_mode):
            for _ in range(N_PAIRS_PER_CB):
                cat = cat_dist.sample()             # 0,1,2
                log_prob = cat_dist.log_prob(cat)
                h = HAMMING_VALUES[int(cat.item())]

                i, j = sample_pair_with_hamming(idx_all, idx_to_row, h)
                pair_i.append(i)
                pair_j.append(j)
                h_list.append(h)
                log_probs.append(log_prob)

        pairs = torch.tensor(np.stack([pair_i, pair_j], axis=1),
                             dtype=torch.long)      # (P,2), CPU
        pair_feats = compute_pair_features_from_h(S, pairs, h_list)  # (P,2) on device

        with torch.set_grad_enabled(train_mode):
            pred = regressor_net(pair_feats, global_feats)
            d_exact_t = torch.tensor(d_exact, dtype=torch.float32,
                                     device=device)
            mse_loss = (pred - d_exact_t) ** 2

        if train_mode:
            log_probs_tensor = torch.stack(log_probs)  # (P,)
            log_prob_mean = log_probs_tensor.mean()

            # reward = -MSE, policy_loss = MSE_detach * log_prob_mean
            policy_loss = mse_loss.detach() * log_prob_mean
            total = mse_loss + PG_LAMBDA * policy_loss

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            loss_val = float(total.detach().cpu())
        else:
            loss_val = float(mse_loss.detach().cpu())

        pred_val = float(pred.detach().cpu())
        rel_err = abs(pred_val - d_exact) / (d_exact + 1e-12)

        total_loss += loss_val
        total_rel_err += rel_err

    n = len(data)
    return total_loss / n, total_rel_err / n



print("\nStart training with learned sampling (REINFORCE)...\n")

best_val_rel = float("inf")
best_state = None

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.perf_counter()
    train_loss, train_rel = run_epoch(train_data, train_mode=True)
    val_loss, val_rel = run_epoch(val_data, train_mode=False)
    t1 = time.perf_counter()

    print(f"Epoch {epoch:02d} | "
          f"Train Loss={train_loss:.4e}, RelErr={train_rel:.4f} | "
          f"Val Loss={val_loss:.4e}, RelErr={val_rel:.4f} | "
          f"time={t1-t0:.3f}s")

    if val_rel < best_val_rel:
        best_val_rel = val_rel
        best_state = {
            "epoch": epoch,
            "val_rel": val_rel,
            "policy_net_state": copy.deepcopy(policy_net.state_dict()),
            "regressor_net_state": copy.deepcopy(regressor_net.state_dict()),
            "config": {
                "M": M,
                "J": J,
                "K": K,
                "N_PAIRS_PER_CB": N_PAIRS_PER_CB,
            },
        }

print("\nTraining finished.\n")

# 保存最优模型
if best_state is not None:
    torch.save(best_state, BEST_MODEL_PATH)
    print(f"Best model saved to '{BEST_MODEL_PATH}' "
          f"(epoch={best_state['epoch']}, val_rel={best_state['val_rel']:.4f})")
else:
    print("Warning: best_state is None, no model saved.")


print(f"\nFinding best {N_BEST} validation examples (smallest relative error)...")

policy_net.eval()
regressor_net.eval()

val_results = []

for idx, item in enumerate(val_data):
    S = item["S"]
    d_exact = item["med_exact"]

    global_feats = compute_global_features(S).to(device)
    logits = policy_net(global_feats)
    cat_dist = torch.distributions.Categorical(logits=logits)

    pair_i = []
    pair_j = []
    h_list = []
    for _ in range(N_PAIRS_PER_CB):
        cat = cat_dist.sample()
        h = HAMMING_VALUES[int(cat.item())]
        i, j = sample_pair_with_hamming(idx_all, idx_to_row, h)
        pair_i.append(i)
        pair_j.append(j)
        h_list.append(h)

    pairs = torch.tensor(np.stack([pair_i, pair_j], axis=1),
                         dtype=torch.long)
    pair_feats = compute_pair_features_from_h(S, pairs, h_list)

    with torch.no_grad():
        pred = regressor_net(pair_feats, global_feats)

    pred_val = float(pred.cpu())
    rel_err = abs(pred_val - d_exact) / (d_exact + 1e-12)
    val_results.append((idx, d_exact, pred_val, rel_err))

val_results.sort(key=lambda x: x[3])

print(f"\nBest {N_BEST} validation results:")
for k in range(min(N_BEST, len(val_results))):
    idx, d_exact, pred_val, rel_err = val_results[k]
    print(
        f"  Rank {k+1}: val_idx={idx}, "
        f"exact MED = {d_exact:.4f}, "
        f"predicted = {pred_val:.4f}, "
        f"rel_err = {rel_err:.4f}"
    )
