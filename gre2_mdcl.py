import math
import argparse
import random
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cosine_sim(a: Tensor, b: Tensor):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()  # [N, N]

def info_nce_logits(anchors: Tensor, samples: Tensor, tau: float):
    # anchors: [N, d], samples: [N, d] (positives aligned by index), negatives are the rest rows of samples
    logits = cosine_sim(anchors, samples) / tau  # [N,N]
    labels = torch.arange(anchors.size(0), device=anchors.device)
    return logits, labels

def supervised_contrastive_loss(anchor: Tensor, pos_idx: List[List[int]], bank: Tensor, tau: float):
    """
    Multi-positive NT-Xent: for each i, positives are pos_idx[i] (indices into bank).
    anchor: [N, d], bank: [M, d] (typically M=2N for two views or >N for multi-head)
    """
    sims = cosine_sim(anchor, bank) / tau  # [N, M]
    N = anchor.size(0)
    losses = []
    for i in range(N):
        pos = pos_idx[i]
        if len(pos) == 0:
            continue
        # log( sum_pos exp(sim) / sum_all_not_self exp(sim) )
        numerator = torch.logsumexp(sims[i, pos], dim=0)
        mask_all = torch.ones(bank.size(0), device=anchor.device, dtype=torch.bool)
        # exclude anchor itself if anchor is a row in bank (often not the case); keep generic:
        # here we cannot guarantee index alignment; so just use all as denominator
        denominator = torch.logsumexp(sims[i, :], dim=0)
        losses.append(-(numerator - denominator))
    if len(losses) == 0:
        return torch.tensor(0.0, device=anchor.device)
    return torch.stack(losses).mean()

# ----------------------------
# LAGNN-style CVAE for local augmentation
# ----------------------------
class CVAE(nn.Module):
    """Conditional VAE: q(z|Xu,Xv), p(z|Xv), p(Xu|Xv,z).
    Trained on neighbor pairs (v,u in N(v)).
    """
    def __init__(self, in_dim: int, z_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim*2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        self.prior = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.prior_mu = nn.Linear(hidden, z_dim)
        self.prior_logvar = nn.Linear(hidden, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(in_dim + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim)
        )

    def encode(self, Xu: Tensor, Xv: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.enc(torch.cat([Xu, Xv], dim=-1))
        mu, logvar = self.enc_mu(h), self.enc_logvar(h)
        return mu, logvar

    def prior_stats(self, Xv: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.prior(Xv)
        mu, logvar = self.prior_mu(h), self.prior_logvar(h)
        return mu, logvar

    def reparam(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, Xv: Tensor, z: Tensor) -> Tensor:
        return self.dec(torch.cat([Xv, z], dim=-1))

    def kld(self, mu_q, logvar_q, mu_p, logvar_p):
        # KL(q||p) for diagonal Gaussians
        return 0.5 * torch.sum(
            logvar_p - logvar_q +
            (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p) - 1,
            dim=-1
        )

    def forward(self, Xu, Xv):
        mu_q, logvar_q = self.encode(Xu, Xv)
        mu_p, logvar_p = self.prior_stats(Xv)
        z = self.reparam(mu_q, logvar_q)
        Xu_hat = self.decode(Xv, z)
        recon = F.mse_loss(Xu_hat, Xu, reduction='none').sum(dim=-1)
        kl = self.kld(mu_q, logvar_q, mu_p, logvar_p)
        return recon.mean() + kl.mean()

    @torch.no_grad()
    def generate(self, Xv: Tensor) -> Tensor:
        # sample z ~ p(z|Xv) and decode to produce neighbor-consistent vector (as "augmentation residual")
        mu_p, logvar_p = self.prior_stats(Xv)
        z = self.reparam(mu_p, logvar_p)
        Xu_hat = self.decode(Xv, z)
        return Xu_hat

class LocalAugmentor:
    def __init__(self, in_dim: int, z_dim: int = 64, hidden: int = 256, mix: float = 0.5, device='cpu'):
        self.cvae = CVAE(in_dim, z_dim, hidden).to(device)
        self.mix = mix
        self.device = device

    def _sample_neighbor_pairs(self, X: Tensor, edge_index: Tensor, max_pairs: int = 200000):
        # edge_index: [2, E]
        row, col = edge_index
        Xu = X[col]  # neighbor features
        Xv = X[row]  # center features
        # shuffle and cap to max_pairs for efficiency
        n = Xu.size(0)
        idx = torch.randperm(n, device=X.device)
        idx = idx[:min(n, max_pairs)]
        return Xu[idx], Xv[idx]

    def pretrain(self, X: Tensor, edge_index: Tensor, steps: int = 200, batch: int = 4096, lr: float = 1e-3):
        opt = torch.optim.Adam(self.cvae.parameters(), lr=lr)
        Xu_all, Xv_all = self._sample_neighbor_pairs(X, edge_index)
        N = Xu_all.size(0)
        for t in range(steps):
            for i in range(0, N, batch):
                Xu = Xu_all[i:i+batch]
                Xv = Xv_all[i:i+batch]
                loss = self.cvae(Xu, Xv)
                opt.zero_grad()
                loss.backward()
                opt.step()
        # freeze after pretrain
        for p in self.cvae.parameters():
            p.requires_grad = False
        self.cvae.eval()

    @torch.no_grad()
    def apply(self, X: Tensor) -> Tensor:
        # Generate neighbor-consistent vector g(Xv), then mix with original X
        gen = self.cvae.generate(X)
        # simple residual-style blend
        out = (1 - self.mix) * X + self.mix * gen
        return out

# ----------------------------
# SVD Global Augmentor
# ----------------------------
class SVDGlobalAugmentor:
    """
    If mode='features': do SVD on feature matrix F and keep top-k singular components to get F_k.
    If mode='adjacency': do SVD on dense adjacency A to get low-rank A_k (not default).
    """
    def __init__(self, k: int = 64, mode: str = 'features'):
        self.k = k
        assert mode in ['features', 'adjacency']
        self.mode = mode

    @torch.no_grad()
    def apply(self, X: Tensor, edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
        if self.mode == 'features':
            # X: [N, d]
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # X = U S Vh
            k = min(self.k, S.size(0))
            Xk = (U[:, :k] * S[:k]) @ Vh[:k, :]
            return Xk, edge_index
        else:
            A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)  # [N, N]
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            k = min(self.k, S.size(0))
            Ak = (U[:, :k] * S[:k]) @ Vh[:k, :]
            Ak = (Ak + Ak.t()) / 2  # symmetrize
            # keep original sparsity pattern (optional); here we keep edges with top-k per row (simple)
            thr = torch.topk(Ak, k=min(64, Ak.size(1)), dim=1).values[:, -1].unsqueeze(1)
            mask = Ak >= thr
            rows, cols = mask.nonzero(as_tuple=True)
            new_edge_index = torch.stack([rows, cols], dim=0).to(edge_index.device)
            return X, new_edge_index

# ----------------------------
# GNN encoders (multi-head GAT) + projector/predictor
# ----------------------------
class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, heads: int = 4,
                 dropout_in: float = 0.0, attn_drop: float = 0.0, num_layers: int = 2):
        super().__init__()
        self.dropout_in = dropout_in
        self.convs = nn.ModuleList()
        self.heads = heads
        h_each = hidden // heads * heads  # ensure divisible
        if num_layers == 1:
            self.convs.append(GATConv(in_dim, out_dim // heads, heads=heads, dropout=attn_drop, concat=True))
        else:
            self.convs.append(GATConv(in_dim, h_each // heads, heads=heads, dropout=attn_drop, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(h_each, h_each // heads, heads=heads, dropout=attn_drop, concat=True))
            self.convs.append(GATConv(h_each, out_dim // heads, heads=heads, dropout=attn_drop, concat=True))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        head_slices = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)  # concat across heads => [N, H*heads]
            if i < len(self.convs) - 1:
                x = F.elu(x)
            # split per head for neighbor contrast later
        # split along channel dimension into heads
        per_head = torch.chunk(x, chunks=self.convs[-1].heads, dim=-1)
        return x, list(per_head)  # x: [N, out], per_head: list of [N, out/heads]

class Projector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, proj_dim: int):
        super().__init__()
        hid = max(2*proj_dim, 128)
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(inplace=True),
            nn.Linear(hid, proj_dim)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Triple-network wrapper
# ----------------------------
class TripleGNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, proj_dim, heads=4, feat_drop=0.2, attn_drop=0.4, layers=2):
        super().__init__()
        # online
        self.online_enc = MultiHeadGAT(in_dim, hidden, out_dim, heads, feat_drop, attn_drop, layers)
        self.online_proj = Projector(out_dim, proj_dim)
        self.online_pred = Predictor(proj_dim)
        # targets (no predictor)
        self.t1_enc = MultiHeadGAT(in_dim, hidden, out_dim, heads, feat_drop, attn_drop, layers)
        self.t1_proj = Projector(out_dim, proj_dim)
        self.t2_enc = MultiHeadGAT(in_dim, hidden, out_dim, heads, feat_drop, attn_drop, layers)
        self.t2_proj = Projector(out_dim, proj_dim)

        # init target = online
        self._copy_params(self.t1_enc, self.online_enc)
        self._copy_params(self.t1_proj, self.online_proj)
        self._copy_params(self.t2_enc, self.online_enc)
        self._copy_params(self.t2_proj, self.online_proj)

        # EMA momentum
        self.m1 = 0.996
        self.m2 = 0.992

        # turn off grads for target
        for p in list(self.t1_enc.parameters()) + list(self.t1_proj.parameters()) + \
                 list(self.t2_enc.parameters()) + list(self.t2_proj.parameters()):
            p.requires_grad = False

    @torch.no_grad()
    def _copy_params(self, tgt: nn.Module, src: nn.Module):
        for p_t, p_s in zip(tgt.parameters(), src.parameters()):
            p_t.data.copy_(p_s.data)

    @torch.no_grad()
    def _ema_update(self):
        with torch.no_grad():
            for t, s in zip(self.t1_enc.parameters(), self.online_enc.parameters()):
                t.data.mul_(self.m1).add_(s.data, alpha=1-self.m1)
            for t, s in zip(self.t1_proj.parameters(), self.online_proj.parameters()):
                t.data.mul_(self.m1).add_(s.data, alpha=1-self.m1)
            for t, s in zip(self.t2_enc.parameters(), self.online_enc.parameters()):
                t.data.mul_(self.m2).add_(s.data, alpha=1-self.m2)
            for t, s in zip(self.t2_proj.parameters(), self.online_proj.parameters()):
                t.data.mul_(self.m2).add_(s.data, alpha=1-self.m2)

    def forward_views(self, x1, x2, edge_index):
        # Online
        h1_all, h1_heads = self.online_enc(x1, edge_index)
        h2_all, h2_heads = self.online_enc(x2, edge_index)
        h1 = self.online_proj(h1_all)
        h2 = self.online_proj(h2_all)
        p1 = self.online_pred(h1)  # predictor applied to projections (BYOL-style)
        p2 = self.online_pred(h2)

        # Targets (no gradients)
        with torch.no_grad():
            z1_all_t1, _ = self.t1_enc(x1, edge_index)
            z1_t1 = self.t1_proj(z1_all_t1)
            z2_all_t1, _ = self.t1_enc(x2, edge_index)
            z2_t1 = self.t1_proj(z2_all_t1)

            z1_all_t2, _ = self.t2_enc(x1, edge_index)
            z1_t2 = self.t2_proj(z1_all_t2)
            z2_all_t2, _ = self.t2_enc(x2, edge_index)
            z2_t2 = self.t2_proj(z2_all_t2)

        return (p1, p2, h1, h2, h1_heads, h2_heads,
                z1_t1, z2_t1, z1_t2, z2_t2)

    def train_step(self, data, x1, x2, alpha=0.5, beta=1.0, gamma=1.0, tau=1.0):
        # 1) forward
        (p1, p2, h1, h2, h1_heads, h2_heads,
         z1_t1, z2_t1, z1_t2, z2_t2) = self.forward_views(x1, x2, data.edge_index)

        N = h1.size(0)

        # 2) Cross-Network loss L_cn: online vs two targets (positives are same indices)
        # view1:
        logits_11, labels = info_nce_logits(p1, z1_t1, tau)
        logits_12, _ = info_nce_logits(p1, z1_t2, tau)
        loss_cn_v1 = - (alpha * F.log_softmax(logits_11, dim=1)[torch.arange(N), labels] +
                        (1 - alpha) * F.log_softmax(logits_12, dim=1)[torch.arange(N), labels]).mean()
        # view2:
        logits_21, labels = info_nce_logits(p2, z2_t1, tau)
        logits_22, _ = info_nce_logits(p2, z2_t2, tau)
        loss_cn_v2 = - (alpha * F.log_softmax(logits_21, dim=1)[torch.arange(N), labels] +
                        (1 - alpha) * F.log_softmax(logits_22, dim=1)[torch.arange(N), labels]).mean()
        L_cn = 0.5 * (loss_cn_v1 + loss_cn_v2)

        # 3) Cross-View loss L_cv: inter-view & intra-view negatives
        # inter-view (h1 <-> h2) symmetrical:
        logits_ih, labels = info_nce_logits(p1, h2.detach(), tau)
        loss_inter_12 = F.cross_entropy(logits_ih, labels)
        logits_hi, labels = info_nce_logits(p2, h1.detach(), tau)
        loss_inter_21 = F.cross_entropy(logits_hi, labels)
        # intra-view: keep same numerator but denominator includes both views' banks
        bank_both = torch.cat([h1.detach(), h2.detach()], dim=0)  # [2N, d]
        logits_intra_1 = cosine_sim(p1, bank_both) / tau
        loss_intra_1 = -F.log_softmax(logits_intra_1, dim=1)[torch.arange(N), labels].mean()
        logits_intra_2 = cosine_sim(p2, bank_both) / tau
        loss_intra_2 = -F.log_softmax(logits_intra_2, dim=1)[torch.arange(N), labels].mean()
        L_cv = (loss_inter_12 + loss_inter_21 + loss_intra_1 + loss_intra_2) / 3.0  # 与论文(8)(9)同阶数量级

        # 4) Neighbor contrast L_head ：多头、跨视图、多正样本
        # 构造正样本索引：同节点跨视图 + 图中的邻居（同/跨视图）
        edge_index = data.edge_index
        # adjacency list
        neighbors = [[] for _ in range(N)]
        row, col = edge_index
        for r, c in zip(row.tolist(), col.tolist()):
            neighbors[r].append(c)
            neighbors[c].append(r)

        # anchors = 每个 head 的表示；bank = 另一 head 的表示拼接两视图
        head_losses = []
        # 使用每个 head 对其它 head 聚合 (k != l)
        for k in range(len(h1_heads)):
            # anchor: head k (view1)
            a1 = h1_heads[k]  # [N, d_h]
            # bank: head l(view1)+head l(view2)  (对所有 l != k)
            banks = []
            pos_idx = []
            for l in range(len(h1_heads)):
                if l == k:
                    continue
                b1 = h1_heads[l].detach()
                b2 = h2_heads[l].detach()
                banks.append(b1)
                banks.append(b2)
            bank = torch.cat(banks, dim=0)  # [M, d_h], M=2*(heads-1)*N

            # 构建每个 i 的 positives：同节点在 bank 中的位置 + 邻居在两视图中的位置
            # bank的布局是 [b1(l!=k), b2(l!=k)]，块大小都是 N
            per_block = N
            blocks = 2*(len(h1_heads)-1)
            # 对每个 i，positives 为：所有块里的 index i （同节点跨头跨视图）以及邻居 j 的对应 index
            for i in range(N):
                this_pos = []
                # 同节点
                for blk in range(blocks):
                    this_pos.append(blk*per_block + i)
                # 邻居
                for j in neighbors[i]:
                    for blk in range(blocks):
                        this_pos.append(blk*per_block + j)
                pos_idx.append(this_pos)

            lh = supervised_contrastive_loss(a1, pos_idx, bank, tau)
            head_losses.append(lh)

        L_head = torch.stack(head_losses).mean() if head_losses else torch.tensor(0.0, device=h1.device)

        # 5) 总损失
        L = alpha * L_cn + beta * L_cv + gamma * L_head

        # 6) EMA 同步 targets
        self._ema_update()
        return L, {'L_cn': L_cn.item(), 'L_cv': L_cv.item(), 'L_head': L_head.item()}

# ----------------------------
# Training / Evaluation
# ----------------------------
def get_dataset(name: str, root: str = './data'):
    name = name.capitalize()
    ds = Planetoid(root=root, name=name)
    return ds

def train_selfsup(model: TripleGNN,
                  data,
                  local_aug: LocalAugmentor,
                  global_aug: SVDGlobalAugmentor,
                  epochs=400, lr=1e-3, tau=1.0, alpha=0.5, beta=1.0, gamma=1.0,
                  feature_drop=0.2,
                  device='cpu'):
    model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)

    # Pretrain CVAE (freeze)
    local_aug.pretrain(data.x, data.edge_index, steps=150, batch=4096, lr=1e-3)

    for epoch in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        # build two views
        x1 = local_aug.apply(data.x)  # local view
        x1 = F.dropout(x1, p=feature_drop, training=True)

        x2, _ = global_aug.apply(data.x, data.edge_index, data.num_nodes)
        x2 = F.dropout(x2, p=feature_drop, training=True)

        loss, logs = model.train_step(data, x1, x2, alpha=alpha, beta=beta, gamma=gamma, tau=tau)
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"[{epoch:4d}] loss={loss.item():.4f} | L_cn={logs['L_cn']:.4f} "
                  f"| L_cv={logs['L_cv']:.4f} | L_head={logs['L_head']:.4f}")
    # final embeddings (use online encoder + projector output on original features)
    model.eval()
    with torch.no_grad():
        z_all, _ = model.online_enc(data.x, data.edge_index)
        z = model.online_proj(z_all)
        z = F.normalize(z, dim=-1)
    return z

def evaluate_lr(z: Tensor, data):
    # train LR on train mask, tune by val mask, report test
    X = z.cpu().numpy()
    y = data.y.cpu().numpy()
    train_idx = data.train_mask.cpu().numpy().astype(bool)
    val_idx = data.val_mask.cpu().numpy().astype(bool)
    test_idx = data.test_mask.cpu().numpy().astype(bool)

    # 简单做法：在 (train+val) 上训练，直接在 test 上评估；若严格，可用 val做超参搜索
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(X[train_idx | val_idx], y[train_idx | val_idx])
    y_pred = clf.predict(X[test_idx])
    acc = accuracy_score(y[test_idx], y_pred)
    return acc

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--embed', type=int, default=256)
    parser.add_argument('--proj', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--svd-k', type=int, default=64)
    parser.add_argument('--svd-mode', type=str, default='features', choices=['features','adjacency'])
    parser.add_argument('--feature-drop', type=float, default=0.2)
    parser.add_argument('--attn-drop', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = get_dataset(args.dataset)
    data = ds[0]

    local_aug = LocalAugmentor(in_dim=data.num_node_features, z_dim=min(64, data.num_node_features),
                               hidden=max(256, 2*data.num_node_features), mix=0.5, device=device)
    global_aug = SVDGlobalAugmentor(k=args.svd_k, mode=args.svd_mode)

    model = TripleGNN(in_dim=data.num_node_features,
                      hidden=args.hidden,
                      out_dim=args.embed,
                      proj_dim=args.proj,
                      heads=args.heads,
                      feat_drop=args.feature_drop,
                      attn_drop=args.attn_drop,
                      layers=args.layers)

    z = train_selfsup(model, data, local_aug, global_aug,
                      epochs=args.epochs, lr=args.lr, tau=args.tau,
                      alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                      feature_drop=args.feature_drop, device=device)

    acc = evaluate_lr(z, data)
    print(f"Test Accuracy (LR on frozen embeddings): {acc*100:.2f}%")

if __name__ == '__main__':
    main()
