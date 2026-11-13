"""Symptom-conditioned diagnosis as link prediction: given a symptom s, rank the 43 pest/disease entities.
Held-out (diagnosis, symptom) arcs are removed from the graph before embedding and training.

Usage:  python experiments/link_prediction.py [n_seeds=3]"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import node_classification as K

N, y = K.N, K.y
POS = np.where(y == 1)[0]
pairs = sorted({(int(d), int(s)) for d, s in zip(K.dst, K.src) if y[d] == 1 and y[s] == 0})  # (diagnosis, symptom)
known = set(pairs)

def adj(edges):
    A = torch.zeros(N, N)
    for a, b in edges: A[a, b] = A[b, a] = 1.0
    return A

def gcn_norm(A, self_loop=True):
    A2 = A + torch.eye(N) if self_loop else A
    d = A2.sum(1).clamp(min=1); di = d.pow(-0.5)
    return di[:, None] * A2 * di[None, :]

class Enc(nn.Module):
    def __init__(self, din, h=32, ego=True, layers=2, dp=0.3):
        super().__init__()
        self.l = nn.ModuleList([nn.Linear(din, h)] + [nn.Linear(h, h) for _ in range(layers - 1)])
        self.s = nn.ModuleList([nn.Linear(din, h, bias=False)] + [nn.Linear(h, h, bias=False) for _ in range(layers - 1)])
        self.ego, self.dp = ego, dp
    def forward(self, x, A):
        for i, (l, s) in enumerate(zip(self.l, self.s)):
            x = A @ l(x) + (s(x) if self.ego else 0)
            if i < len(self.l) - 1: x = F.dropout(F.relu(x), self.dp, self.training)
        return x

def rank_of(scores, d, s):
    cand = [c for c in POS if c == d or (c, s) not in known]          # filtered setting
    sc = scores[cand]; true = scores[d]
    return 1 + int(np.sum(sc > true)) + 0.5 * int(np.sum(sc == true) - 1)

def summarize(ranks):
    r = np.array(ranks)
    return dict(MRR=np.mean(1 / r), H1=np.mean(r <= 1), H3=np.mean(r <= 3), H10=np.mean(r <= 10), MR=np.mean(r))

def run(method, seeds):
    allr = []
    for sd in seeds:
        rng = np.random.RandomState(sd); perm = rng.permutation(len(pairs))
        for f in range(10):
            te = [pairs[i] for i in perm[f::10]]; tes = set(te)
            tr_arcs = [(int(a), int(b)) for a, b in zip(K.dst, K.src) if (int(a), int(b)) not in tes]
            A = adj(tr_arcs); trp = [p for p in pairs if p not in tes]
            if method == 'random':
                sc = lambda s: rng.rand(N)
            elif method == 'popularity':
                deg = A.sum(1).numpy(); sc = lambda s: deg + 1e-6 * rng.rand(N)
            elif method == 'paths3':      # number of length-3 walks d-s'-d'-s (symptom co-occurrence)
                A3 = (A @ A @ A).numpy(); deg = A.sum(1).numpy()
                sc = lambda s: A3[:, s] + 1e-3 * deg + 1e-6 * rng.rand(N)
            else:
                src = np.array([b for a, b in tr_arcs]); dst = np.array([a for a, b in tr_arcs])
                Ksrc, Kdst = K.src, K.dst; K.src, K.dst = src, dst
                X = K.node2vec(sd * 100 + f, directed=False); K.src, K.dst = Ksrc, Kdst
                if method == 'node2vec-dot':
                    Z = X.numpy(); sc = lambda s: Z @ Z[s]
                else:
                    torch.manual_seed(sd * 100 + f)
                    enc = Enc(X.shape[1], ego=(method == 'ego-gcn'))
                    W = nn.Parameter(torch.eye(32))                  # bilinear decoder z_d^T W z_s
                    opt = torch.optim.Adam(list(enc.parameters()) + [W], lr=1e-2, weight_decay=5e-4)
                    An = gcn_norm(A, self_loop=(method != 'ego-gcn'))
                    P = torch.tensor(trp)
                    for ep in range(200):
                        enc.train(); opt.zero_grad(); Z = enc(X, An)
                        neg = torch.stack([torch.tensor(rng.choice(POS, len(P))), P[:, 1]], 1)
                        lp = ((Z[P[:, 0]] @ W) * Z[P[:, 1]]).sum(1); ln = ((Z[neg[:, 0]] @ W) * Z[neg[:, 1]]).sum(1)
                        loss = F.binary_cross_entropy_with_logits(torch.cat([lp, ln]), torch.cat([torch.ones_like(lp), torch.zeros_like(ln)]))
                        loss.backward(); opt.step()
                    enc.eval()
                    with torch.no_grad(): Z = enc(X, An); S = ((Z @ W) @ Z.T).numpy()
                    sc = lambda s: S[:, s]
            for d, s in te: allr.append(rank_of(sc(s), d, s))
    return summarize(allr)

if __name__ == '__main__':
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    print(f"(diagnosis,symptom) pairs={len(pairs)}, candidates per query={len(POS)}")
    out = {}
    for m in ['random', 'popularity', 'paths3', 'node2vec-dot', 'gcn', 'ego-gcn']:
        out[m] = run(m, seeds)
        print(f"{m:14s} " + " ".join(f"{k}={v:.3f}" for k, v in out[m].items()), flush=True)
    json.dump(out, open('results/link_prediction.json', 'w'), indent=1)
