"""Re-implementation of the KG-GCN experiment from the RDF ontology (pure torch/numpy, no Neo4j).
Compares the original protocol with fixes/optimisations and simple baselines.

Usage:  python experiments/node_classification.py [n_seeds=5] [out.json]
The ontology path can be overridden with the KG_RDF environment variable."""
import os, sys, math, copy, random, collections, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import rdflib
from rdflib.namespace import RDF, OWL

RDF_PATH = os.environ.get("KG_RDF", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "Ontology", "OntologiHamaPenyakitPadi_v1.1.rdf"))
NS = "http://www.semanticweb.org/Ariful/ontologies/2023/06/OntologiHamaPenyakitPadi#"

# ---------------- graph ----------------
g = rdflib.Graph(); g.parse(RDF_PATH)
typ = {}
for k in ['Gejala', 'HamaPadi', 'PenyakitPadi']:
    for i in g.subjects(RDF.type, rdflib.URIRef(NS + k)): typ.setdefault(i, set()).add(k)
op = set(g.subjects(RDF.type, OWL.ObjectProperty))
triples = sorted([(str(s), str(p), str(o)) for s, p, o in g if p in op and s in typ and o in typ])
nodes = sorted({s for s, _, o in triples} | {o for s, _, o in triples})
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)
y = np.array([1 if typ[rdflib.URIRef(n)] & {'HamaPadi', 'PenyakitPadi'} else 0 for n in nodes])
# message direction as in original code: object -> subject
src = np.array([idx[o] for s, _, o in triples]); dst = np.array([idx[s] for s, _, o in triples])

def norm_adj(src, dst, undirected=False):
    A = torch.zeros(N, N)
    A[dst, src] = 1.0                      # A[u,v]=1 : v sends to u
    if undirected: A = ((A + A.T) > 0).float()
    A = A + torch.eye(N)
    d = A.sum(1)                           # in-degree + 1 (PyG gcn_norm uses target degree)
    dinv = d.pow(-0.5)
    return dinv[:, None] * A * dinv[None, :]

A_dir = norm_adj(src, dst); A_und = norm_adj(src, dst, True)
indeg = np.bincount(dst, minlength=N); outdeg = np.bincount(src, minlength=N)

# ---------------- node2vec (p=q=1, walks on DiGraph object->subject as in original) ----------------
def node2vec(seed, dim=64, walk_len=10, num_walks=50, window=5, neg=5, epochs=5, directed=True):
    rng = random.Random(seed); torch.manual_seed(seed)
    nbr = collections.defaultdict(list)
    for a, b in zip(src, dst):
        nbr[a].append(b)
        if not directed: nbr[b].append(a)
    walks = []
    for _ in range(num_walks):
        order = list(range(N)); rng.shuffle(order)
        for s in order:
            w = [s]
            while len(w) < walk_len and nbr[w[-1]]: w.append(rng.choice(nbr[w[-1]]))
            walks.append(w)
    pairs = [(w[i], w[j]) for w in walks for i in range(len(w))
             for j in range(max(0, i - window), min(len(w), i + window + 1)) if i != j]
    if not pairs: pairs = [(i, i) for i in range(N)]
    pairs = torch.tensor(pairs)
    cnt = torch.bincount(torch.tensor([v for w in walks for v in w]), minlength=N).float() ** 0.75
    noise = cnt / cnt.sum()
    emb = nn.Embedding(N, dim); ctx = nn.Embedding(N, dim)
    nn.init.uniform_(emb.weight, -0.5 / dim, 0.5 / dim); nn.init.zeros_(ctx.weight)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ctx.parameters()), lr=0.01)
    for _ in range(epochs * 4):
        perm = torch.randperm(len(pairs))
        for b in range(0, len(pairs), 1024):
            pb = pairs[perm[b:b + 1024]]
            u, v = emb(pb[:, 0]), ctx(pb[:, 1])
            ng = ctx(torch.multinomial(noise, len(pb) * neg, replacement=True)).view(len(pb), neg, dim)
            loss = -(F.logsigmoid((u * v).sum(1)) + F.logsigmoid(-(ng @ u.unsqueeze(2)).squeeze(2)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return emb.weight.detach().clone()

# ---------------- models ----------------
class GCN(nn.Module):
    def __init__(self, din, h, layers=3, dropout=0.7, bn=True):
        super().__init__()
        dims = [din] + [h] * (layers - 1) + [2]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(h) if bn else nn.Identity() for _ in range(layers - 1))
        self.dp = dropout
    def forward(self, x, A):
        for i, lin in enumerate(self.lins):
            x = A @ lin(x)          # GCNConv: A_hat (X W) + b  (bias after propagation in PyG; negligible)
            if i < len(self.lins) - 1:
                x = F.dropout(F.relu(self.bns[i](x)), self.dp, self.training)
        return x

class EgoNbrGCN(GCN):
    """h' = W_self h_u + W_nbr * sum_v A_hat_uv h_v  (self and neighbour weights separated)."""
    def __init__(self, din, h, layers=3, dropout=0.5, bn=True):
        super().__init__(din, h, layers, dropout, bn)
        dims = [din] + [h] * (layers - 1) + [2]
        self.selfl = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
    def forward(self, x, A):
        An = A - torch.diag(torch.diag(A))
        for i, (lin, sl) in enumerate(zip(self.lins, self.selfl)):
            x = An @ lin(x) + sl(x)
            if i < len(self.lins) - 1:
                x = F.dropout(F.relu(self.bns[i](x)), self.dp, self.training)
        return x

class MLP(GCN):
    def forward(self, x, A): return super().forward(x, torch.eye(N))

# ---------------- metrics ----------------
def metrics(yt, s, rank_score=None):
    rs = s if rank_score is None else rank_score
    o = np.argsort(-rs, kind='stable'); r = np.where(yt[o] == 1)[0] + 1
    n, m = len(r), len(yt)
    ap = np.mean(np.arange(1, n + 1) / r)
    Hn = sum(1 / j for j in range(1, n + 1))
    nmrr = np.mean(1 / r) * n / Hn
    ndcg = sum(1 / math.log2(x + 1) for x in r) / sum(1 / math.log2(j + 1) for j in range(1, n + 1))
    neg = rs[yt == 0]; pos = rs[yt == 1]
    auc = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    p = np.clip(s, 1e-7, 1 - 1e-7)
    ll = -np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))
    return dict(AP=ap, nMRR=nmrr, NDCG=ndcg, AUC=auc, P1=float(r[0] == 1), P3=np.sum(r <= 3) / 3,
                Acc=np.mean((s > 0.5) == yt), LogLoss=ll)

def strat_folds(y, k, seed):
    rng = np.random.RandomState(seed); folds = [[] for _ in range(k)]
    for c in (0, 1):
        ids = np.where(y == c)[0]; rng.shuffle(ids)
        for j, i in enumerate(ids): folds[j % k].append(i)
    return [np.array(f) for f in folds]

def fit_temperature(z, yt):
    lt = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([lt], max_iter=100)
    z = torch.tensor(z, dtype=torch.float); yt = torch.tensor(yt, dtype=torch.float)
    def cl():
        opt.zero_grad(); l = F.binary_cross_entropy_with_logits(z / lt.clamp(-3, 3).exp(), yt); l.backward(); return l
    opt.step(cl); return float(lt.detach().clamp(-3, 3).exp())

# ---------------- training protocol ----------------
def run_fold(X, A, tr, te, cfg, seed):
    torch.manual_seed(seed)
    Model = MLP if cfg.get('mlp') else (EgoNbrGCN if cfg.get('ego') else GCN)
    model = Model(X.shape[1], cfg['h'], cfg['layers'], cfg['dropout'], cfg['bn'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    yt = torch.tensor(y)
    if cfg['val'] == 'test':            # original: early stopping on the held-out fold
        fit, val = tr, te
    else:                               # fixed: inner stratified validation split from training fold
        rng = np.random.RandomState(seed); v = []
        for c in (0, 1):
            ids = tr[y[tr] == c].copy(); rng.shuffle(ids); v += list(ids[:max(1, len(ids) // 9)])
        val = np.array(v); fit = np.setdiff1d(tr, val)
    best, bad, state = 1e9, 0, None
    for ep in range(cfg['epochs']):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(X, A)[fit], yt[fit]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = F.cross_entropy(model(X, A)[val], yt[val]).item()
        if vl < best:
            best, bad = vl, 0
            state = model.state_dict() if cfg.get('bug') else copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg['patience']: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): Z = model(X, A)
    marg = (Z[:, 1] - Z[:, 0]).numpy()
    T = fit_temperature(marg[val], y[val]) if cfg.get('temp') else 1.0
    s = 1 / (1 + np.exp(-marg / T))
    return metrics(y[te], s[te], marg[te])

def evaluate(cfg, seeds):
    res = []
    for sd in seeds:
        X = node2vec(sd, directed=cfg.get('n2v_directed', True))
        if cfg.get('feat') == 'onehot': X = torch.eye(N)
        A = A_und if cfg.get('undirected') else A_dir
        for f, te in enumerate(strat_folds(y, 10, sd)):
            tr = np.setdiff1d(np.arange(N), te)
            res.append(run_fold(X, A, tr, te, cfg, sd * 100 + f))
    return {k: (np.mean([r[k] for r in res]), np.std([r[k] for r in res])) for k in res[0]}, \
        sum(r['AUC'] < 0.5 for r in res) / len(res)

def baseline(score_fn, seeds):
    res = []
    for sd in seeds:
        rng = np.random.RandomState(sd)
        for te in strat_folds(y, 10, sd):
            res.append(metrics(y[te], score_fn(te, rng)))
    return {k: (np.mean([r[k] for r in res]), np.std([r[k] for r in res])) for k in res[0]}, \
        sum(r['AUC'] < 0.5 for r in res) / len(res)

if __name__ == '__main__':
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
    print(f"N={N} pos={y.sum()} arcs={len(triples)} indeg0 among neg={np.sum((indeg==0)&(y==0))}/{np.sum(y==0)}, "
          f"indeg0 among pos={np.sum((indeg==0)&(y==1))}")
    orig = dict(h=16, layers=3, dropout=0.7, bn=True, lr=1e-3, wd=0.05, epochs=50, patience=5, val='test', bug=True)
    fixed = dict(orig, val='inner', bug=False)
    tuned = dict(fixed, dropout=0.5, lr=1e-2, wd=5e-4, epochs=300, patience=30)
    ego = dict(tuned, ego=True, temp=True)
    exps = {
        'B0 random scorer': None, 'B1 in-degree rule': None,
        'B2 MLP on Node2Vec (no graph)': dict(tuned, mlp=True, temp=True),
        'E0 original protocol': orig,
        'E1 + inner val, deepcopy fix': fixed,
        'E2 + tuned (lr1e-2, do.5, wd5e-4, 300ep)': tuned,
        'E3 E2 + temperature scaling': dict(tuned, temp=True),
        'E4 E3 + undirected propagation': dict(tuned, temp=True, undirected=True),
        'E6 ego/nbr-separated GCN (directed)': ego,
        'E7 ego/nbr-separated GCN (undirected)': dict(ego, undirected=True),
        'E8 ego/nbr GCN, undirected, one-hot feats': dict(ego, undirected=True, feat='onehot'),
    }
    out = {}
    for name, cfg in exps.items():
        if name.startswith('B0'): r = baseline(lambda te, rng: rng.rand(len(te)), seeds)
        elif name.startswith('B1'): r = baseline(lambda te, rng: indeg[te] + 1e-3 * rng.rand(len(te)), seeds)
        else: r = evaluate(cfg, seeds)
        out[name] = r
        m, inv = r
        print(f"{name:42s} " + " ".join(f"{k}={v[0]:.3f}±{v[1]:.3f}" for k, v in m.items()) + f" invertedFolds={inv:.0%}", flush=True)
    json.dump({k: [{kk: list(vv) for kk, vv in v[0].items()}, v[1]] for k, v in out.items()},
              open(sys.argv[2] if len(sys.argv) > 2 else 'results/node_classification.json', 'w'), indent=1)
