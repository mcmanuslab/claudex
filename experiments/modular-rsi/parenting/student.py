"""Stage 1: train the infant mutator to imitate the search parent; evaluate the
pre-registered gates G1-G4 on held-out universes (see PREREG.md)."""
import glob, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from scipy_free import wilcoxon_signed_rank_one_sided as wilcoxon

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
HELD_OUT = {"invB", "min"}
S, B, PROBE_N = 150, 32, 256


def load(ids):
    out = []
    for u in ids:
        p = os.path.join(RES, f"universe_{u}.json")
        if os.path.exists(p):
            out += json.load(open(p))
    return out


def prep(d, mu=None, sd=None):
    f = np.array(d["feats"], np.float32)
    f[:, 7:11] = np.log1p(f[:, 7:11])                    # counts / age
    if mu is not None:
        f = (f - mu) / sd
    loss = np.array(d["loss"], np.float32)
    rng = loss.max() - loss.min()
    norm = (loss - loss.min()) / (rng if rng > 1e-9 else 1.0)    # 0 = parent's pick, 1 = worst
    return f, norm


class Mutator(nn.Module):
    """Tiny transformer over the set of candidates -> choice logits + predicted outcome."""

    def __init__(self, F, d=32):
        super().__init__()
        self.inp = nn.Linear(F, d)
        layer = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.score, self.value = nn.Linear(d, 1), nn.Linear(d, 1)

    def forward(self, f):
        h = self.enc(self.inp(f)[None])[0]
        return self.score(h)[:, 0], self.value(h)[:, 0]


def train_student(train, val, seed, epochs=200):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    allf = np.concatenate([np.array(d["feats"], np.float32) for d in train])
    allf[:, 7:11] = np.log1p(allf[:, 7:11])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    data = [prep(d, mu, sd) + (d["best"],) for d in train]
    vdata = [prep(d, mu, sd) + (d["best"],) for d in val]
    m = Mutator(allf.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = 1e9, None
    for ep in range(epochs):
        m.train()
        for i in rng.permutation(len(data)):
            f, norm, y = data[i]
            s, v = m(torch.from_numpy(f))
            loss = nn.functional.cross_entropy(s[None], torch.tensor([y])) + \
                nn.functional.mse_loss(torch.sigmoid(v), torch.from_numpy(norm))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vr = np.mean([norm[int(m(torch.from_numpy(f))[0].argmax())] for f, norm, _ in vdata]) if vdata else 0
        if vr < best:
            best, best_state = vr, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m, mu, sd


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def baselines(d):
    f = np.array(d["feats"]); cands = d["cands"]
    r16 = [c for c, (s, r) in enumerate(cands) if r == 16]
    probe = min(r16, key=lambda c: f[c, 2])                # lowest zero-shot probe loss (fresh included)
    default = next(c for c, (s, r) in enumerate(cands) if s == -1 and r == 16)
    return dict(probe_best=probe, default=default)


def evaluate(models, test):
    rows = []
    for d in test:
        _, norm = prep(d)
        C = len(norm)
        picks, sps = [], []
        for m, mu, sd in models:
            f, _ = prep(d, mu, sd)
            with torch.no_grad():
                s, v = m(torch.from_numpy(f))
            picks.append(int(s.argmax())); sps.append(spearman(v.numpy(), norm))
        bl = baselines(d)
        rows.append(dict(op=d["skill"][0], C=C, best=d["best"],
                         student_regret=float(np.mean([norm[p] for p in picks])),
                         student_agree=float(np.mean([p == d["best"] for p in picks])),
                         student_spearman=float(np.nanmean(sps)) if not all(np.isnan(sps)) else np.nan,
                         probe_regret=float(norm[bl["probe_best"]]), probe_agree=float(bl["probe_best"] == d["best"]),
                         default_regret=float(norm[bl["default"]]), default_agree=float(bl["default"] == d["best"]),
                         random_regret=float(norm.mean()), random_agree=1.0 / C))
    return rows


def summarize(rows, tag):
    g = lambda k: float(np.mean([r[k] for r in rows]))
    sr = np.array([r["student_regret"] for r in rows])
    out = dict(n=len(rows), student_agree=g("student_agree"), probe_agree=g("probe_agree"),
               default_agree=g("default_agree"), chance_agree=g("random_agree"),
               student_regret=g("student_regret"), probe_regret=g("probe_regret"),
               default_regret=g("default_regret"), random_regret=g("random_regret"),
               student_spearman=float(np.nanmean([r["student_spearman"] for r in rows])),
               p_student_lt_probe=wilcoxon(sr - np.array([r["probe_regret"] for r in rows])),
               p_student_lt_random=wilcoxon(sr - np.array([r["random_regret"] for r in rows])))
    if tag == "all":
        out["G1"] = out["student_agree"] >= out["probe_agree"] + 0.10 and out["student_agree"] >= 2 * out["chance_agree"]
        out["G2"] = (out["student_regret"] <= 0.5 * out["probe_regret"] and out["student_regret"] < out["random_regret"]
                     and out["p_student_lt_probe"] < 0.05 and out["p_student_lt_random"] < 0.05)
        out["G3"] = out["student_spearman"] >= 0.6
        C = np.mean([r["C"] for r in rows])
        parent_cost = C * S * B * 3                                  # all candidates trained (fwd+bwd ~ 3x)
        student_cost = C * PROBE_N + S * B * 3                       # probes + training the chosen one
        out["G4_ratio"] = float(student_cost / parent_cost); out["G4"] = out["G4_ratio"] <= 0.10
        out["STAGE1_PASS"] = bool(out["G1"] and out["G2"] and out["G3"])
    return out


def diagnostic():
    rows = []
    for u in range(200, 212):
        p = os.path.join(RES, f"universe_{u}_long.json")
        if os.path.exists(p):
            for d in json.load(open(p)):
                ll = np.array(d["loss_long"]); rows.append(dict(
                    spearman=spearman(np.array(d["loss"]), ll),
                    pick_top2_long=bool(d["best"] in np.argsort(ll)[:2])))
    if not rows:
        return {}
    return dict(n=len(rows), spearman_150_vs_1500=float(np.nanmean([r["spearman"] for r in rows])),
                parent_pick_in_top2_at_1500=float(np.mean([r["pick_top2_long"] for r in rows])))


def main():
    train_all = load(range(0, 48)); test = load(range(100, 112))
    val_ids = set(range(40, 48))
    train = [d for d in train_all if d["uid"] not in val_ids]; val = [d for d in train_all if d["uid"] in val_ids]
    print(f"train {len(train)} val {len(val)} test {len(test)} decisions", flush=True)
    models = [train_student(train, val, seed) for seed in range(5)]
    rows = evaluate(models, test)
    res = dict(all=summarize(rows, "all"),
               held_out_families=summarize([r for r in rows if r["op"] in HELD_OUT], "held"),
               seen_families=summarize([r for r in rows if r["op"] not in HELD_OUT], "seen"),
               diagnostic=diagnostic(), per_decision=rows)
    json.dump(res, open(os.path.join(RES, "stage1.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_decision"}, indent=1))


if __name__ == "__main__":
    main()
