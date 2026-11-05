"""Offline evaluation: hold-out split, hyper-parameter tuning on a validation split,
and comparison of all models (RMSE/MAE for rating predictors; P/R/NDCG@10, coverage, diversity for rankings)."""
import itertools
import json
import os
import numpy as np
import pandas as pd
from data_utils import load_data, genre_matrix
from models import (PopularityRecommender, ItemKNNRecommender, ALSRecommender,
                    ContentRecommender, HybridRecommender, zscore, top_n)

K = 10
REL = 4.0


def holdout_split(R, frac, rng, min_keep=5):
    train, test = R.copy(), np.full_like(R, np.nan)
    for u in range(R.shape[0]):
        idx = np.where(~np.isnan(R[u]))[0]
        n_test = int(round(frac * len(idx)))
        n_test = max(0, min(n_test, len(idx) - min_keep))
        if n_test:
            ch = rng.choice(idx, n_test, replace=False)
            test[u, ch] = R[u, ch]
            train[u, ch] = np.nan
    return train, test


def score_matrix(model, R_train, fn="score"):
    return np.vstack([zscore(getattr(model, fn)(R_train[u])) if fn == "score" else getattr(model, fn)(R_train[u])
                      for u in range(R_train.shape[0])])


def rating_errors(P, R_test):
    m = ~np.isnan(R_test)
    d = P[m] - R_test[m]
    return float(np.sqrt((d ** 2).mean())), float(np.abs(d).mean())


def topk_lists(S, R_train, k=K, sim=None, lam=None):
    return [top_n(S[u], R_train[u], n=k, sim=sim, lam=lam) for u in range(S.shape[0])]


def ranking_metrics(lists, R_test, G, n_items, k=K):
    p, r, nd, ild, used = [], [], [], [], set()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
    gsim = Gn @ Gn.T
    for u, L in enumerate(lists):
        used.update(L)
        rel = set(np.where(R_test[u] >= REL)[0])
        hits = np.array([1.0 if i in rel else 0.0 for i in L])
        if len(L) > 1:
            pairs = [1 - gsim[a, b] for a, b in itertools.combinations(L, 2)]
            ild.append(np.mean(pairs))
        if not rel:
            continue
        p.append(hits.sum() / k)
        r.append(hits.sum() / len(rel))
        dcg = (hits * disc[:len(hits)]).sum()
        idcg = disc[:min(len(rel), k)].sum()
        nd.append(dcg / idcg)
    return {"Precision@10": np.mean(p), "Recall@10": np.mean(r), "NDCG@10": np.mean(nd),
            "Coverage@10": len(used) / n_items, "Diversity@10": np.mean(ild)}


def tune_hybrid(Z, R_fit, R_val, G, n_items, grid=(0, 1, 2)):
    best, best_w = -1, None
    names = list(Z)
    for combo in itertools.product(grid, repeat=len(names)):
        if sum(combo) == 0:
            continue
        S = sum(w * Z[n] for w, n in zip(combo, names))
        nd = ranking_metrics(topk_lists(S, R_fit), R_val, G, n_items)["NDCG@10"]
        if nd > best:
            best, best_w = nd, dict(zip(names, map(float, combo)))
    return best_w, best


def run_experiment(data_dir="data", out_dir="results", seed=7, force_synthetic=False):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    R, items, src = load_data(data_dir, force_synthetic)
    G, _ = genre_matrix(items)
    n_users, n_items = R.shape
    print(f"Data source: {src} | users={n_users} items={n_items} ratings={int((~np.isnan(R)).sum())} "
          f"density={(~np.isnan(R)).mean():.3f}")

    R_train, R_test = holdout_split(R, 0.20, rng)
    R_fit, R_val = holdout_split(R_train, 0.15, rng)

    # ---- tune model hyper-parameters on validation RMSE
    knn_res = {}
    for k in (10, 20, 40):
        m = ItemKNNRecommender(k=k).fit(R_fit, items)
        knn_res[k] = rating_errors(score_matrix(m, R_fit, "predict"), R_val)[0]
    best_k = min(knn_res, key=knn_res.get)
    als_res = {}
    for f in (4, 8, 16):
        m = ALSRecommender(factors=f).fit(R_fit, items)
        als_res[f] = rating_errors(score_matrix(m, R_fit, "predict"), R_val)[0]
    best_f = min(als_res, key=als_res.get)
    print("kNN val RMSE by k:", {k: round(v, 3) for k, v in knn_res.items()}, "-> k =", best_k)
    print("ALS val RMSE by factors:", {k: round(v, 3) for k, v in als_res.items()}, "-> factors =", best_f)

    def build(Rt):
        return {"pop": PopularityRecommender().fit(Rt, items),
                "knn": ItemKNNRecommender(k=best_k).fit(Rt, items),
                "als": ALSRecommender(factors=best_f).fit(Rt, items),
                "content": ContentRecommender().fit(Rt, items)}

    # ---- tune hybrid weights on validation NDCG
    mods = build(R_fit)
    Zval = {k: score_matrix(m, R_fit) for k, m in mods.items()}
    weights, val_ndcg = tune_hybrid(Zval, R_fit, R_val, G, n_items)
    print("Hybrid weights:", weights, "| val NDCG@10 = %.4f" % val_ndcg)

    # ---- final evaluation on the untouched test split
    mods = build(R_train)
    Z = {k: score_matrix(m, R_train) for k, m in mods.items()}
    S_h = sum(weights[k] * Z[k] for k in Z)
    sim = mods["content"].sim

    rows = []
    labels = {"pop": "Popularity", "knn": "Item-kNN CF", "als": "ALS matrix factorisation", "content": "Content-based"}
    for k, m in mods.items():
        row = {"Model": labels[k]}
        if m.predicts_ratings:
            row["RMSE"], row["MAE"] = rating_errors(score_matrix(m, R_train, "predict"), R_test)
        row.update(ranking_metrics(topk_lists(Z[k], R_train), R_test, G, n_items))
        rows.append(row)
    rows.append({"Model": "Hybrid (tuned)", **ranking_metrics(topk_lists(S_h, R_train), R_test, G, n_items)})
    rows.append({"Model": "Hybrid + MMR (lambda=0.75)",
                 **ranking_metrics(topk_lists(S_h, R_train, sim=sim, lam=0.75), R_test, G, n_items)})
    rows.append({"Model": "Hybrid + MMR (lambda=0.60)",
                 **ranking_metrics(topk_lists(S_h, R_train, sim=sim, lam=0.60), R_test, G, n_items)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump({"weights": weights, "knn_k": best_k, "als_factors": best_f, "data": src}, f, indent=2)
    print(df.round(4).to_string(index=False))
    make_chart(df, os.path.join(out_dir, "metrics.png"))
    return df, weights


def make_chart(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = {"ALS matrix factorisation": "ALS MF", "Hybrid + MMR (lambda=0.75)": "Hybrid+MMR .75",
             "Hybrid + MMR (lambda=0.60)": "Hybrid+MMR .60", "Hybrid (tuned)": "Hybrid"}
    names = [short.get(n, n) for n in df["Model"]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    colors = ["#8c9eb5"] * 4 + ["#2f6f9f", "#d98a2b", "#d98a2b"]
    ax[0].barh(names, df["NDCG@10"], color=colors)
    ax[0].set_title("Ranking quality (NDCG@10, higher is better)")
    ax[1].barh(names, df["Diversity@10"], color=colors)
    ax[1].set_title("List diversity (genre ILD@10)")
    sub = df.dropna(subset=["RMSE"])
    ax[2].barh([short.get(n, n) for n in sub["Model"]], sub["RMSE"], color="#8c9eb5")
    ax[2].set_title("Rating error (RMSE, lower is better)")
    for a in ax:
        a.invert_yaxis()
        a.spines[["top", "right"]].set_visible(False)
    ax[1].set_yticklabels([])
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


if __name__ == "__main__":
    run_experiment()
