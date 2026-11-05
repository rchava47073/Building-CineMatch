"""Recommender models used by CineMatch.

Every model implements:
    fit(R, items)      R = users x items matrix, NaN = unrated
    score(r)           ranking scores for ALL items given ONE user's rating vector r (NaN = unrated)
    predict(r)         rating-scale predictions (only models with predicts_ratings = True)

Because models take a rating *vector* rather than a user id, the same code serves users in the
training matrix and brand-new users typing ratings into the chat interface (cold start / fold-in).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v, dtype=float)


# --------------------------------------------------------------------------- baseline: popularity
class PopularityRecommender:
    name, predicts_ratings = "Popularity", True

    def __init__(self, damping=5.0):
        self.damping = damping

    def fit(self, R, items=None):
        M = ~np.isnan(R)
        self.mu = np.nanmean(R)
        self.count = M.sum(0).astype(float)
        self.item_mean = (np.nansum(R, 0) + self.damping * self.mu) / (self.count + self.damping)
        return self

    def score(self, r):
        return self.count + 0.01 * self.item_mean          # most-rated first, quality breaks ties

    def predict(self, r):
        return self.item_mean.copy()


# --------------------------------------------------------------------------- item-based CF (foundation)
class ItemKNNRecommender:
    """Item-based collaborative filtering with adjusted cosine similarity (Sarwar et al., 2001).
    Modifications vs. the textbook version: significance shrinkage on co-rating counts and
    top-k neighbour pruning."""
    name, predicts_ratings = "Item-kNN CF", True

    def __init__(self, k=20, shrink=10.0):
        self.k, self.shrink = k, shrink

    def fit(self, R, items=None):
        M = ~np.isnan(R)
        self.mu = np.nanmean(R)
        user_mean = np.where(M.any(1), np.nansum(R, 1) / np.maximum(M.sum(1), 1), self.mu)
        Rc = np.where(M, R - user_mean[:, None], 0.0)
        num = Rc.T @ Rc
        norm = np.sqrt((Rc ** 2).sum(0))
        S = num / (np.outer(norm, norm) + 1e-9)
        co = M.T.astype(float) @ M.astype(float)
        S *= co / (co + self.shrink)
        np.fill_diagonal(S, 0.0)
        n_items = S.shape[0]
        Sk = np.zeros_like(S)
        k = min(self.k, n_items - 1)
        for i in range(n_items):
            top = np.argpartition(-S[:, i], k - 1)[:k]
            Sk[top, i] = np.maximum(S[top, i], 0.0)
        self.S, self.Sk = S, Sk
        cnt = M.sum(0)
        self.item_mean = (np.nansum(R, 0) + 5 * self.mu) / (cnt + 5)
        return self

    def predict(self, r):
        m = ~np.isnan(r)
        mu_u = r[m].mean() if m.any() else self.mu
        rc = np.where(m, r - mu_u, 0.0)
        num, den = rc @ self.Sk, m.astype(float) @ self.Sk
        pred = np.where(den > 1e-6, mu_u + num / np.maximum(den, 1e-6), self.item_mean)
        return np.clip(pred, 1, 5)

    def score(self, r):
        return self.predict(r)


# --------------------------------------------------------------------------- matrix factorisation (ALS)
class ALSRecommender:
    """Biased matrix factorisation trained with alternating least squares.
    rating ~ mu + b_u + b_i + p_u . q_i. New users are 'folded in' by solving for p_u only."""
    name, predicts_ratings = "Matrix factorisation (ALS)", True

    def __init__(self, factors=8, reg=0.08, iters=15, bias_reg=10.0, seed=0):
        self.f, self.reg, self.iters, self.bias_reg, self.seed = factors, reg, iters, bias_reg, seed

    def fit(self, R, items=None):
        U, I = R.shape
        M = ~np.isnan(R)
        self.mu = np.nanmean(R)
        self.bi = np.nansum(R - self.mu, 0) / (M.sum(0) + self.bias_reg)
        res = np.where(M, R - self.mu - self.bi[None, :], 0.0)
        self.bu = res.sum(1) / (M.sum(1) + self.bias_reg)
        E = np.where(M, res - self.bu[:, None], 0.0)
        rng = np.random.default_rng(self.seed)
        P = 0.1 * rng.normal(size=(U, self.f))
        Q = 0.1 * rng.normal(size=(I, self.f))
        eye = np.eye(self.f)
        for _ in range(self.iters):
            for u in range(U):
                idx = np.where(M[u])[0]
                if idx.size:
                    Qo = Q[idx]
                    P[u] = np.linalg.solve(Qo.T @ Qo + self.reg * idx.size * eye, Qo.T @ E[u, idx])
            for i in range(I):
                idx = np.where(M[:, i])[0]
                if idx.size:
                    Po = P[idx]
                    Q[i] = np.linalg.solve(Po.T @ Po + self.reg * idx.size * eye, Po.T @ E[idx, i])
        self.P, self.Q = P, Q
        return self

    def predict(self, r):
        m = ~np.isnan(r)
        if not m.any():
            return np.clip(self.mu + self.bi, 1, 5)
        base = self.mu + self.bi
        bu = (r[m] - base[m]).sum() / (m.sum() + self.bias_reg)
        e = r[m] - base[m] - bu
        Qo = self.Q[m]
        p = np.linalg.solve(Qo.T @ Qo + self.reg * m.sum() * np.eye(self.f), Qo.T @ e)
        return np.clip(base + bu + self.Q @ p, 1, 5)

    def score(self, r):
        return self.predict(r)


# --------------------------------------------------------------------------- content-based (TF-IDF)
class ContentRecommender:
    name, predicts_ratings = "Content-based (TF-IDF)", False

    def __init__(self, genre_weight=3):
        self.genre_weight = genre_weight

    def fit(self, R, items):
        def tok(g):
            return g.lower().replace("-", "").replace(" ", "")
        text = [" ".join([tok(g) for g in row.genres.split("|")] * self.genre_weight) + " " + str(row.description).lower()
                for row in items.itertuples()]
        self.X = TfidfVectorizer(token_pattern=r"[a-z0-9\-]+").fit_transform(text)
        self.sim = (self.X @ self.X.T).toarray()
        return self

    def score(self, r):
        m = ~np.isnan(r)
        if not m.any():
            return np.zeros(self.X.shape[0])
        w = r[m] - r[m].mean()
        if np.allclose(w, 0):
            w = np.ones_like(w)
        profile = np.asarray(self.X[m].T @ w).ravel()
        n = np.linalg.norm(profile)
        return np.asarray(self.X @ (profile / n)).ravel() if n > 0 else np.zeros(self.X.shape[0])


# --------------------------------------------------------------------------- hybrid
class HybridRecommender:
    """Weighted blend of z-scored model scores. Weights are tuned on a validation split."""
    name = "Hybrid"

    def __init__(self, models, weights):
        self.models, self.weights = models, weights

    def components(self, r):
        return {k: zscore(m.score(r)) for k, m in self.models.items()}

    def score(self, r):
        comps = self.components(r)
        return sum(self.weights.get(k, 0.0) * v for k, v in comps.items())


# --------------------------------------------------------------------------- diversity re-ranking
def mmr_rerank(scores, sim, candidates, n=10, lam=0.75):
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998): trade relevance for diversity."""
    cand = list(candidates)
    s = np.asarray(scores, dtype=float)
    lo, hi = s[cand].min(), s[cand].max()
    sn = (s - lo) / (hi - lo + 1e-12)
    chosen = []
    while cand and len(chosen) < n:
        if not chosen:
            best = max(cand, key=lambda c: sn[c])
        else:
            best = max(cand, key=lambda c: lam * sn[c] - (1 - lam) * sim[c, chosen].max())
        chosen.append(best)
        cand.remove(best)
    return chosen


def top_n(scores, r, n=10, allowed=None, boost=None, sim=None, lam=None):
    """Pick the top-n unrated, allowed items. If sim and lam are given, apply MMR diversification."""
    s = np.array(scores, dtype=float)
    if boost is not None:
        s = s + boost
    ok = np.isnan(r)
    if allowed is not None:
        ok &= allowed
    cand = np.where(ok)[0]
    if cand.size == 0:
        return []
    if sim is not None and lam is not None:
        return mmr_rerank(s, sim, cand, n, lam)
    return list(cand[np.argsort(-s[cand])][:n])
