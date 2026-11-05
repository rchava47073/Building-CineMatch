"""Data loading for CineMatch.

Two sources are supported:

1. MovieLens-format files (ratings.csv + movies.csv) placed in ./data  -> used automatically.
2. A reproducible synthetic MovieLens-style dataset (fictional titles) that is generated when no
   MovieLens files are present, so the project runs offline with zero downloads.

Both paths return the same objects: a dense ratings matrix R (users x items, NaN = unrated)
and an items DataFrame (title, genres, description, runtime, year).
"""
import os
import numpy as np
import pandas as pd

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance",
          "Thriller", "Animation", "Documentary", "Horror", "Fantasy"]

KEYWORDS = {
    "Action": ["chase", "explosion", "mercenary", "heist", "rescue", "fortress", "duel", "pursuit"],
    "Comedy": ["mixup", "roommates", "wedding", "prank", "awkward", "road-trip", "rivals", "farce"],
    "Drama": ["family", "grief", "redemption", "courtroom", "inheritance", "secrets", "struggle", "reunion"],
    "Sci-Fi": ["starship", "android", "colony", "time-loop", "quantum", "signal", "orbit", "wormhole"],
    "Romance": ["letters", "reunion", "summer", "longing", "ballroom", "proposal", "heartbreak", "destiny"],
    "Thriller": ["conspiracy", "witness", "blackmail", "stalker", "alibi", "manhunt", "twist", "paranoia"],
    "Animation": ["adventure", "talking-animals", "friendship", "magic-school", "quest", "toys", "dragon", "festival"],
    "Documentary": ["expedition", "interviews", "archive", "ecosystem", "history", "investigation", "portrait", "survey"],
    "Horror": ["haunted", "cabin", "curse", "ritual", "creature", "nightmare", "asylum", "possession"],
    "Fantasy": ["kingdom", "sorcerer", "prophecy", "relic", "elves", "enchanted", "dragon", "realm"],
}
ADJ = ["Silent", "Crimson", "Hidden", "Last", "Broken", "Golden", "Distant", "Midnight", "Electric", "Hollow",
       "Wandering", "Frozen", "Burning", "Secret", "Lonely", "Iron", "Paper", "Glass", "Velvet", "Savage",
       "Gentle", "Restless", "Pale", "Neon", "Ancient", "Final", "Wild", "Quiet", "Scarlet", "Sunken"]
NOUN = ["Harbor", "Empire", "Garden", "Signal", "Horizon", "Letter", "Orchard", "Mirror", "Voyage", "Archive",
        "Frontier", "Lantern", "Summit", "Tide", "Kingdom", "Carnival", "Engine", "Promise", "Shadow", "Bridge",
        "Station", "Canyon", "Winter", "Compass", "Ledger", "Harvest", "Atlas", "Echo", "Citadel", "Meridian"]


def make_synthetic(n_users=300, n_items=150, seed=42):
    """Generate a MovieLens-style dataset with latent genre taste, item quality, popularity skew
    and selection bias (people tend to watch what they expect to like)."""
    rng = np.random.default_rng(seed)
    g, d = len(GENRES), 5
    F = rng.normal(0, 1, (g, d))

    # ---- items
    titles, seen = [], set()
    while len(titles) < n_items:
        t = f"{rng.choice(['The ', ''])}{rng.choice(ADJ)} {rng.choice(NOUN)}".strip()
        if t not in seen:
            seen.add(t)
            titles.append(t)
    gw = np.array([1.4, 1.3, 1.6, 0.9, 0.9, 1.0, 0.7, 0.5, 0.7, 0.8])
    gw = gw / gw.sum()
    multi = np.zeros((n_items, g))
    for i in range(n_items):
        k = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20])
        multi[i, rng.choice(g, k, replace=False, p=gw)] = 1
    item_vec = multi @ F / np.sqrt(multi.sum(1, keepdims=True)) + 0.3 * rng.normal(size=(n_items, d))
    quality = rng.normal(0, 0.5, n_items)
    descs = []
    for i in range(n_items):
        words = []
        for gi in np.where(multi[i] > 0)[0]:
            words += list(rng.choice(KEYWORDS[GENRES[gi]], 4, replace=False))
        descs.append(" ".join(words))
    items = pd.DataFrame({
        "item_idx": np.arange(n_items),
        "title": titles,
        "genres": ["|".join(GENRES[j] for j in np.where(multi[i] > 0)[0]) for i in range(n_items)],
        "description": descs,
        "runtime": np.clip(rng.normal(105, 20, n_items), 70, 180).round().astype(int),
        "year": rng.integers(1990, 2025, n_items),
    })

    # ---- users
    user_vec = np.zeros((n_users, d))
    for u in range(n_users):
        fav = rng.choice(g, rng.choice([2, 3]), replace=False, p=gw)
        user_vec[u] = F[fav].sum(0) / np.sqrt(len(fav)) + 0.3 * rng.normal(size=d)
    user_bias = rng.normal(0, 0.3, n_users)
    aff = user_vec @ item_vec.T
    aff = (aff - aff.mean()) / aff.std()

    raw = 3.3 + 0.9 * aff + quality[None, :] + user_bias[:, None] + rng.normal(0, 0.5, aff.shape)
    rating = np.clip(np.round(raw), 1, 5)

    pop = rng.lognormal(0, 1.0, n_items)
    act = rng.lognormal(0, 0.5, n_users)
    p_obs = np.minimum(0.9, 0.055 * act[:, None] * pop[None, :] * np.exp(0.5 * aff))
    obs = rng.random((n_users, n_items)) < p_obs
    for u in range(n_users):                      # guarantee >= 12 ratings per user
        while obs[u].sum() < 12:
            obs[u, rng.choice(n_items, p=np.exp(aff[u]) / np.exp(aff[u]).sum())] = True
    for i in range(n_items):                      # guarantee >= 6 ratings per item
        while obs[:, i].sum() < 6:
            obs[rng.integers(n_users), i] = True
    R = np.where(obs, rating, np.nan)
    return R, items


def _from_movielens(data_dir, max_items=1500):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    top = ratings["movieId"].value_counts().head(max_items).index
    ratings = ratings[ratings["movieId"].isin(top)]
    movies = movies[movies["movieId"].isin(top)].reset_index(drop=True)
    uid = {u: k for k, u in enumerate(sorted(ratings["userId"].unique()))}
    iid = {m: k for k, m in enumerate(movies["movieId"])}
    R = np.full((len(uid), len(iid)), np.nan)
    for u, m, r in ratings[["userId", "movieId", "rating"]].itertuples(index=False):
        R[uid[u], iid[m]] = r
    items = pd.DataFrame({
        "item_idx": np.arange(len(movies)),
        "title": movies["title"],
        "genres": movies["genres"].str.replace("(no genres listed)", "Unknown", regex=False),
        "description": movies["genres"].str.replace("|", " ", regex=False).str.lower(),
        "runtime": 105,
        "year": movies["title"].str.extract(r"\((\d{4})\)")[0].fillna(2000).astype(int),
    })
    return R, items


def load_data(data_dir="data", force_synthetic=False):
    """Return (R, items, source_name)."""
    ml_ok = all(os.path.exists(os.path.join(data_dir, f)) for f in ("ratings.csv", "movies.csv"))
    if ml_ok and not force_synthetic:
        R, items = _from_movielens(data_dir)
        return R, items, "MovieLens"
    R, items = make_synthetic()
    os.makedirs(data_dir, exist_ok=True)
    users, its = np.where(~np.isnan(R))
    pd.DataFrame({"user_idx": users, "item_idx": its, "rating": R[users, its]}).to_csv(
        os.path.join(data_dir, "synthetic_ratings.csv"), index=False)
    items.to_csv(os.path.join(data_dir, "synthetic_movies.csv"), index=False)
    return R, items, "synthetic"


def genre_matrix(items):
    """Multi-hot genre matrix (items x genres) and genre name list."""
    names = sorted({g for s in items["genres"] for g in s.split("|")})
    G = np.zeros((len(items), len(names)))
    for i, s in enumerate(items["genres"]):
        for g in s.split("|"):
            G[i, names.index(g)] = 1
    return G, names
