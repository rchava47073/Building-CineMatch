"""CineMatch - conversational recommender with explanations, context awareness and proactive nudges.

Run interactively:   python app.py
Scripted demo:       python app.py --demo
Options:             --time morning|afternoon|evening|night   --no-proactive
"""
import argparse
import difflib
import json
import os
import re
import sys
from datetime import datetime
import numpy as np
from data_utils import load_data, genre_matrix
from models import (PopularityRecommender, ItemKNNRecommender, ALSRecommender,
                    ContentRecommender, HybridRecommender, top_n)

# Design assumptions (heuristics, not learned): context nudges the blended score in z-score units.
TIME_BOOST = {"morning": {"Comedy": 0.6, "Documentary": 0.6, "Animation": 0.4},
              "afternoon": {"Action": 0.5, "Comedy": 0.4, "Fantasy": 0.4},
              "evening": {"Drama": 0.6, "Thriller": 0.5, "Romance": 0.5, "Sci-Fi": 0.4},
              "night": {"Horror": 0.6, "Thriller": 0.5, "Sci-Fi": 0.4}}
FAMILY_BLOCK = {"Horror", "Thriller"}
SIGNAL_TEXT = {"pop": "it is widely watched and rated by other viewers",
               "knn": "viewers with ratings like yours enjoyed it",
               "als": "it fits the taste pattern in your ratings",
               "content": "its genres and themes match what you liked"}
HELP = """Things you can say:
  I like <title>  /  I dislike <title>  /  rate <title> 1-5      (use ; to add several)
  recommend  |  recommend 5 comedy  |  suggest sci-fi
  why <title>        explains a recommendation
  context evening | morning | afternoon | night | family | short | clear
  proactive on / off  (alias: quiet)      my ratings      reset      help      quit"""


def time_of_day(now=None):
    h = (now or datetime.now()).hour
    return "morning" if 5 <= h < 12 else "afternoon" if h < 17 else "evening" if h < 22 else "night"


class Assistant:
    def __init__(self, data_dir="data", results_dir="results", tod=None, proactive=True):
        self.R, self.items, self.source = load_data(data_dir)
        self.G, self.gnames = genre_matrix(self.items)
        cfg = {"weights": {"pop": 2, "knn": 0, "als": 2, "content": 1}, "knn_k": 20, "als_factors": 4}
        wp = os.path.join(results_dir, "weights.json")
        if os.path.exists(wp):
            cfg.update(json.load(open(wp)))
        mods = {"pop": PopularityRecommender().fit(self.R, self.items),
                "knn": ItemKNNRecommender(k=cfg["knn_k"]).fit(self.R, self.items),
                "als": ALSRecommender(factors=cfg["als_factors"]).fit(self.R, self.items),
                "content": ContentRecommender().fit(self.R, self.items)}
        self.hybrid = HybridRecommender(mods, cfg["weights"])
        self.sim = mods["content"].sim
        self.titles = list(self.items["title"])
        self.tod, self.family, self.short = tod or time_of_day(), False, False
        self.proactive, self.since_nudge = proactive, 0
        self.r = np.full(len(self.items), np.nan)

    # ---------------- helpers
    def find(self, text):
        t = text.strip().lower().strip("'\"")
        low = [s.lower() for s in self.titles]
        if t in low:
            return low.index(t), []
        sub = [i for i, s in enumerate(low) if t and t in s]
        if len(sub) == 1:
            return sub[0], []
        close = difflib.get_close_matches(t, low, n=3, cutoff=0.6)
        if len(close) >= 1 and (not sub):
            if difflib.SequenceMatcher(None, t, close[0]).ratio() > 0.8:
                return low.index(close[0]), []
        cands = [low.index(c) for c in close] or sub[:3]
        return None, cands

    def boost(self):
        b = np.zeros(len(self.items))
        for i, gs in enumerate(self.items["genres"]):
            parts = gs.split("|")
            b[i] += sum(TIME_BOOST[self.tod].get(g, 0) for g in parts)
        return b

    def allowed(self, genre=None):
        ok = np.ones(len(self.items), dtype=bool)
        for i, row in enumerate(self.items.itertuples()):
            gs = set(row.genres.split("|"))
            if self.family and gs & FAMILY_BLOCK:
                ok[i] = False
            if self.short and row.runtime > 95:
                ok[i] = False
            if genre and genre not in {g.lower() for g in gs}:
                ok[i] = False
        return ok

    def picks(self, n=5, genre=None, diversify=True):
        s = self.hybrid.score(self.r)
        return top_n(s, self.r, n, self.allowed(genre), self.boost(),
                     self.sim if diversify else None, 0.75 if diversify else None)

    def line(self, i):
        row = self.items.iloc[i]
        return f"{row.title} ({row.year}) [{row.genres.replace('|', ', ')}] {row.runtime} min"

    def explain(self, i):
        comps = self.hybrid.components(self.r)
        contrib = {k: self.hybrid.weights.get(k, 0) * v[i] for k, v in comps.items()}
        top = [k for k, _ in sorted(contrib.items(), key=lambda kv: -kv[1])[:2] if contrib[k] > 0]
        out = []
        liked = np.where(self.r >= 4)[0]
        if liked.size:
            j = liked[np.argmax(self.sim[i, liked])]
            shared = sorted(set(self.items.genres[i].split("|")) & set(self.items.genres[j].split("|")))
            out.append(f"Because you liked '{self.titles[j]}'"
                       + (f" (shared genres: {', '.join(shared)})" if shared else "") + f", similarity {self.sim[i, j]:.2f}.")
        if top:
            out.append("Main signals: " + "; ".join(SIGNAL_TEXT[k] for k in top) + ".")
        b = self.boost()[i]
        if b > 0:
            out.append(f"It also gets a small boost for {'an' if self.tod[0] in 'aeiou' else 'a'} {self.tod} viewing context.")
        pop = self.hybrid.models["pop"]
        out.append(f"Catalog stats: {int(pop.count[i])} ratings, average {pop.item_mean[i]:.2f}.")
        return " ".join(out)

    # ---------------- proactive behaviour
    def greeting(self):
        msg = f"CineMatch ready ({self.source} data, {len(self.items)} titles). Say 'help' for commands."
        if self.proactive:
            ids = self.picks(3)
            msg += (f"\nProactive tip: it's {self.tod}. Before you tell me anything, popular picks for this time of day are:\n  - "
                    + "\n  - ".join(self.line(i) for i in ids) + "\n(Say 'quiet' if you prefer I only answer when asked.)")
        return msg

    def nudge(self):
        if not (self.proactive and self.since_nudge >= 3):
            return ""
        self.since_nudge = 0
        ids = self.picks(3)
        return ("\n\nProactive tip: your taste is taking shape, so here are 3 titles you haven't rated:\n  - "
                + "\n  - ".join(self.line(i) for i in ids) + "\nSay 'why <title>' to see my reasoning, or 'quiet' to turn tips off.")

    # ---------------- dialogue
    def rate_titles(self, text, value):
        done, notes = [], []
        for part in [p for p in re.split(r";", text) if p.strip()]:
            i, cands = self.find(part)
            if i is None:
                hint = f" Did you mean: {', '.join(self.titles[c] for c in cands)}?" if cands else ""
                notes.append(f"I couldn't find '{part.strip()}'.{hint}")
            else:
                if np.isnan(self.r[i]):
                    self.since_nudge += 1
                self.r[i] = value
                done.append(f"{self.titles[i]} = {value:g}")
        msg = ("Saved: " + "; ".join(done) + ".") if done else ""
        return " ".join([msg] + notes).strip() + self.nudge()

    def handle(self, text):
        t = text.strip()
        low = t.lower()
        if low in ("help", "?"):
            return HELP
        if low in ("quit", "exit", "bye"):
            return "__quit__"
        if low == "reset":
            self.r[:] = np.nan
            self.since_nudge = 0
            return "Ratings cleared."
        if low in ("quiet", "proactive off"):
            self.proactive = False
            return "Okay, I'll only suggest things when you ask."
        if low == "proactive on":
            self.proactive = True
            return "Proactive tips are on."
        if low in ("my ratings", "profile"):
            idx = np.where(~np.isnan(self.r))[0]
            return "\n".join(f"  {self.titles[i]}: {self.r[i]:g}" for i in idx) or "No ratings yet."
        m = re.match(r"context\s+(\w+)", low)
        if m:
            c = m.group(1)
            if c in TIME_BOOST:
                self.tod = c
            elif c == "family":
                self.family = True
            elif c == "short":
                self.short = True
            elif c == "clear":
                self.family = self.short = False
            else:
                return "Context options: morning, afternoon, evening, night, family, short, clear."
            return f"Context set: time={self.tod}, family-friendly={self.family}, short-runtime={self.short}."
        m = re.match(r"(?:i\s+)?(?:really\s+)?(?:like|love|liked|loved)\s+(.+)", t, re.I)
        if m:
            return self.rate_titles(m.group(1), 5)
        m = re.match(r"(?:i\s+)?(?:dislike|hate|disliked|hated)\s+(.+)", t, re.I)
        if m:
            return self.rate_titles(m.group(1), 1)
        m = re.match(r"rate\s+(.+?)\s+([1-5](?:\.5)?)$", t, re.I)
        if m:
            return self.rate_titles(m.group(1), float(m.group(2)))
        m = re.match(r"why\s+(.+)", t, re.I)
        if m:
            i, cands = self.find(m.group(1))
            return self.explain(i) if i is not None else "Which title do you mean? " + ", ".join(self.titles[c] for c in cands)
        if re.match(r"(recommend|suggest|what should i watch)", low):
            n = int(re.search(r"\b(\d+)\b", low).group(1)) if re.search(r"\b(\d+)\b", low) else 5
            genre = next((g.lower() for g in self.gnames if g.lower() in low.replace("scifi", "sci-fi")), None)
            ids = self.picks(min(n, 15), genre)
            if not ids:
                return "Nothing matches those filters."
            head = "Top picks" + (f" in {genre}" if genre else "") + f" (context: {self.tod}" + (", family" if self.family else "") + (", short" if self.short else "") + "):"
            return head + "\n" + "\n".join(f"  {k + 1}. {self.line(i)}" for k, i in enumerate(ids))
        return "I didn't catch that. Say 'help' to see what I can do."


def demo(a):
    pop = a.hybrid.models["pop"].count
    def best(genre, k=0):
        ids = [i for i in np.argsort(-pop) if genre in a.items.genres[i].split("|")]
        return a.titles[ids[k]]
    script = [f"I like {best('Sci-Fi', 0)}", f"I like {best('Sci-Fi', 1)}; {best('Thriller', 0)}",
              f"I dislike {best('Romance', 0)}", "recommend 5", "__why__", "context family", "recommend 3",
              "context clear", "recommend 3 comedy", "quiet", f"I like {best('Drama', 0)}", "my ratings"]
    out = [a.greeting()]
    last = []
    for cmd in script:
        if cmd == "__why__":
            cmd = f"why {a.titles[last[0]]}"
        print_cmd = f"\n> {cmd}"
        out.append(print_cmd)
        reply = a.handle(cmd)
        out.append(reply)
        if cmd.startswith("recommend 5"):
            last = a.picks(5)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--time", choices=list(TIME_BOOST))
    ap.add_argument("--no-proactive", action="store_true")
    args = ap.parse_args()
    a = Assistant(tod=args.time, proactive=not args.no_proactive)
    if args.demo:
        text = demo(a)
        print(text)
        os.makedirs("results", exist_ok=True)
        open(os.path.join("results", "demo_transcript.txt"), "w").write(text + "\n")
        return
    print(a.greeting())
    while True:
        try:
            cmd = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        reply = a.handle(cmd)
        if reply == "__quit__":
            break
        print(reply)


if __name__ == "__main__":
    main()
