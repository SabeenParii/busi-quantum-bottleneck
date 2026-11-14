"""Data preparation (protocol Section 2): BUSI cleaning, duplicate groups, folds, limited-data subsets, arrays for
BUSI, BUS-BRA and BrEaST. Images are never written outside the user's Drive."""
import glob
import hashlib
import os
import re

import numpy as np
import pandas as pd
from PIL import Image

SIZE = 256
NAME = re.compile(r"(benign|malignant|normal)\s*\((\d+)\)", re.I)


# ----------------------------------------------------------------------------- image helpers
def load_square(path, mask=False):
    im = Image.open(path)
    im = im.convert("L")
    a = np.asarray(im)
    h, w = a.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    out = Image.fromarray(sq).resize((SIZE, SIZE), Image.NEAREST if mask else Image.BILINEAR)
    a = np.asarray(out)
    return (a >= 128).astype(np.uint8) * 255 if mask else a


def dhash(path):
    a = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- BUSI
def busi_files(root):
    """DataFrame image, cls, path, masks (list) for every BUSI image."""
    rows = []
    for cls in ("benign", "malignant", "normal"):
        cand = [p for p in glob.glob(os.path.join(root, "**", cls, "*.png"), recursive=True)]
        imgs = sorted([p for p in cand if "_mask" not in os.path.basename(p)],
                      key=lambda p: int(NAME.search(os.path.basename(p)).group(2)))
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            masks = sorted(q for q in cand if os.path.basename(q).startswith(stem + "_mask"))
            rows.append(dict(image=stem, cls=cls, path=p, masks=masks))
    return pd.DataFrame(rows)


def norm_name(cls, num):
    return f"{cls.lower()} ({int(num)})"


def parse_issue_list(files):
    """Pawlowska et al. supplementary list. Returns (dup_groups: list of sets of image names, flags: dict name -> set).
    Every row with two or more image names is a duplicate group. Keyword flags: axilla, needle, questionable, annotation."""
    groups, flags = [], {}
    for f in files:
        if f.lower().endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(f, sheet_name=None, dtype=str)
        elif f.lower().endswith(".csv"):
            try:
                sheets = {"csv": pd.read_csv(f, dtype=str, sep=None, engine="python")}
            except Exception:
                sheets = {"csv": pd.read_csv(f, dtype=str, encoding="latin1", sep=None, engine="python")}
        else:
            continue
        for sh, df in sheets.items():
            cols = [str(c).lower() for c in df.columns]
            for _, row in df.fillna("").iterrows():
                cells = [str(v) for v in row.values]
                names = sorted({norm_name(*m) for c in cells for m in NAME.findall(c)})
                if len(names) >= 2:
                    groups.append(set(names))
                text = " ".join(cells).lower()
                obj = " ".join(str(row.iloc[i]).lower() for i, c in enumerate(cols) if "object" in c) or text
                ann = " ".join(str(row.iloc[i]).lower() for i, c in enumerate(cols) if "annot" in c).strip()
                fl = set()
                if "axill" in obj:
                    fl.add("axilla")
                if "needle" in obj:
                    fl.add("needle")
                if any(w in obj for w in ("questionab", "doubt", "uncertain", "wrong class", "misclass")):
                    fl.add("questionable")
                if ann and ann not in ("no", "none", "-", "0", "nan"):
                    fl.add("annotation")
                for nm in names:
                    flags.setdefault(nm, set()).update(fl)
    return groups, flags


class UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb, key=order_key)] = min(ra, rb, key=order_key)


def order_key(name):
    cls, num = NAME.search(name).groups()
    return ("benign", "malignant", "normal").index(cls.lower()), int(num)


def clean_busi(files, groups, flags, hash_thr=4):
    """Apply protocol 2.1. Returns manifest of all images with columns keep, reason, group, annotated, hash_pair."""
    df = files.copy()
    names = list(df.image)
    uf = UF(names)
    for g in groups:
        g = [x for x in g if x in uf.p]
        for x in g[1:]:
            uf.union(g[0], x)
    df["group"] = [uf.find(x) for x in names]
    # hash audit
    h = {r.image: dhash(r.path) for r in df.itertuples()}
    extra = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if bin(h[a] ^ h[b]).count("1") <= hash_thr and uf.find(a) != uf.find(b):
                extra.append((a, b))
    df["reason"] = ""
    cls_of_group = df.groupby("group").cls.agg(lambda s: set(s))
    df.loc[df.group.map(lambda g: len(cls_of_group[g]) > 1), "reason"] = "cross-class duplicate"
    first = df.groupby("group").image.transform(lambda s: min(s, key=order_key))
    df.loc[(df.reason == "") & (df.image != first), "reason"] = "duplicate (kept another copy)"
    for fl in ("axilla", "needle", "questionable"):
        df.loc[(df.reason == "") & df.image.map(lambda x: fl in flags.get(x, set())), "reason"] = fl
    df["annotated"] = df.image.map(lambda x: "annotation" in flags.get(x, set())).astype(int)
    df["keep"] = (df.reason == "").astype(int)
    return df, extra


def assign_folds(man, extra_pairs, seed=2027):
    """Stratified 5-fold assignment of kept lesion images (benign, malignant); hash pairs kept in one fold."""
    k = man[(man.keep == 1) & (man.cls != "normal")].copy()
    uf = UF(list(k.image))
    for a, b in extra_pairs:
        if a in uf.p and b in uf.p:
            uf.union(a, b)
    k["unit"] = [uf.find(x) for x in k.image]
    rng = np.random.default_rng(seed)
    fold_of = {}
    for cls in ("benign", "malignant"):
        units = sorted(k[k.cls == cls].unit.unique(), key=order_key)
        units = [units[i] for i in rng.permutation(len(units))]
        for i, u in enumerate(units):
            fold_of[u] = i % 5
    k["fold"] = k.unit.map(fold_of)
    # nested limited-data subsets per fold (stratified, seed 2027)
    for f in range(5):
        tr = k[~k.fold.isin([f, (f + 1) % 5])]
        order = {}
        for cls in ("benign", "malignant"):
            ids = sorted(tr[tr.cls == cls].image, key=order_key)
            perm = np.random.default_rng(seed + 10 * f + (cls == "malignant")).permutation(len(ids))
            for rank, j in enumerate(perm):
                order[ids[j]] = (rank + 0.5) / len(ids)
        for p in (10, 25, 50):
            k[f"sub{p:03d}_f{f}"] = k.image.map(lambda x: int(x in order and order[x] < p / 100))
    return k.sort_values("image", key=lambda s: s.map(order_key)).reset_index(drop=True)


def arrays(paths, mask_lists):
    X = np.stack([load_square(p) for p in paths])
    Y = np.stack([np.max(np.stack([load_square(m, mask=True) for m in ms]), axis=0) if len(ms) else
                  np.zeros((SIZE, SIZE), np.uint8) for ms in mask_lists])
    return X, Y


# ----------------------------------------------------------------------------- external sets
def busbra(root):
    """BUS-BRA: images bus_XXXX-{l,r,s}.png, masks mask_XXXX-*.png; patient = Case column of bus_data.csv
    (falls back to the number in the file name)."""
    imgs = sorted(glob.glob(os.path.join(root, "**", "bus_*.png"), recursive=True))
    masks = {os.path.basename(p).replace("mask_", "bus_"): p for p in glob.glob(os.path.join(root, "**", "mask_*.png"), recursive=True)}
    meta = None
    for c in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        t = pd.read_csv(c)
        if "ID" in t.columns:
            meta = t.set_index("ID")
    rows = []
    for p in imgs:
        b = os.path.basename(p)
        if b not in masks:
            continue
        iid = os.path.splitext(b)[0]
        pat = None
        if meta is not None and iid in meta.index:
            for col in ("Case", "case", "Patient", "patient"):
                if col in meta.columns:
                    pat = str(meta.loc[iid, col])
        if pat is None:
            pat = re.findall(r"\d+", iid)[0]
        rows.append(dict(image=iid, patient=pat, path=p, masks=[masks[b]]))
    return pd.DataFrame(rows)


def breast(root):
    """BrEaST-Lesions_USG: caseXXX.png with caseXXX_tumor.png masks; images without a tumor mask are excluded."""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "**", "case*.png"), recursive=True)):
        b = os.path.basename(p)
        if "_" in os.path.splitext(b)[0]:
            continue
        stem = os.path.splitext(b)[0]
        m = glob.glob(os.path.join(os.path.dirname(p), stem + "_tumor*.png"))
        if m:
            rows.append(dict(image=stem, patient=stem, path=p, masks=sorted(m)))
    return pd.DataFrame(rows)
