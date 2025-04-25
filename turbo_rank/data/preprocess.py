# ------------------------------------------------------------
# data/preprocess.py
# ------------------------------------------------------------
"""Light‑memory helpers that convert raw Parquet → disk‑backed numpy arrays
ready for NRMS / Two‑Tower training.

*No runtime Spark/JVM dependency.*  Use `scripts/convert_mind_to_parquet.py`
for the one‑off TSV → Parquet ETL step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

PAD_TOKEN = "<pad>"
PAD_ID = 0  # reserve 0 for padding


def _to_str_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return x.strip().split()
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [str(el) for el in x if str(el)]
    return [str(x)]


# ---------------------------------------------------------------------------
# Two‑Tower pipeline – small in‑memory helper
# ---------------------------------------------------------------------------


def load_and_prepare_two_tower(data_dir: Path) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    beh = pd.read_parquet(
        data_dir / "train" / "behaviors.parquet",
        columns=["user_id", "impressions"],
    )

    def _parse(imps: List[str]):
        out = []
        for imp in _to_str_list(imps):
            try:
                nid, lbl = imp.split("-")
                out.append((nid, int(lbl)))
            except ValueError:
                log.warning("Bad impression %s – skipping", imp)
        return out

    beh["parsed"] = beh["impressions"].apply(_parse)
    s = beh.explode("parsed").dropna(subset=["parsed"])
    s[["item_id", "label"]] = pd.DataFrame(s["parsed"].tolist(), index=s.index)

    u_enc, i_enc = LabelEncoder(), LabelEncoder()
    s["user_id"] = u_enc.fit_transform(s["user_id"])
    s["item_id"] = i_enc.fit_transform(s["item_id"])

    return s[["user_id", "item_id", "label"]].copy(), u_enc, i_enc


# ---------------------------------------------------------------------------
# NRMS helpers – disk‑backed variant
# ---------------------------------------------------------------------------


def _build_vocab_and_tokenise(news_df: pd.DataFrame, max_len: int):
    vocab = {PAD_TOKEN: PAD_ID}

    def _idx(tok: str):
        if tok not in vocab:
            vocab[tok] = len(vocab)
        return vocab[tok]

    def _tok(text: str):
        toks = text.lower().split()[:max_len]
        ids = [_idx(t) for t in toks]
        ids.extend([PAD_ID] * (max_len - len(ids)))
        return ids

    news_df["tokens"] = news_df["title"].astype(str).apply(_tok)
    return vocab


def load_and_prepare_nrms(
    data_dir: Path,
    *,
    split: str = "train",          # ← NEW
    max_news_len: int = 30,
    max_hist_len: int = 50,
    cache_dir: Path | None = None,
    regen: bool = False,
):
    """
    Return (cand_np, hist_np, lbl_np, vocab_size) for a single split
    (`train`, `dev`, or `test`).

    The per-split cache lives in  .../<split>/cache/.
    """
    data_dir  = Path(data_dir)
    cache_dir = cache_dir or (data_dir / split / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cand_f = cache_dir / f"cand_{max_news_len}_{max_hist_len}.npy"
    hist_f = cache_dir / f"hist_{max_news_len}_{max_hist_len}.npy"
    lbl_f  = cache_dir / "lbl.npy"
    vocab_f = cache_dir / "vocab.json"

    if not regen and cand_f.exists() and hist_f.exists() and lbl_f.exists():
        cand = np.load(cand_f, mmap_mode="r")
        hist = np.load(hist_f, mmap_mode="r")
        lbls = np.load(lbl_f, mmap_mode="r")
        vocab_size = json.loads(vocab_f.read_text())["size"]
        log.info("NRMS cache hit | %d samples | vocab=%d", len(lbls), vocab_size)
        return cand, hist, lbls, vocab_size

    news = pd.read_parquet(data_dir / split / "news.parquet", columns=["id", "title"])
    beh  = pd.read_parquet(data_dir / split / "behaviors.parquet",
                           columns=["history", "impressions"])

    vocab = _build_vocab_and_tokenise(news, max_news_len)
    nid2tok = dict(zip(news["id"], news["tokens"]))

    pad_vec = [PAD_ID] * max_news_len
    cand_lst, hist_lst, lbl_lst = [], [], []

    for _, row in beh.iterrows():
        clicks = [nid2tok[n] for n in _to_str_list(row["history"]) if n in nid2tok]
        clicks = (clicks + [pad_vec] * max_hist_len)[:max_hist_len]
        if not clicks:
            clicks = [pad_vec] * max_hist_len

        for imp in _to_str_list(row["impressions"]):
            try:
                nid, lbl = imp.split("-")
            except ValueError:
                continue
            if nid not in nid2tok:
                continue
            cand_lst.append(nid2tok[nid])
            hist_lst.append(clicks)
            lbl_lst.append(int(lbl))

    cand_np = np.asarray(cand_lst, dtype=np.int32)
    hist_np = np.asarray(hist_lst, dtype=np.int32)
    lbl_np = np.asarray(lbl_lst, dtype=np.int8)

    np.save(cand_f, cand_np)  # will actually write cand_f + ".npy" if no suffix
    np.save(hist_f, hist_np)
    np.save(lbl_f, lbl_np)
    vocab_f.write_text(json.dumps({"size": len(vocab)}))

    log.info("NRMS cache rebuilt | %d samples | vocab=%d", len(lbl_np), len(vocab))
    return (
        np.memmap(cand_f, dtype=np.int32, mode="r", shape=cand_np.shape),
        np.memmap(hist_f, dtype=np.int32, mode="r", shape=hist_np.shape),
        np.memmap(lbl_f, dtype=np.int8, mode="r", shape=lbl_np.shape),
        len(vocab),
    )
