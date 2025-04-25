"""Helpers that turn raw parquet → model-ready tensors/data-frames."""
from __future__ import annotations
import logging, tempfile
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------------
from collections.abc import Sequence
import numpy as np

def _to_str_list(x) -> list[str]:
    """Return a Python list[str] regardless of the original container."""
    if x is None:
        return []
    if isinstance(x, str):
        return x.strip().split()            # original TSV string
    if isinstance(x, (list, tuple, np.ndarray, pd.Series, Sequence)):
        # Flatten any zero-dim ndarray and ensure str elements
        return [str(el) for el in list(x) if str(el)]
    # Fallback: single element
    return [str(x)]


# ---------------------------------------------------------------------------
# Two-Tower pipeline
# ---------------------------------------------------------------------------

def _parse_impressions(impressions: list[str]) -> list[tuple[str, int]]:
    out = []
    if impressions is None:
        return out
    for imp in impressions:
        try:
            nid, lbl = imp.split("-")
            out.append((nid, int(lbl)))
        except ValueError:
            log.warning("Bad impression %s – skipping", imp)
    return out


def _create_samples(df: pd.DataFrame) -> pd.DataFrame:
    df["parsed"] = df["impressions"].apply(_parse_impressions)
    s = df.explode("parsed").dropna(subset=["parsed"])
    s[["item_id", "label"]] = pd.DataFrame(s["parsed"].tolist(), index=s.index)
    return s[["user_id", "item_id", "label"]].copy()


def load_and_prepare_two_tower(data_dir: Path) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    beh = pd.read_parquet(data_dir / "train" / "behaviors.parquet")
    samples = _create_samples(beh)

    u_enc, i_enc = LabelEncoder(), LabelEncoder()
    samples["user_id"] = u_enc.fit_transform(samples["user_id"])
    samples["item_id"] = i_enc.fit_transform(samples["item_id"])
    return samples, u_enc, i_enc


# ---------------------------------------------------------------------------
# NRMS helpers – very lightweight tokeniser placeholder
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
PAD_ID    = 0                 # reserve 0 for the padding vector

def load_and_prepare_nrms(data_dir: Path,
                          max_news_len: int = 30,
                          max_hist_len: int = 50) -> tuple[
                              list[list[int]],          # candidate news tokens
                              list[list[list[int]]],    # history tokens
                              list[int],                # labels
                              int                       # vocab size
                          ]:
    """
    Parse **news.parquet** + **behaviors.parquet** and return fully-padded
    tensors ready for `NRMSDataset`.

    * Every news vector is right-padded / truncated to **max_news_len**.
    * Every history list is padded / truncated to **max_hist_len**.
    * Empty histories receive one PAD vector to keep shape.
    """
    news = pd.read_parquet(data_dir / "train" / "news.parquet")
    beh  = pd.read_parquet(data_dir / "train" / "behaviors.parquet")

    # -----------------------------------------------------------------------
    # 1. Build vocabulary and tokenise news titles
    # -----------------------------------------------------------------------
    vocab: dict[str, int] = {PAD_TOKEN: PAD_ID}

    def _index(word: str) -> int:
        if word not in vocab:
            vocab[word] = len(vocab)
        return vocab[word]

    def _tokenise_and_pad(text: str) -> list[int]:
        tokens = text.lower().split()[:max_news_len]
        ids    = [_index(tok) for tok in tokens]
        ids.extend([PAD_ID] * (max_news_len - len(ids)))   # right-pad
        return ids

    news["tokens"] = news["title"].apply(_tokenise_and_pad)
    nid2tok = dict(zip(news["id"], news["tokens"]))

    # -----------------------------------------------------------------------
    # 2. Build candidate, history, label lists (fully rectangular)
    # -----------------------------------------------------------------------
    cand, hist, lbls = [], [], []
    pad_vec = [PAD_ID] * max_news_len

    for _, row in beh.iterrows():
        # ---------- clicks --------------------------------------------------
        clicks_raw  = _to_str_list(row["history"])
        clicks_tok  = [nid2tok[c] for c in clicks_raw if c in nid2tok]

        # pad / truncate to exactly H = max_hist_len
        if len(clicks_tok) < max_hist_len:
            clicks_tok.extend([pad_vec] * (max_hist_len - len(clicks_tok)))
        else:
            clicks_tok = clicks_tok[:max_hist_len]

        # guarantee at least one valid vector
        if not clicks_tok:
            clicks_tok = [pad_vec] * max_hist_len

        # ---------- impressions --------------------------------------------
        for imp in _to_str_list(row["impressions"]):
            try:
                nid, lbl = imp.split("-")
            except ValueError:
                continue                      # malformed
            if nid not in nid2tok:
                continue                      # unseen news
            cand.append(nid2tok[nid])         # already padded length = L
            hist.append(clicks_tok)           # length = H × L
            lbls.append(int(lbl))

    log.info("NRMS data prepared | %d samples | vocab=%d",
             len(lbls), len(vocab))
    return cand, hist, lbls, len(vocab)