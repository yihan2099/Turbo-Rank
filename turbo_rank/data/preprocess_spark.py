from __future__ import annotations

import json, logging, os
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession, functions as F, types as T

PAD_TOKEN = "<pad>"
PAD_ID    = 0          # reserve 0 for padding
log       = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def _spark() -> SparkSession:
    """Create / get a SparkSession with Arrow enabled for fast collects."""
    return (
        SparkSession.builder.appName("NRMS-Preprocess")
        .config("spark.sql.execution.arrow.enabled", "true")
        .getOrCreate()
    )


def _tokenise(title: str) -> list[str]:
    return [w.lower() for w in title.split() if w.strip()]


tokenise_udf = F.udf(_tokenise, T.ArrayType(T.StringType()))


def _build_vocab_and_tokenise_spark(news_df, max_news_len: int):
    """
    Returns:
        news_tok_df : news_id | tokens (padded → list[int])
        vocab_dict  : token → int
    """
    spark = news_df.sparkSession

    # tokenise & explode to build vocab
    tokens_df = (
        news_df.select("id", tokenise_udf("title").alias("tok"))
        .withColumn("tok", F.slice(F.col("tok"), 1, max_news_len))  # truncate
        .cache()
    )

    vocab_df = tokens_df.select(F.explode("tok").alias("w")).distinct()
    vocab_dict = (
        vocab_df.rdd.map(lambda r: r["w"])
        .zipWithIndex()
        .map(lambda x: (x[0], x[1] + 1))  # 0 is PAD
        .collectAsMap()
    )
    b_vocab = spark.sparkContext.broadcast(vocab_dict)

    # map tokens → ints & pad
    def tok2id(tok_list):
        arr = [b_vocab.value.get(t, PAD_ID) for t in tok_list][:max_news_len]
        return arr + [PAD_ID] * (max_news_len - len(arr))

    tok2id_udf = F.udf(tok2id, T.ArrayType(T.IntegerType()))
    news_tok_df = tokens_df.select(
        "id", tok2id_udf("tok").alias("tokens")
    ).cache()

    return news_tok_df, vocab_dict


def _pad_hist(clicks, max_hist_len, pad_vec):
    clks = clicks[:max_hist_len]
    return clks + [pad_vec] * (max_hist_len - len(clks))


# ──────────────────────────────────────────────────────────────
#  Public API (drop-in replacement)
# ──────────────────────────────────────────────────────────────
def load_and_prepare_nrms(
    data_dir: Path,
    *,
    split: str = "train",
    max_news_len: int = 30,
    max_hist_len: int = 50,
    cache_dir: Path | None = None,
    regen: bool = False,
):
    """
    Spark version – identical contract & artefacts to the pandas version.
    """
    data_dir  = Path(data_dir)
    cache_dir = cache_dir or (data_dir / split / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cand_f  = cache_dir / f"cand_{max_news_len}_{max_hist_len}.npy"
    hist_f  = cache_dir / f"hist_{max_news_len}_{max_hist_len}.npy"
    lbl_f   = cache_dir / "lbl.npy"
    vocab_f = cache_dir / "vocab.json"

    if (
        not regen
        and cand_f.exists()
        and hist_f.exists()
        and lbl_f.exists()
        and vocab_f.exists()
    ):
        cand = np.load(cand_f, mmap_mode="r")
        hist = np.load(hist_f, mmap_mode="r")
        lbls = np.load(lbl_f,  mmap_mode="r")
        vocab_size = json.loads(vocab_f.read_text())["size"]
        log.info("NRMS Spark cache hit | %d samples | vocab=%d", len(lbls), vocab_size)
        return cand, hist, lbls, vocab_size

    spark = _spark()
    spark.conf.set("spark.sql.shuffle.partitions", os.cpu_count() or 8)

    news_df = spark.read.parquet(str(data_dir / split / "news.parquet")).select("id", "title")
    beh_df  = spark.read.parquet(str(data_dir / split / "behaviors.parquet")).select(
        "history", "impressions"
    )

    # ------------------------------------------------------------------
    # 1. vocabulary & tokenised news
    # ------------------------------------------------------------------
    news_tok_df, vocab = _build_vocab_and_tokenise_spark(news_df, max_news_len)
    pad_vec   = [PAD_ID] * max_news_len
    b_news    = spark.sparkContext.broadcast(
        {row["id"]: row["tokens"] for row in news_tok_df.collect()}
    )

    # ------------------------------------------------------------------
    # 2. expand behaviors → (cand, hist, lbl)
    # ------------------------------------------------------------------
    def parse_beh(row):
        _to_list = lambda x: [] if not x else str(x).split()
        hist_ids = [nid for nid in _to_list(row.history) if nid in b_news.value]

        hist = _pad_hist([b_news.value[n] for n in hist_ids], max_hist_len, pad_vec)

        for imp in _to_list(row.impressions):
            try:
                nid, lbl = imp.split("-")
                if nid not in b_news.value:
                    continue
                yield (
                    b_news.value[nid],  # cand
                    hist,               # hist
                    int(lbl),           # lbl
                )
            except ValueError:
                continue

    schema = T.StructType(
        [
            T.StructField("cand", T.ArrayType(T.IntegerType()), False),
            T.StructField("hist", T.ArrayType(T.ArrayType(T.IntegerType())), False),
            T.StructField("lbl",  T.IntegerType(), False),
        ]
    )

    samples_df = beh_df.rdd.flatMap(parse_beh).toDF(schema).repartition(64).cache()
    n_samples  = samples_df.count()

    # ------------------------------------------------------------------
    # 3. save to .npy in chunks → concatenate
    # ------------------------------------------------------------------
    tmp_dir = cache_dir / "_spark_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def save_partition(idx, iter_rows):
        cand, hist, lbl = [], [], []
        for r in iter_rows:
            cand.append(r.cand)
            hist.append(r.hist)
            lbl.append(r.lbl)
        if cand:
            np.save(tmp_dir / f"cand_{idx}.npy", np.asarray(cand, dtype=np.int32))
            np.save(tmp_dir / f"hist_{idx}.npy", np.asarray(hist, dtype=np.int32))
            np.save(tmp_dir / f"lbl_{idx}.npy",  np.asarray(lbl,  dtype=np.int8))
        yield 1  # dummy

    samples_df.rdd.mapPartitionsWithIndex(save_partition).count()
    spark.stop()

    # merge partition files → final contiguous arrays
    def _cat_and_save(prefix, dtype, out_f):
        parts = sorted((tmp_dir).glob(f"{prefix}_*.npy"))
        arrs  = [np.load(p, mmap_mode="r") for p in parts]
        cat   = np.concatenate(arrs, axis=0).astype(dtype, copy=False)
        np.save(out_f, cat, allow_pickle=False)
        for p in parts:
            p.unlink()

    _cat_and_save("cand", np.int32, cand_f)
    _cat_and_save("hist", np.int32, hist_f)
    _cat_and_save("lbl",  np.int8,  lbl_f)
    tmp_dir.rmdir()

    vocab_f.write_text(json.dumps({"size": len(vocab)}, ensure_ascii=False))

    log.info("NRMS Spark cache built | %d samples | vocab=%d", n_samples, len(vocab))

    # return memmaps
    cand_np = np.memmap(cand_f, dtype=np.int32, mode="r")
    hist_np = np.memmap(hist_f, dtype=np.int32, mode="r").reshape(
        (n_samples, max_hist_len, max_news_len)
    )
    lbl_np  = np.memmap(lbl_f,  dtype=np.int8,  mode="r")
    return cand_np, hist_np, lbl_np, len(vocab)