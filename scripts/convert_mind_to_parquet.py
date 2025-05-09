# ------------------------------------------------------------
# scripts/convert_mind_to_parquet.py
# ------------------------------------------------------------
"""Cluster‑scale TSV → Parquet converter for the MIND dataset.

Usage
-----
$ PYSPARK_PYTHON=python3 spark-submit scripts/convert_mind_to_parquet.py \
      --input-dir data/mind --output-dir data/processed/large --size large --splits train,dev

Only needs to be run once after downloading the raw .zip files. The training
code never touches Spark.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, split, when
from pyspark.sql.types import StringType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

NEWS_COLS = [
    "id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEH_COLS = ["impression_id", "user_id", "timestamp", "history", "impressions"]


def parse_args():
    p = argparse.ArgumentParser(description="Convert raw MIND TSV to Parquet using Spark")
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder that contains MINDsmall_train/news.tsv etc.",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Destination root for Parquet")
    p.add_argument("--size", choices=["small", "large"], default="small", help="MIND release size")
    p.add_argument(
        "--splits",
        default="train",
        help="Comma‑separated list of dataset splits to process (train,dev,test)",
    )
    return p.parse_args()


def spark_session(app_name: str = "MIND‑TSV‑to‑Parquet") -> SparkSession:
    log.info("Starting Spark session")
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.parquet.writeLegacyFormat", "true")
        .getOrCreate()
    )


def load_tsv(spark: SparkSession, path: Path, schema: List[str]) -> DataFrame:
    log.info("Loading %s", path)
    return spark.read.option("delimiter", "\t").csv(str(path)).toDF(*schema)


def preprocess_news(df: DataFrame) -> DataFrame:
    return df  # placeholder for future NLP steps


def preprocess_behaviors(df: DataFrame) -> DataFrame:
    df = df.withColumn(
        "history",
        when(col("history").isNull(), lit(None).cast(StringType())).otherwise(col("history")),
    )
    return df.withColumn("history", split(col("history"), " ")).withColumn(
        "impressions", split(col("impressions"), " ")
    )


def save(df: DataFrame, path: Path):
    log.info("Writing %s", path)
    df.write.mode("overwrite").parquet(str(path))


def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",")]
    spark = spark_session()

    try:
        for split in splits:
            log.info("--- Processing %s split ---", split)
            news_tsv = args.input_dir / f"MIND{args.size}_{split}" / "news.tsv"
            beh_tsv = args.input_dir / f"MIND{args.size}_{split}" / "behaviors.tsv"

            news_df = preprocess_news(load_tsv(spark, news_tsv, NEWS_COLS))
            beh_df = preprocess_behaviors(load_tsv(spark, beh_tsv, BEH_COLS))

            save(news_df, args.output_dir / split / "news.parquet")
            save(beh_df, args.output_dir / split / "behaviors.parquet")

        log.info("All splits complete")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
