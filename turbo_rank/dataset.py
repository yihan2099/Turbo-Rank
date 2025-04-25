import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, split, when
from pyspark.sql.types import StringType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration for file paths (relative to project root)
PATHS = {
    "news_train_tsv": "data/mind/MINDsmall_train/news.tsv",
    "behaviors_train_tsv": "data/mind/MINDsmall_train/behaviors.tsv",
    "news_dev_tsv": "data/mind/MINDsmall_dev/news.tsv",
    "behaviors_dev_tsv": "data/mind/MINDsmall_dev/behaviors.tsv",
    "processed_news_train": "data/processed/train/news.parquet",
    "processed_behaviors_train": "data/processed/train/behaviors.parquet",
    "processed_news_dev": "data/processed/dev/news.parquet",
    "processed_behaviors_dev": "data/processed/dev/behaviors.parquet",
}

NEWS_COLUMNS = [
    "id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEHAVIORS_COLUMNS = ["impression_id", "user_id", "timestamp", "history", "impressions"]


def create_spark_session(app_name: str = "MIND Preprocessing") -> SparkSession:
    """Creates and returns a Spark session."""
    logging.info(f"Starting Spark session: {app_name}")
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.parquet.writeLegacyFormat", "true")  # Optional: for compatibility
        .getOrCreate()
    )
    return spark


def load_tsv(spark: SparkSession, path: str, schema: list[str]) -> DataFrame:
    """Loads a TSV file into a Spark DataFrame with specified column names."""
    logging.info(f"Loading TSV file: {path}")
    df = spark.read.option("delimiter", "\t").csv(path)
    df = df.toDF(*schema)
    return df


def preprocess_news(df: DataFrame) -> DataFrame:
    """Basic preprocessing for news data (currently just returns input)."""
    logging.info("Preprocessing news data...")
    # Add any news-specific preprocessing here if needed in the future
    return df


def preprocess_behaviors(df: DataFrame) -> DataFrame:
    """Preprocesses behaviors data: handles null history and splits history/impressions."""
    logging.info("Preprocessing behaviors data...")
    # Handle potential nulls in history before splitting
    df = df.withColumn(
        "history",
        when(col("history").isNull(), lit(None).cast(StringType())).otherwise(col("history")),
    )
    # Split history and impressions strings into arrays
    df = df.withColumn("history", split(col("history"), " "))
    df = df.withColumn("impressions", split(col("impressions"), " "))
    return df


def save_parquet(df: DataFrame, path: str):
    """Saves a DataFrame to Parquet format."""
    logging.info(f"Saving DataFrame to Parquet: {path}")
    df.write.mode("overwrite").parquet(path)


def main():
    spark = create_spark_session()

    try:
        # Process Training Data
        logging.info("--- Processing Training Data ---")
        news_train_df = load_tsv(spark, PATHS["news_train_tsv"], NEWS_COLUMNS)
        behaviors_train_df = load_tsv(spark, PATHS["behaviors_train_tsv"], BEHAVIORS_COLUMNS)

        processed_news_train_df = preprocess_news(news_train_df)
        processed_behaviors_train_df = preprocess_behaviors(behaviors_train_df)

        save_parquet(processed_news_train_df, PATHS["processed_news_train"])
        save_parquet(processed_behaviors_train_df, PATHS["processed_behaviors_train"])

        # Process Dev Data (Optional - can be added similarly)
        # logging.info("--- Processing Dev Data ---")
        # news_dev_df = load_tsv(spark, PATHS["news_dev_tsv"], NEWS_COLUMNS)
        # behaviors_dev_df = load_tsv(spark, PATHS["behaviors_dev_tsv"], BEHAVIORS_COLUMNS)
        # processed_news_dev_df = preprocess_news(news_dev_df)
        # processed_behaviors_dev_df = preprocess_behaviors(behaviors_dev_df)
        # save_parquet(processed_news_dev_df, PATHS["processed_news_dev"])
        # save_parquet(processed_behaviors_dev_df, PATHS["processed_behaviors_dev"])

        logging.info("Preprocessing complete.")

    finally:
        logging.info("Stopping Spark session.")
        spark.stop()


if __name__ == "__main__":
    main()
