from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round as spark_round

def initialize_spark_session(app_name="Binge_Watching_Insights"):
    """
    Initialize and return a SparkSession instance.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data_from_csv(spark, file_path):
    """
    Read movie ratings data from a CSV file and load it into a Spark DataFrame.
    """
    return spark.read.option("header", True).option("inferSchema", True).csv(file_path)

def compute_binge_watching_statistics(df):
    """
    Determine the percentage of users in each age group who engage in binge-watching.
    """
    binge_users = df.filter(col("IsBingeWatched") == True)
    binge_counts = binge_users.groupBy("AgeGroup").agg(count("UserID").alias("Binge_Watchers"))
    total_users = df.groupBy("AgeGroup").agg(count("UserID").alias("Total_Users"))

    binge_watch_summary = binge_counts.join(total_users, "AgeGroup")\
        .withColumn("Binge_Watching_Percentage", spark_round((col("Binge_Watchers") / col("Total_Users")) * 100, 2))

    return binge_watch_summary

def export_results(df, output_path):
    """
    Save the computed binge-watching statistics to a CSV file.
    """
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(output_path)

def main():
    """
    Run the binge-watching analysis pipeline.
    """
    spark = initialize_spark_session()
    input_file = "input/movie_ratings_data.csv"  # Adjust file path if necessary
    df = load_data_from_csv(spark, input_file)

    # Perform binge-watching analysis
    binge_watching_stats = compute_binge_watching_statistics(df)
    export_results(binge_watching_stats, "Outputs/binge_watching_patterns.csv")

    spark.stop()

if __name__ == "__main__":
    main()
