from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

def initialize_spark_session(app_name="Movie_Trend_Analysis"):
    """
    Set up and return a SparkSession instance.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_movie_data(spark, file_path):
    """
    Load movie rating details from a CSV file into a Spark DataFrame.
    """
    return spark.read.option("header", True).option("inferSchema", True).csv(file_path)

def evaluate_watching_trends(df):
    """
    Analyze yearly movie-watching trends and identify peak activity periods.
    """
    trend_data = df.groupBy("WatchedYear").agg(count("MovieID").alias("Total_Movies_Watched"))
    return trend_data.orderBy("WatchedYear")

def export_trend_results(df, output_path):
    """
    Write the analyzed trend data to a CSV file.
    """
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(output_path)

def main():
    """
    Execute the movie-watching trend analysis process.
    """
    spark = initialize_spark_session()
    input_file = "input/movie_ratings_data.csv"  # Update the file path as needed
    df = load_movie_data(spark, input_file)

    # Conduct trend analysis
    watching_trend_results = evaluate_watching_trends(df)
    export_trend_results(watching_trend_results, "Outputs/movie_watching_trends.csv")

    spark.stop()

if __name__ == "__main__":
    main()
