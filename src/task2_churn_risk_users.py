from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit

def initialize_spark_session(app_name="Churn_Risk_Detection"):
    """
    Set up and return a SparkSession instance.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_csv_data(spark, file_path):
    """
    Read movie ratings dataset from a CSV file into a Spark DataFrame.
    """
    return spark.read.option("header", True).option("inferSchema", True).csv(file_path)

def identify_churn_risk_users(df):
    """
    Detect users at risk of churning by filtering those who have canceled subscriptions 
    and have watch time below 100 minutes.
    """
    churn_candidates = df.filter((col("SubscriptionStatus") == "Canceled") & (col("WatchTime") < 100))
    churn_count = churn_candidates.agg(count("UserID").alias("Total_Users"))

    # Adding a label for churn risk category
    churn_count = churn_count.withColumn("Churn_Risk_Category", 
                                         lit("Users with low watch time & canceled subscriptions"))

    return churn_count.select("Churn_Risk_Category", "Total_Users")

def export_results(df, output_path):
    """
    Write the processed churn risk data to a CSV file.
    """
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(output_path)

def main():
    """
    Execute the workflow to identify churn risk users.
    """
    spark = initialize_spark_session()
    input_file = "input/movie_ratings_data.csv"  # Modify the path as needed
    df = load_csv_data(spark, input_file)

    # Perform churn risk analysis
    churn_risk_data = identify_churn_risk_users(df)
    export_results(churn_risk_data, "Outputs/churn_risk_users.csv")

    spark.stop()

if __name__ == "__main__":
    main()
