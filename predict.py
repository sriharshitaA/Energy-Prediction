
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import argparse
import os

def create_spark_session():
    """Create and return a Spark session configured for a cluster."""
    return SparkSession.builder \
        .appName("EnergyConsumptionPrediction") \
        .master("spark://master:7077") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

def load_data(spark, path):
    """Load CSV data into a DataFrame."""
    return spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(path)

def preprocess_data(df):
    """Preprocess the data for ML training."""
    print("Data Schema:")
    df.printSchema()

    # Display basic statistics
    print("Dataset Statistics:")
    df.describe().show()

    # Check for null values
    print("Null Values Count:")
    for col in df.columns:
        null_count = df.filter(df[col].isNull()).count()
        if null_count > 0:
            print(f"{col}: {null_count}")

    # Identify feature columns (all columns except 'target')
    feature_cols = [col for col in df.columns if col != 'target']

    # Create feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    return assembler

def train_models(training_data, validation_data, feature_assembler):
    """Train multiple regression models and return the best one."""
    # Prepare training and validation data
    training_data = feature_assembler.transform(training_data)
    validation_data = feature_assembler.transform(validation_data)

    # Initialize evaluator
    evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="rmse")

    # Define models to train
    models = {
        "Linear Regression": LinearRegression(featuresCol="features", labelCol="target"),
        "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="target", numTrees=100),
        "Gradient Boosted Trees": GBTRegressor(featuresCol="features", labelCol="target", maxIter=50)
    }

    best_model = None
    best_rmse = float("inf")
    best_model_name = None

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model_fit = model.fit(training_data)

        # Make predictions on validation data
        predictions = model_fit.transform(validation_data)

        # Evaluate model
        rmse = evaluator.evaluate(predictions)
        print(f"{name} RMSE on validation data: {rmse}")

        # Save the best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model_fit
            best_model_name = name

    print(f"\nBest model: {best_model_name} with RMSE: {best_rmse}")
    return best_model, best_model_name, feature_assembler

def main():
    parser = argparse.ArgumentParser(description='Train energy consumption prediction models')
    parser.add_argument('--training', type=str, default='TrainingDataset.csv',
                        help='Path to training dataset CSV file')
    parser.add_argument('--validation', type=str, default='ValidationDataset.csv',
                        help='Path to validation dataset CSV file')
    parser.add_argument('--output', type=str, default='model',
                        help='Output directory for saving the model')

    args = parser.parse_args()

    # Create Spark session
    spark = create_spark_session()

    try:
        # Load datasets
        print("Loading training data...")
        training_data = load_data(spark, args.training)
        print(f"Training data loaded: {training_data.count()} records")

        print("\nLoading validation data...")
        validation_data = load_data(spark, args.validation)
        print(f"Validation data loaded: {validation_data.count()} records")

        # Preprocess data
        print("\nPreprocessing data...")
        feature_assembler = preprocess_data(training_data)

        # Train models
        print("\nTraining models...")
        best_model, best_model_name, feature_assembler = train_models(training_data, validation_data, feature_assembler)

        # Save the best model and feature assembler
        output_path = args.output
        print(f"\nSaving {best_model_name} model to {output_path}")
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)

        # Create pipeline with feature assembler and model
        pipeline = Pipeline(stages=[feature_assembler, best_model])
        pipeline_model = pipeline.fit(training_data)

        # Save pipeline model
        pipeline_model.write().overwrite().save(output_path)
        print(f"Model saved successfully to {output_path}")

    finally:
        # Stop the Spark session
        spark.stop()

if __name__ == "__main__":
    main()
