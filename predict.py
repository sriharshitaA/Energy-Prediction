import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

if len(sys.argv) != 2:
    print("Usage: spark-submit predict_model.py <path_to_test_file>")
    sys.exit(1)

# Ensure absolute path
test_path = os.path.abspath(sys.argv[1])
spark = SparkSession.builder.appName("EnergyModelPrediction").getOrCreate()

model = PipelineModel.load("file:///home/ubuntu/energy_model")
test_df = spark.read.csv(f"file://{test_path}", header=True, inferSchema=True).na.drop()

predictions = model.transform(test_df)
evaluator = RegressionEvaluator(labelCol="Energy Consumption", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"✅ RMSE on test set: {rmse:.3f}")

spark.stop()
