from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer

print("🚀 Starting Spark job...")

spark = SparkSession.builder \
    .appName("EnergyConsumptionPrediction") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

try:
    print("📥 Reading validation data from HDFS...")
    validation_data = spark.read.csv("hdfs://172.17.0.1:9000/user/ubuntu/ValidationDataset.csv", header=True, inferSchema=True)
    row_count = validation_data.count()
    print(f"✅ Loaded validation data with {row_count} rows.\n")

    print("🔣 Applying StringIndexer...")
    bt_indexer = StringIndexer(inputCol="Building Type", outputCol="Building Type_indexed", handleInvalid="keep")
    dow_indexer = StringIndexer(inputCol="Day of Week", outputCol="Day of Week_indexed", handleInvalid="keep")
    validation_data = bt_indexer.fit(validation_data).transform(validation_data)
    validation_data = dow_indexer.fit(validation_data).transform(validation_data)

    print("🧮 Assembling feature vector...")
    assembler = VectorAssembler(
        inputCols=["Square Footage", "Number of Occupants", "Appliances Used", "Average Temperature",
                   "Building Type_indexed", "Day of Week_indexed"],
        outputCol="features"
    )
    validation_data = assembler.transform(validation_data)

    print("📦 Loading model from HDFS...")
    model = LinearRegressionModel.load("hdfs://172.17.0.1:9000/user/ubuntu/EnergyConsumptionModel")
    print("✅ Model loaded.\n")

    print("🔍 Running predictions...")
    predictions = model.transform(validation_data)
    predictions.select("prediction", "Energy Consumption").show(10)

    print("📐 Evaluating RMSE...")
    evaluator = RegressionEvaluator(labelCol="Energy Consumption", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"\n✅ Root Mean Squared Error (RMSE): {rmse:.2f} kWh")

except Exception as e:
    print(f"\n❌ ERROR: {e}")

spark.stop()
print("🏁 Spark job completed.")