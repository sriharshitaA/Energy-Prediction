# train_model.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder \
    .appName("TrainEnergyModel") \
    .master("spark://172.31.24.180:7077") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

# Load training data
df = spark.read.csv("hdfs://namenode:9000/user/ubuntu/TrainingDataset.csv", header=True, inferSchema=True)

# Index categorical columns
bt_indexer = StringIndexer(inputCol="Building Type", outputCol="Building Type_indexed", handleInvalid="keep")
dow_indexer = StringIndexer(inputCol="Day of Week", outputCol="Day of Week_indexed", handleInvalid="keep")
df = bt_indexer.fit(df).transform(df)
df = dow_indexer.fit(df).transform(df)

# Assemble features
assembler = VectorAssembler(
    inputCols=["Square Footage", "Number of Occupants", "Appliances Used", "Average Temperature",
               "Building Type_indexed", "Day of Week_indexed"],
    outputCol="features"
)
df = assembler.transform(df)

# Train the model
lr = LinearRegression(featuresCol="features", labelCol="Energy Consumption")
model = lr.fit(df)

# Save the model to HDFS
model.save("hdfs://namenode:9000/user/ubuntu/EnergyConsumptionModel")
spark.stop()