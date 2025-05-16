from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("EnergyModelTraining").getOrCreate()

df = spark.read.csv("file:///home/ubuntu/TrainingDataset.csv", header=True, inferSchema=True).na.drop()

# Feature columns
cat_cols = ["Building Type", "Day of Week"]
num_cols = ["Square Footage", "Number of Occupants", "Appliances Used", "Average Temperature"]
label_col = "Energy Consumption"

# Pipeline stages
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in cat_cols]
assembler = VectorAssembler(inputCols=[f"{c}_idx" for c in cat_cols] + num_cols, outputCol="features_raw")
scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol=label_col)

pipeline = Pipeline(stages=indexers + [assembler, scaler, lr])
model = pipeline.fit(df)

model.write().overwrite().save("file:///home/ubuntu/energy_model")
print("✅ Model training complete and saved to /home/ubuntu/energy_model")

spark.stop()
