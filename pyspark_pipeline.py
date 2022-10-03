# Importing relevant libraries.
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Pyspark libraries.
import pyspark.sql.functions as rand
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType
import pyspark.sql.functions as F

# Creating spark dataframe from the grouped Resilient Distributed Datasets.
spdf_grouped = spark.createDataFrame(grouped_rdd).toDF('symbol', 'features_vector')
spdf_grouped.printSchema()

# Defining a function that scaled data by its minimum and maximum values.
def MinMax_Scaler(data):
    """
    Function that applies a minimum and maximum scaler to normalize the data.
    It performe the folowing steps:
    -----------
    1. It standardize the data by taking a away from each data point the minimum score of 
       the data and it divides it by the diference between the maximum and the minimum data scores.
    2. It multiplies the already standardized data by the diference between the maximum and the minimum 
       standardized data scores and it sums to it the minimum score in case min and max are the same.
    
    Parameters:
    -----------
    data : Any numeric numpy array or pandas dataset. 
    
    Returns:
    -----------
    It returns a dataset with the same shape as the input but all instances are scaled by 
    their minimum and maximum values.
    
    """
    data_std = (data - np.min(data, axis=0)) / (
                       np.max(data, axis=0) - np.min(data, axis=0))
    data_scaled = data_std * (np.max(data_std, axis=0) - np.min(data_std, axis=0)) + np.min(data_std, axis=0)
    return data_scaled

# Scaling the data by minimum and maximum value.
array_list = [dft.features_vector[x][0] for x in range(0, len(dft))]
scaled_list = [MinMax_Scaler(x) for x in array_list]

# Converting the scaled data back to spark dataframe with label and faatures vector.
scaled_array = map(lambda x: (float(x[4]), Vectors.dense(x[0:4])), np.vstack(scaled_list))
spdf_scaled = spark.createDataFrame(scaled_array).toDF('close', 'x_vector')

# Splitting the scaled data into train test datasets with 30% test sample.
train_scaled, test_scaled = spdf_scaled.select('x_vector', 'close').randomSplit([0.7, 0.3], seed=2022)

# Creating a new model with the unscaled data by applying VectorAssembler transformation.
assembler = VectorAssembler(inputCols=['open', 'high', 'low', 'volume'], outputCol='x_vector')
spdf_vector = assembler.transform(spdf)

# Splitting the unscaled data into train test datasets with 30% test sample.
train_unscaled, test_unscaled = spdf_vector.select('x_vector', 'close').randomSplit([0.7, 0.3], seed=2022)


# Setting up a Random Forest Regressor model.
RFR = RandomForestRegressor(featuresCol='x_vector', labelCol='close', predictionCol='prediction', 
                             numTrees=100, impurity='variance', maxDepth=5, maxBins=32, seed=2022)

# Fit and transform the scaled and unscaled datasets on a pipeline.
train_datasets = [train_scaled, train_unscaled]
test_datasets = [test_scaled, test_unscaled]

# Running both models in a pipeline.
for i, (train, test) in enumerate(zip(train_datasets, test_datasets)):
    pipe = Pipeline(stages=[RFR])
    globals()['predicted'+str(i+1)] = pipe.fit(train).transform(test)