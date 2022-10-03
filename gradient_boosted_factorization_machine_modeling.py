# Converting to Pandas dataframe.
df_1m = spdf1m.toPandas()

# Creating a temporary dataset with all features for scaling.
df_temp = df_1m.loc[:, ['symbol','open','high','low', 'volume', 'close']]

# Grouping the data by indicated column name and scaling data by their minimum and maximum score.
scaled_data = df_temp.groupby('symbol').apply(lambda x: MinMax_Scaler(x))

# Droping rows with not a number values.
scaled_data.dropna(inplace=True)
scaled_data.reset_index(drop=True)

# Selecting independent variables and training size from the scaled data.
data_set = scaled_data.loc[:, ['open','high','low','volume', 'close']]

# Convert back to Spark dataframe.
spdf_1m = spark.createDataFrame(data_set).toDF('open','high','low','volume', 'close')

# Creating a new model with the 1min dataset by applying VectorAssembler transformation.
assembler = VectorAssembler(inputCols=['open', 'high', 'low', 'volume'], outputCol='x_vector')
spdf_temp = assembler.transform(spdf_1m)

# Splitting the 1min dataset into train test datasets with 30% test sample.
train_1m, test_1m = spdf_temp.select('x_vector', 'close').randomSplit([0.7, 0.3], seed=2022)

# Running the three models in a pipeline.
for i, model in enumerate([RFR, GBTR, FMR]):    
    pipe = Pipeline(stages=[model])
    globals()['predicted'+str(i+5)] = pipe.fit(train_1m).transform(test_1m)
    
# Evaluating both Gradient-Boosted Tree and Factorization Machine Models with common measures for regression.
for i, prediction in enumerate([predicted5, predicted6, predicted7]):
    for m in ['r2', 'rmse', 'mae', 'var']:
        metric = RegressionEvaluator(labelCol='close', predictionCol='prediction', metricName=f'{m}')
        globals()[str(m)+'_model'+str(i+5)] = metric.evaluate(prediction)
        
# Scaling the data with mean equals to zero and standard deviation equals to one.
std_scaled_list = [Stand_Scaler(x) for x in array_list]
std_scaled_array = map(lambda x: (float(x[4]), Vectors.dense(x[0:4])), np.vstack(std_scaled_list))
spdf_std = spark.createDataFrame(std_scaled_array).toDF('close', 'x_vector')

# Splitting the standard scaled data into train test datasets with 30% test sample.
train_std, test_std = spdf_std.select('x_vector', 'close').randomSplit([0.7, 0.3], seed=2022)

# Setting up Gradient-Boosted Tree Model.
GBTR = GBTRegressor(featuresCol='x_vector', labelCol='close', predictionCol='prediction')

# Setting up Factorization Machine Model.
FMR = FMRegressor(featuresCol='x_vector', labelCol='close', predictionCol='prediction', factorSize=3)

# Running the two models in a pipeline.
for i, model in enumerate([GBTR, FMR]):    
    pipe = Pipeline(stages=[model])
    globals()['predicted'+str(i+3)] = pipe.fit(train_std).transform(test_std)
    
# Evaluating both Gradient-Boosted Tree and Factorization Machine Models with common measures for regression.
for i, prediction in enumerate([predicted3, predicted4]):
    for m in ['r2', 'rmse', 'mae', 'var']:
        metric = RegressionEvaluator(labelCol='close', predictionCol='prediction', metricName=f'{m}')
        globals()[str(m)+'_model'+str(i+3)] = metric.evaluate(prediction)