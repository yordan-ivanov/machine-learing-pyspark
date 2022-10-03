# Defining a function that runs Keras Long Short Term Memory model.
def RunLSTM_Pyspark(df):

    # Creating a temporary dataset with all features for scaling.
    df_temp = df.loc[:, ['symbol','open','high','low', 'volume', 'close']]

    # Grouping the data by indicated column name and scaling data by their minimum and maximum score.
    scaled_data = df_temp.groupby('symbol').apply(lambda x: MinMax_Scaler(x))

    # Droping rows with not a number values.
    scaled_data.dropna(inplace=True)
    scaled_data.reset_index(drop=True)

    # Selecting independent variables and training size from the scaled data.
    data_set = scaled_data.loc[:, ['open','high','low','volume', 'close']].values
    train_size = int(len(data_set)*0.7)

    # Splitting the data to "X" and "y" sets with 70% for training.
    train_set = data_set[:train_size]
    X_train, y_train = [],[]
    for i in range(0, len(train_set)):
        X_train.append(train_set[i][0:4])
        y_train.append(train_set[i][4:])
    X_train, y_train = np.array(X_train, dtype=float), np.array(y_train, dtype=float)

    # Reshaping the train data to three dimensional array for LSTM model.
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Setting up Long Short Term Memory model from Keras library.
    lstm = Sequential()
    lstm.add(LSTM(50, input_shape=(X_train.shape[1], 1), activation='tanh', return_sequences=True))
    lstm.add(LSTM(50, return_sequences=False, activation='tanh'))
    lstm.add(Dense(25))
    lstm.add(Dense(1))

    # Compiling LSTM model.
    lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Training the LSTM model.
    lstm.fit(X_train, y_train, batch_size=1, epochs=1)

    # Creating a test set with "X" and "y" for the remaining 30% of the data.
    test_set = data_set[train_size:]
    X_test, y_test = [],[]
    for i in range(0, len(test_set)):
        X_test.append(test_set[i][0:4])
        y_test.append(test_set[i][4:])
    X_test, y_test = np.array(X_test, dtype=float), np.array(y_test, dtype=float)

    # Reshaping the test data to three dimensional array for LSTM model.
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Obtaining LSTM predictions.
    predictions = lstm.predict(X_test)

    # Computing common measures for regression such as:
    # R-square, Root Mean Square Error, Absolut Mean Error, Variance.
    zx = (predictions-np.mean(predictions))/np.std(predictions, ddof=1)
    zy = (y_test-np.mean(y_test))/np.std(y_test, ddof=1)
    r = np.sum(zx*zy)/(len(predictions)-1)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    mae = np.mean(np.absolute(predictions - y_test))
    var = np.var(predictions)
    measures = [['R-square', r**2], ['RMSE', rmse], ['MAE', mae], ['Variance', var]]
    model_results = pd.DataFrame(measures, columns=['Measures', 'LSTM_results'])

    return model_results