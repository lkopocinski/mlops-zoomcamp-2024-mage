from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data, *args, **kwargs):
    df_train = data
    dv = DictVectorizer()

    # train
    target_column = 'duration'
    feature_columns = ['PULocationID', 'DOLocationID']
    train_dicts = df_train[feature_columns].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df_train[target_column].values

    lr = LinearRegression()

    # train
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)

    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    print(f"Q5 - The RMSE of a training set: {rmse_train:.5f}.")
    
    print(lr.intercept_)
    return lr, dv