import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
client = mlflow.tracking.MlflowClient()
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
def build_data(data_path):
   # Read the wine-quality csv file from the URL
    data = pd.read_csv(data_path, low_memory=False,encoding='latin-1', na_values=['(NA)']).fillna(0)
    for column in data.columns:
         data[column] = pd.Categorical(data[column]).codes
      # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train_x = train.drop(["Sold_Amount"], axis=1)
    test_x = test.drop(["Sold_Amount"], axis=1)
    train_y = train[["Sold_Amount"]]
    test_y = test[["Sold_Amount"]]
    return train_x, test_x, train_y, test_y
    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run {}".format(run_id))
    # show logged data
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        print(data)
train_x, test_x, train_y, test_y = build_data(r'datium_train.csv')
def xgb_model(training_data, test_data, max_depth, ntrees, lr):
    with mlflow.start_run():
        xgbRegressor = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=ntrees,
            learning_rate=lr,
            random_state=42,
            seed=42,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=1,
            gamma=1,
            objective="reg:squarederror"
        )
        pipeline = Pipeline(steps=[("regressor", xgbRegressor)])
        xgbRegressor.fit(train_x, train_y)
        predicted_prices = xgbRegressor.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_prices)
        print('rmse":', rmse)
        print('r2', r2)
        print('mae', mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(xgbRegressor, "xgb_model")
        feature_importances = pd.DataFrame(xgbRegressor.feature_importances_, index=train_x.columns.tolist(),
                                           columns=['importance'])
        feature_importances.sort_values('importance', ascending=False)

if __name__ == "__main__":

    xgb_model(train_y, test_y, 3, 130, 0.3)
train_x, test_x, train_y, test_y = build_data(r'datium_test.csv')
xgb_model(train_y,test_y, 3,130,0.3)