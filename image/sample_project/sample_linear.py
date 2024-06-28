import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 設定 MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("linear_regression_experiment")

# 生成一些隨機數據
X = np.array([[i] for i in range(100)])
y = np.array([2 * i + np.random.normal() for i in range(100)])

# 切分數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 MLflow 追蹤實驗
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 預測測試集
    predictions = model.predict(X_test)
    
    # 計算 MSE
    mse = mean_squared_error(y_test, predictions)
    
    # 記錄參數、指標和模型
    mlflow.log_param("coef", model.coef_[0])
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    
    print("Model coefficient:", model.coef_[0])
    print("Mean squared error:", mse)
