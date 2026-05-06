from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def get_metrics(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"{model_name} Performance:")
    print(f"  MAE: {mae}")
    print(f"  RMSE: {rmse}")
    print(f"  R²: {r2}")
    print(f"  MAPE: {mape}%")
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}