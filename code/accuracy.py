import pandas as pd
import numpy as np

def calculate_accuracy(file_name):
    data = pd.read_csv(file_name, header=None)
    global_sum_mape, global_sum_mae, global_sum_rmse = 0.0, 0.0, 0.0
    global_num_mape, global_num_mae, global_num_rmse = 0, 0, 0

    for i, row in data.iterrows():
        node_id = row[0] 
        for t in range(1, len(row)):
            if row[t] != '':
                pred, gt = map(float, row[t].split(','))
                if gt != 0:  
                    mape = abs(pred - gt) / abs(gt)  
                    mae = abs(pred - gt)  
                    rmse = (pred - gt) ** 2  
                    global_sum_mape += mape
                    global_sum_mae += mae
                    global_sum_rmse += rmse
                    global_num_mape += 1
                    global_num_mae += 1
                    global_num_rmse += 1

    global_mape = global_sum_mape / global_num_mape if global_num_mape != 0 else np.nan
    global_mae = global_sum_mae / global_num_mae if global_num_mae != 0 else np.nan
    global_rmse = (global_sum_rmse / global_num_rmse) ** 0.5 if global_num_rmse != 0 else np.nan
    print(f"Global MAPE: {global_mape:.6f}, Global MAE: {global_mae:.6f}, Global RMSE: {global_rmse:.6f}")

if __name__ == "__main__":
    calculate_accuracy('./1010_test_predictions.csv')