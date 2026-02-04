import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

directory = 'data/'
# Quiet warnings
# Loop through each file in the directory
# Create a table to store the results where each row is one method and the columns are the mean and std for each task (leg, leg-e, floor, floor-e, goal, goal-e), meaning there will be 12 columns in total
results = pd.DataFrame(columns=["Method", "leg-e_max", "leg-e_mean", "leg-e_std", "leg_max", "leg_mean", "leg_std", "floor-e_max", "floor-e_mean", "floor-e_std", "floor_max", "floor_mean", "floor_std", "goal-e_max", "goal-e_mean", "goal-e_std", "goal_max", "goal_mean", "goal_std"])
results_temp = {}
for file in os.listdir(directory):
    print(f"Processing file: {file}")
    if file.endswith('.csv'):
        # remove .csv from file name
        file_name = file[:-4]
        method_name = file_name.split('-')[0]
        if len(file_name.split('-')) > 2:
            task_name = file_name.split('-')[1] + '-e'
        else:
            task_name = file_name.split('-')[1]
        print(f"Method: {method_name}, Task: {task_name}")
        # Load the data from the CSV file
        data = pd.read_csv(os.path.join(directory, file))
        # get name of second column
        second_column_name = data.columns[1]
        # Ignore Step before 1_000_000
        if method_name == "ppo":
            print("PPO detected starting from beginning")
            start_idx = 0
        elif method_name == "ppow":
            print("PPOW detected starting from beginning")
            start_idx = data[(data["Step"] >= 500_000) & (data[second_column_name] < 0.5)].index.min()
        else:
            start_idx = data[(data["Step"] >= 1_000_000) & (data[second_column_name] < 0.5)].index.min()
        data = data.loc[start_idx:]
        # Get the mean, standard deviation and max
        mean_values = data.mean()
        std_dev_values = data.std()
        max_values = data.max()
        # Print only the second column's values as float without scientific notation
        pd.set_option('display.float_format', '{:.6f}'.format)
        print("Max:", max_values[1])
        print("Mean:", mean_values[1])
        print("Standard Deviation:", std_dev_values[1])
        print("\n")
        # Add the results to temp dictionary
        if method_name not in results_temp:
            results_temp[method_name] = {}
        results_temp[method_name][f"{task_name}_max"] = max_values[1]
        results_temp[method_name][f"{task_name}_mean"] = mean_values[1]
        results_temp[method_name][f"{task_name}_std"] = std_dev_values[1]
# Convert the temp dictionary to a DataFrame
for method, metrics in results_temp.items():
    row = {"Method": method}
    row.update(metrics)
    results = results._append(row, ignore_index=True)
# Save the results to a new CSV file
results.to_csv('summary_results.csv', index=False)
print("Summary results saved to summary_results.csv")
