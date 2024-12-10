import os  
import json  
import logging  
import concurrent.futures  
import importlib as imp  
from functools import partial  
from typing import Dict, List, Callable, Any  

import numpy as np  
import pandas as pd  

from scipy.io import arff  
from tqdm import tqdm  

from sklearn.base import clone  
from sklearn.datasets import load_svmlight_file  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import OneHotEncoder  

from river import metrics  
 
from river import drift  

from openai import AzureOpenAI  
from openai_config import get_openai_config  

import helpers  
from helpers import (  
    adaptive_batch_self_healing, adaptive_ensemble_with_ddm,  
    adaptive_batch_with_sliding_window, batch_learning_no_retraining,  
    adaptive_batch_with_drift_detection_ddm, visualize_all  
)  

import HLLM  
from HLLM import H_LLM  

from dataset_helpers import load_five_datasets  

#### Data generation 

# Function to generate diabetes data
def generate_diabetes_data(n_samples, noise_distribution_params, coefficients, seed):
    np.random.seed(seed)
    HbA1c = np.random.normal(5.7, 0.5, n_samples)
    FastingGlucose = np.random.normal(100, 15, n_samples)
    Age = np.random.normal(50, 12, n_samples)
    BMI = np.random.normal(25, 4, n_samples)
    BloodPressure = np.random.normal(120, 15, n_samples)
    Cholesterol = np.random.normal(200, 40, n_samples)
    Insulin = np.random.normal(85, 45, n_samples)
    PhysicalActivity = np.random.normal(3, 1, n_samples)
    
    X = np.vstack((HbA1c, FastingGlucose, Age, BMI, BloodPressure, Cholesterol, Insulin, PhysicalActivity)).T
    noise = np.random.normal(noise_distribution_params['mean'], noise_distribution_params['std'], n_samples)
    linear_combination = np.dot(X, coefficients) + noise
    probabilities = 1 / (1 + np.exp(-linear_combination))
    adjusted_probabilities = np.clip(probabilities, 0.4, 0.6)
    outcomes = np.where(adjusted_probabilities > 0.5, 1, 0)
    variables = ['HbA1c', 'FastingGlucose', 'Age', 'BMI', 'BloodPressure', 'Cholesterol', 'Insulin', 'PhysicalActivity']
    data = pd.DataFrame(X, columns=variables)
    data['Outcome'] = outcomes
    return data.drop('Outcome', axis=1), data['Outcome']

# Function to prepare datasets with covariate shift and corruption
def prepare_datasets_with_covariate_shift_and_corruption(n_samples1, n_samples2, coefficients1, coefficients2, noise_params1, noise_params2, seed, prop_outliers=1, columns_to_corrupt=None, outlier_factor = 20):
    """
    Prepare datasets with a covariate shift and corruption of specified variables.
    
    Parameters:
    - n_samples1: Number of samples in the first dataset.
    - n_samples2: Number of samples in the second dataset.
    - coefficients1: Coefficients for the first dataset.
    - coefficients2: Coefficients for the second dataset.
    - noise_params1: Noise parameters for the first dataset.
    - noise_params2: Noise parameters for the second dataset.
    - seed: Random seed for reproducibility.
    - prop_outliers: Proportion of outliers to introduce.
    - columns_to_corrupt: List of columns to corrupt in the second dataset.
    
    Returns:
    - X_train: Training features with time indices.
    - y_train: Training labels.
    - X_test: Testing features with time indices, including covariate shift and corruption.
    - y_test: Testing labels, including covariate shift and corruption.
    - shift_index: Time index where the covariate shift happens.
    """
    
    # Generate the first dataset
    X1, y1 = generate_diabetes_data(n_samples1, noise_params1, coefficients1, seed)
    
    # Split the first dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.7, random_state=42)
    
    # Generate the second dataset
    X2, y2 = generate_diabetes_data(n_samples2, noise_params2, coefficients2, seed)
    
    # Introduce outliers in the second dataset
    n_outliers = int(prop_outliers * len(X2))

    if columns_to_corrupt:
        idx_outliers = np.random.choice(X2.index, n_outliers, replace=False)
        for column in columns_to_corrupt:
            #print(idx_outliers)
            #display(X2)
            X2.loc[idx_outliers, column] *= outlier_factor
            #display(X2)
        y2.loc[idx_outliers] = np.random.binomial(1, 0.5, n_outliers)
    
    # Concatenate X_test from the first dataset with the entire second dataset
    X_test = pd.concat([X_test, X2], ignore_index=True)
    y_test = pd.concat([y_test, y2], ignore_index=True)
    
    # Add time column
    X_train['time'] = np.arange(len(X_train))
    X_test['time'] = np.arange(len(X_train), len(X_train) + len(X_test))
    
    # The shift index is the start of the second dataset in X_test
    shift_index = len(X_test) - len(X2)
    
    return X_train, y_train, X_test, y_test, shift_index



def batch_learning_no_retraining(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted):
    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Process non-corrupted batches without retraining
    for xi, yi in zip(X_test_batches[:-1], y_test_batches[:-1]):
        model.predict(xi)  # Prediction made but not used

    # Final prediction on corrupted batch
    y_pred = model.predict(x_corrupted)
    return accuracy_score(y_test_batches[-1], y_pred)

def adaptive_batch_with_drift_detection_ddm(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector):
    model.fit(X_train, y_train)
    
    for xi, yi in zip(X_test_batches[:-1], y_test_batches[:-1]):
        y_pred = model.predict(xi)
        accuracy = accuracy_score(yi, y_pred)
        
        _ = drift_detector.update(1 - accuracy)
        if drift_detector.drift_detected:
            model.fit(xi, yi)
    
    y_pred = model.predict(x_corrupted)
    return accuracy_score(y_test_batches[-1], y_pred)

def adaptive_batch_with_sliding_window(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, window_size=5):
    model.fit(X_train, y_train)
    
    X_buffer, y_buffer = [], []
    
    for xi, yi in zip(X_test_batches[:-1], y_test_batches[:-1]):
        X_buffer.append(xi)
        y_buffer.append(yi)
        
        if len(X_buffer) > window_size:
            X_buffer.pop(0)
            y_buffer.pop(0)
        
        if len(X_buffer) == window_size:
            X_retrain = np.concatenate(X_buffer, axis=0)
            y_retrain = np.concatenate(y_buffer, axis=0)
            model.fit(X_retrain, y_retrain)
    
    y_pred = model.predict(x_corrupted)
    return accuracy_score(y_test_batches[-1], y_pred)

def adaptive_ensemble_with_ddm(base_model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector):
    models = [clone(base_model)]
    model_weights = [1.0]
    
    models[0].fit(X_train, y_train)
    
    for xi, yi in zip(X_test_batches[:-1], y_test_batches[:-1]):
        weighted_predictions = np.zeros(len(yi))
        
        for model, weight in zip(models, model_weights):
            preds = model.predict(xi)
            weighted_predictions += preds * weight
        
        final_predictions = np.round(weighted_predictions / sum(model_weights))
        accuracy = accuracy_score(yi, final_predictions)
        
        current_accuracies = [accuracy_score(yi, model.predict(xi)) for model in models]
        model_weights = [w * acc for w, acc in zip(model_weights, current_accuracies)]
        
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]

        _ = drift_detector.update(1 - accuracy)
        if drift_detector.drift_detected:
            new_model = clone(base_model)
            new_model.fit(xi, yi)
            models.append(new_model)
            model_weights.append(1.0)
            total_weight = sum(model_weights)
            model_weights = [w / total_weight for w in model_weights]
    
    weighted_predictions = np.zeros(len(y_test_batches[-1]))
    for model, weight in zip(models, model_weights):
        preds = model.predict(x_corrupted)
        weighted_predictions += preds * weight
    
    final_predictions = np.round(weighted_predictions / sum(model_weights))
    return accuracy_score(y_test_batches[-1], final_predictions)

 

def adaptive_batch_self_healing(
    H: 'H_LLM', 
    model,
    X_train,
    y_train,
    X_test_batches,
    y_test_batches,
    x_corrupted,
    drift_detector,
    corrupt_val,
    n_cols_corrupt,
    backtesting_window=0
):
    """
    Adaptive batch self-healing function that utilizes the H_LLM class for diagnosing and mitigating model drift.

    Parameters:
    - H (H_LLM): Instance of the H_LLM class for self-healing.
    - model: Machine learning model to be trained and evaluated.
    - X_train (pd.DataFrame): Training feature data.
    - y_train (pd.Series): Training target data.
    - X_test_batches (List[pd.DataFrame]): List of testing feature data batches.
    - y_test_batches (List[pd.Series]): List of testing target data batches.
    - x_corrupted (pd.DataFrame): Corrupted feature data to be decorrupted.
    - drift_detector: Drift detection mechanism instance.
    - corrupt_val (float): Value used to decorrupt the corrupted data.
    - n_cols_corrupt (int): Number of columns in x_corrupted to be decorrupted.
    - backtesting_window (int): Number of previous batches to use for backtesting.

    Returns:
    - final_accuracy (float): Accuracy score on the decorrupted data.
    """
    
    # Fit the initial model on the training data
    model.fit(X_train, y_train)
    
    # Initialize buffers to store processed batches
    X_buffer, y_buffer = [], []
    
    # Iterate through all but the last test batch
    for idx, (xi, yi) in enumerate(tqdm(zip(X_test_batches[:-1], y_test_batches[:-1]), 
                                       total=len(X_test_batches[:-1]),
                                       desc="Processing Batches")):
        # Make predictions and calculate accuracy
        y_pred = model.predict(xi)
        accuracy = accuracy_score(yi, y_pred)
        
        # Update the drift detector with the current batch's prediction error
        drift_detector.update(1 - accuracy)
        
        # Add current batch to buffers
        X_buffer.append(xi)
        y_buffer.append(yi)
        
        # Check if drift is detected
        if drift_detector.drift_detected:
            print(f"Drift detected at batch {idx + 1}! Initiating self-healing process...")
            
            # Define the inspection window (current batch)
            x_inspect = xi
            y_inspect = yi
            
            # Define the data before drift
            if len(X_buffer) > 1:
                x_before = pd.concat(X_buffer[:-1], axis=0)
                y_before = pd.concat(y_buffer[:-1], axis=0)
            else:
                x_before = X_train
                y_before = y_train
            
            # Define the backtesting window
            if backtesting_window > 0 and len(X_buffer) > backtesting_window:
                x_backtest = pd.concat(X_buffer[-backtesting_window:], axis=0)
                y_backtest = pd.concat(y_buffer[-backtesting_window:], axis=0)
            else:
                x_backtest = x_before
                y_backtest = y_before
            
            # Utilize H_LLM to fit the model with self-healing
            model = H.fit_model(model, x_before, x_inspect, y_inspect, x_backtest, y_backtest)
            
            # Optionally, reset buffers after handling drift
            X_buffer.clear()
            y_buffer.clear()
     
    y_pred_final = model.predict(X_test_batches[-1])
    final_accuracy = accuracy_score(y_test_batches[-1], y_pred_final)
    return final_accuracy


# Wrapper functions to ensure consistent interface
def batch_learning_no_retraining_wrapper(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector, num_batches_test, corrupt_val, n_cols_corrupt):
    return batch_learning_no_retraining(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted)

def adaptive_batch_with_sliding_window_wrapper(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector, num_batches_test, corrupt_val, n_cols_corrupt):
    return adaptive_batch_with_sliding_window(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, num_batches_test)

def adaptive_batch_with_drift_detection_ddm_wrapper(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector, num_batches_test, corrupt_val, n_cols_corrupt):
    return adaptive_batch_with_drift_detection_ddm(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector)

def adaptive_ensemble_with_ddm_wrapper(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector, num_batches_test, corrupt_val, n_cols_corrupt):
    return adaptive_ensemble_with_ddm(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector)

def adaptive_batch_self_healing_wrapper(model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector, num_batches_test, corrupt_val, n_cols_corrupt):
    return adaptive_batch_self_healing(H, model, X_train, y_train, X_test_batches, y_test_batches, x_corrupted, drift_detector, corrupt_val, n_cols_corrupt)



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom progress bar function
def custom_progress_bar(iterable, desc=None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc)
    except ImportError:
        print(f"{desc}: Starting...")
        result = list(iterable)
        print(f"{desc}: Completed")
        return result

def run_experiments(datasets: Dict[str, Dict[str, np.ndarray]], 
                    methods: List[Callable], 
                    num_runs: int = 5) -> Dict[str, Any]:
    results = {}
    
    for dataset_name, dataset in custom_progress_bar(datasets.items(), desc="Datasets"):
        try:
            results[dataset_name] = process_dataset(dataset_name, dataset, methods, num_runs)
        except Exception as exc:
            logging.error(f'{dataset_name} generated an exception: {exc}')
            results[dataset_name] = {'error': str(exc)}
    
    return results

def process_dataset(dataset_name: str, 
                    dataset: Dict[str, np.ndarray], 
                    methods: List[Callable], 
                    num_runs: int) -> Dict[str, Any]:
    X, y = dataset['X'], dataset['y']
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Experiment 1: Vary n_cols_corrupt
        future_to_n_cols = {
            executor.submit(
                run_experiment, 
                X, y, methods, num_runs, 
                n_cols_corrupt=n, 
                corrupt_val=100
            ): n for n in range(1, 7)
        }
        results['vary_cols'] = {
            future_to_n_cols[future]: future.result() 
            for future in concurrent.futures.as_completed(future_to_n_cols)
        }
        
        # Experiment 2: Vary corrupt_val
        future_to_corrupt_val = {
            executor.submit(
                run_experiment, 
                X, y, methods, num_runs, 
                n_cols_corrupt=4, 
                corrupt_val=val
            ): val for val in [1, 2, 5, 10, 20, 50, 100]
        }
        results['vary_corrupt_val'] = {
            future_to_corrupt_val[future]: future.result() 
            for future in concurrent.futures.as_completed(future_to_corrupt_val)
        }
    
    return results

def run_experiment(X: np.ndarray, 
                   y: np.ndarray, 
                   methods: List[Callable], 
                   num_runs: int, 
                   n_cols_corrupt: int, 
                   corrupt_val: float) -> Dict[str, Any]:
    results = {method.__name__: {'accuracies': []} for method in methods}
    
    for _ in range(num_runs): # test_size = 0.9 before.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=_)
        
        model = RandomForestClassifier(n_estimators=1, random_state=_)
        num_batches_test = 30
        threshold = 0.8
        warm_start = 3
        
        X_test_batches = np.array_split(X_test, num_batches_test)
        y_test_batches = np.array_split(y_test, num_batches_test)
        
        x_corrupted = X_test_batches[-1].copy()
        if isinstance(x_corrupted, pd.DataFrame):
            x_corrupted.iloc[:, :n_cols_corrupt] *= corrupt_val
        else:
            x_corrupted[:, :n_cols_corrupt] *= corrupt_val
        
        drift_detector = drift.binary.DDM(drift_threshold=threshold, warm_start=warm_start)
        
        for method in methods:
            try:
                accuracy = method(model, X_train, y_train, X_test_batches, y_test_batches, 
                                  x_corrupted, drift_detector, num_batches_test, corrupt_val, n_cols_corrupt)
                results[method.__name__]['accuracies'].append(accuracy)
            except Exception as exc:
                logging.error(f"Error in method {method.__name__}: {exc}")
                results[method.__name__]['accuracies'].append(None)
    
    # Calculate mean and std for each method
    for method_name in results:
        accuracies = [acc for acc in results[method_name]['accuracies'] if acc is not None]
        if accuracies:
            results[method_name]['mean'] = np.mean(accuracies)
            results[method_name]['std'] = np.std(accuracies)
            results[method_name]['accuracies'] = accuracies
        else:
            results[method_name]['mean'] = None
            results[method_name]['std'] = None
        #del results[method_name]['accuracies']  # Remove individual accuracies to save space
    
    return results


# Example usage:
n_samples1 = 20_000
n_samples2 = 100_000
coefficients1 = [0.3, 0.0075, -0.01, 0.05, 0.04, -0.03, -0.02, -0.1]
coefficients2 = [-0.3, -0.0075, 0.2, -0.05, -0.015, -0.001, 0.02, -2]
noise_params1 = {'mean': 0, 'std': 0.2}
noise_params2 = {'mean': 0, 'std': 0.2}
seed = 42
columns_to_corrupt = ['Age']
prop_outliers = 0.4

X_train, y_train, X_test, y_test, shift_index = prepare_datasets_with_covariate_shift_and_corruption(
    n_samples1, n_samples2, coefficients1, coefficients2, noise_params1, noise_params2, seed, prop_outliers, columns_to_corrupt, outlier_factor=20)

X_train = X_train.drop('time', axis=1)
X_test = X_test.drop('time', axis=1)


# Get config
config = get_openai_config()

# Load the LLM (Azure Config object)
llm = AzureOpenAI(
    api_version=config["api_version"],
    azure_endpoint=config["api_base"],
    api_key=config["api_key"],
)

context = "The goal is to hypothesize concrete reasons for why the model has underperformed. "
H = H_LLM(config=config, llm=llm, context=context)

model = LogisticRegression()

X_test_corrupted = X_test.copy()
y_test_corrupted = y_test.copy()

datasets = load_five_datasets()


# Run the experiments
methods = [
    batch_learning_no_retraining_wrapper,
    adaptive_batch_with_sliding_window_wrapper,
    adaptive_batch_with_drift_detection_ddm_wrapper,
    adaptive_ensemble_with_ddm_wrapper,
    adaptive_batch_self_healing_wrapper
]

results = run_experiments(datasets, methods)

# Save results to JSON file
with open('./results/0806_v4_real_datasets_experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)

logging.info("Experiments completed and results saved to 'experiment_results.json'")



# Load the JSON data
with open('/results/0806_v4_real_datasets_experiment_results.json', 'r') as f:
    results = json.load(f)

# Method name mapping
method_names = {
    'batch_learning_no_retraining_wrapper': 'No retraining',
    'adaptive_batch_with_sliding_window_wrapper': 'Partially Updating',
    'adaptive_batch_with_drift_detection_ddm_wrapper': 'New model training',
    'adaptive_ensemble_with_ddm_wrapper': 'Ensemble Method',
    'adaptive_batch_self_healing_wrapper': 'Self-Healing ML'
}

# Initialize lists to store data
vary_cols_data = []
vary_corrupt_val_data = []

# Process the data
for dataset, dataset_results in results.items():
    # Process vary_cols data
    for n_cols_corrupt, methods_data in dataset_results['vary_cols'].items():
        for method, metrics in methods_data.items():
            vary_cols_data.append({
                'Dataset': dataset,
                'N_Cols_Corrupt': int(n_cols_corrupt),
                'Method': method_names[method],
                'Mean_Accuracy': metrics['mean'],
                'Std_Accuracy': metrics['std']
            })
    
    # Process vary_corrupt_val data
    for corrupt_val, methods_data in dataset_results['vary_corrupt_val'].items():
        for method, metrics in methods_data.items():
            vary_corrupt_val_data.append({
                'Dataset': dataset,
                'Corrupt_Val': float(corrupt_val),
                'Method': method_names[method],
                'Mean_Accuracy': metrics['mean'],
                'Std_Accuracy': metrics['std']
            })

# Create dataframes
df_vary_cols = pd.DataFrame(vary_cols_data)
df_vary_corrupt_val = pd.DataFrame(vary_corrupt_val_data)

# Sort the dataframes
df_vary_cols = df_vary_cols.sort_values(['Dataset', 'N_Cols_Corrupt', 'Method'])
df_vary_corrupt_val = df_vary_corrupt_val.sort_values(['Dataset', 'Corrupt_Val', 'Method'])

# Reset index for both dataframes
df_vary_cols = df_vary_cols.reset_index(drop=True)
df_vary_corrupt_val = df_vary_corrupt_val.reset_index(drop=True)

# Display the first few rows of each dataframe
print("df_vary_cols:")
print(df_vary_cols.head())
print("\ndf_vary_corrupt_val:")
print(df_vary_corrupt_val.head())

# Save the dataframes to CSV files (optional)
df_vary_cols.to_csv('./results/df_vary_cols.csv', index=False)
df_vary_corrupt_val.to_csv('./results/df_vary_corrupt_val.csv', index=False)