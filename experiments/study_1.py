import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from river import drift as river_drift

import helpers  # Ensure this module contains the required helper functions
import importlib as imp
imp.reload(helpers)

from helpers import (
    adaptive_batch_self_healing,
    adaptive_ensemble_with_ddm,
    adaptive_batch_with_sliding_window,
    batch_learning_no_retraining,
    adaptive_batch_with_drift_detection_ddm
)

import HLLM
from HLLM import H_LLM

from openai_config import get_openai_config
from openai import AzureOpenAI

# ===============================
# Configuration and Parameters
# ===============================

# Set drift detection parameters
drift_threshold = 1.1
warm_start = False  # Set to True if warm start is desired

# Define corruption coefficients and columns to corrupt
corruption_coeffs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
cols_corrupt = ['Age', 'HbA1c', 'FastingGlucose']
seeds = range(5)  # Seeds 0 to 4

# Model and LLM Configuration
model = LogisticRegression()
config = get_openai_config()
llm = AzureOpenAI(
    api_version=config["api_version"],
    azure_endpoint=config["api_base"],
    api_key=config["api_key"]
)
context = "The goal is to hypothesize concrete reasons for why the model has underperformed."
H = H_LLM(config=config, llm=llm, context=context)

# Number of test batches
num_batches_test = 10

# ===============================
# Helper Functions
# ===============================

def generate_diabetes_data(n_samples, noise_distribution_params, coefficients, seed):
    """
    Generates synthetic diabetes data with specified noise and coefficients.
    """
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

def prepare_datasets_with_covariate_shift_and_corruption(
    n_samples1, n_samples2, coefficients1, coefficients2,
    noise_params1, noise_params2, seed, prop_outliers=0.2,
    columns_to_corrupt=None
):
    """
    Prepares training and testing datasets with covariate shift and specified corruption.
    """
    # Generate the first dataset
    X1, y1 = generate_diabetes_data(n_samples1, noise_params1, coefficients1, seed)
    
    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X1, y1, test_size=0.7, random_state=42
    )
    
    # Generate the second dataset
    X2, y2 = generate_diabetes_data(n_samples2, noise_params2, coefficients2, seed)
    
    # Introduce outliers in specified columns
    outlier_factor = 20
    n_outliers = int(prop_outliers * len(X2))
    for i, column in enumerate(columns_to_corrupt):
        idx_outliers = np.random.choice(X2.index, n_outliers, replace=False)
        np.random.seed(seed + i)  # Ensure reproducibility for each column
        X2.loc[idx_outliers, column] *= outlier_factor
        y2.loc[idx_outliers] = np.random.binomial(1, 0.5, n_outliers)
    
    # Concatenate test data with the second dataset
    X_test = pd.concat([X_test, X2], ignore_index=True)
    y_test = pd.concat([y_test, y2], ignore_index=True)
    
    # Add time indices
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['time'] = np.arange(len(X_train))
    X_test['time'] = np.arange(len(X_train), len(X_train) + len(X_test))
    
    # Determine shift index
    shift_index = len(X_test) - len(X2)
    
    return X_train, y_train, X_test, y_test, shift_index

def save_results(results_dict, corr_coef, corrupt, seed):
    """
    Saves the results dictionary to a pickle file.
    """
    filename = f'results/results_corr_{corr_coef}_corrupt_{corrupt}_seed_{seed}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)

# ===============================
# Setup Logging and Directories
# ===============================

# Configure logging
logging.basicConfig(
    filename='overnight_run.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# ===============================
# Main Processing Loop
# ===============================

for corr_coef in corruption_coeffs:
    for corrupt in cols_corrupt:
        for seed in seeds:
            print(f'Processing: corr_coef={corr_coef}, corrupt={corrupt}, seed={seed}')
    
            try:
                # Reset LLM cache
                H.clear_cache()
                
                # Prepare datasets
                X_train, y_train, X_test, y_test, shift_index = prepare_datasets_with_covariate_shift_and_corruption(
                    n_samples1=100_000,
                    n_samples2=100_000,
                    coefficients1=[0.3, 0.0075, -0.01, 0.05, 0.04, -0.03, -0.02, -0.1],
                    coefficients2=[-0.3, -0.0075, 0.2, -0.05, -0.015, -0.001, 0.02, -2],
                    noise_params1={'mean': 0, 'std': 0.2},
                    noise_params2={'mean': 0, 'std': 0.2},
                    seed=seed,
                    prop_outliers=corr_coef,
                    columns_to_corrupt=[corrupt]
                )
                
                # Remove 'time' column
                X_train = X_train.drop('time', axis=1)
                X_test = X_test.drop('time', axis=1)
    
                logging.info(f'Starting seed={seed}, corruption={corrupt}, coeff={corr_coef}')
    
                # Information dictionary
                info_dict = {
                    'seed': seed,
                    'corrupt': corrupt,
                    'corr_coef': corr_coef
                }
    
                # Initialize drift detector
                drift_detector = river_drift.binary.DDM(drift_threshold=drift_threshold, warm_start=warm_start)
                
                # Run the models
                t1, m1 = batch_learning_no_retraining(model, X_train, y_train, X_test, y_test, num_batches_test)
                t2, m2 = adaptive_batch_with_sliding_window(model, X_train, y_train, X_test, y_test, num_batches_test)
                t3, m3 = adaptive_batch_with_drift_detection_ddm(
                    model, X_train, y_train, X_test, y_test, num_batches_test, drift_detector
                )
                
                # Re-initialize drift detector for ensemble method
                drift_detector = river_drift.binary.DDM(drift_threshold=drift_threshold, warm_start=warm_start)
                t4, m4 = adaptive_ensemble_with_ddm(
                    model, X_train, y_train, X_test, y_test, num_batches_test, drift_detector
                )
                
                # Re-initialize drift detector for self-healing
                drift_detector = river_drift.binary.DDM(drift_threshold=drift_threshold, warm_start=warm_start)
                t5, m5 = adaptive_batch_self_healing(
                    H, model, X_train, y_train, X_test, y_test, num_batches_test, drift_detector
                )
    
                # Compile results
                results_dict = {
                    'info': info_dict,
                    't1': t1, 'm1': m1,
                    't2': t2, 'm2': m2,
                    't3': t3, 'm3': m3,
                    't4': t4, 'm4': m4,
                    't5': t5, 'm5': m5
                }
    
                # Save intermediate results
                save_results(results_dict, corr_coef, corrupt, seed)
    
                logging.info(f'Finished seed={seed}, corruption={corrupt}, coeff={corr_coef}')
                
            except Exception as e:
                logging.error(f'Error with seed={seed}, corruption={corrupt}, coeff={corr_coef}: {e}')

# ===============================
# Aggregating Results
# ===============================

# Initialize list to store aggregated results
results_list = []

# Define method names and corresponding keys
method_names = {
    'No retraining': 'm1',
    'Partially Updating': 'm2',
    'New model training': 'm3',
    'Ensemble Method': 'm4',
    '$\\mathcal{H}$-LLM': 'm5'
}

# Iterate over all combinations to load and aggregate results
for corr_coef in corruption_coeffs:
    for corrupt in cols_corrupt:
        for seed in seeds:
            try:
                # Load the individual result file
                filepath = f'results/results_corr_{corr_coef}_corrupt_{corrupt}_seed_{seed}.pkl'
                result = pd.read_pickle(filepath)
                
                # Aggregate results for each method
                for method, m_key in method_names.items():
                    m_values = result[m_key]
                    mean_m = np.mean(m_values[-150:])  # Mean of the last 150 elements
                    
                    # Append to results list
                    results_list.append({
                        'Method': method,
                        'corr_coef': corr_coef,
                        'corrupt': corrupt,
                        'seed': seed,
                        'mean_result': mean_m
                    })
            except Exception as e:
                print(f'Error processing file {filepath}: {e}')

# Create a DataFrame from the aggregated results
results_df = pd.DataFrame(results_list)

# Ensure the 'results/' directory exists
os.makedirs('results', exist_ok=True)

# Save the aggregated DataFrame
results_df.to_pickle('results/study1_results.pkl')
results_df.to_csv('results/study1_results.csv', index=False)

# Log completion
logging.info('Run completed.')

# Display the DataFrame
print(results_df)
