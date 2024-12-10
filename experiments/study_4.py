import os
import pickle
import logging
import math
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from helpers import (
    adaptive_batch_self_healing,
    adaptive_ensemble_with_ddm,
    adaptive_batch_with_sliding_window,
    batch_learning_no_retraining,
    visualize_all,
    adaptive_batch_with_drift_detection_ddm
)
from HLLM import H_LLM
from openai import AzureOpenAI
from openai_config import get_openai_config

# Constants and Configuration
DATA_DIR = "results"
LOG_FILE = "experiment_log.log"
NUM_BATCHES_TEST = 10

# Ensure the results directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def setup_logging(log_file: str = LOG_FILE) -> None:
    """
    Configure the logging settings.

    Parameters:
    - log_file: Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def generate_diabetes_data(
    n_samples: int,
    noise_distribution_params: Dict[str, float],
    coefficients: List[float],
    seed: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic diabetes dataset.

    Parameters:
    - n_samples: Number of samples to generate.
    - noise_distribution_params: Parameters for the noise distribution.
    - coefficients: Coefficients for the linear combination.
    - seed: Random seed for reproducibility.

    Returns:
    - X: Features DataFrame.
    - y: Outcome Series.
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
    
    X = np.vstack((
        HbA1c,
        FastingGlucose,
        Age,
        BMI,
        BloodPressure,
        Cholesterol,
        Insulin,
        PhysicalActivity
    )).T
    noise = np.random.normal(
        noise_distribution_params['mean'],
        noise_distribution_params['std'],
        n_samples
    )
    linear_combination = np.dot(X, coefficients) + noise
    probabilities = 1 / (1 + np.exp(-linear_combination))
    adjusted_probabilities = np.clip(probabilities, 0.4, 0.6)
    outcomes = np.where(adjusted_probabilities > 0.5, 1, 0)
    variables = [
        'HbA1c', 'FastingGlucose', 'Age', 'BMI',
        'BloodPressure', 'Cholesterol', 'Insulin', 'PhysicalActivity'
    ]
    data = pd.DataFrame(X, columns=variables)
    data['Outcome'] = outcomes
    return data.drop('Outcome', axis=1), data['Outcome']

def prepare_datasets_with_covariate_shift_and_corruption(
    n_samples1: int,
    n_samples2: int,
    coefficients1: List[float],
    coefficients2: List[float],
    noise_params1: Dict[str, float],
    noise_params2: Dict[str, float],
    seed: int,
    prop_outliers: float = 0.2,
    columns_to_corrupt: Optional[List[str]] = None,
    outlier_factor: float = 20.0
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int]:
    """
    Prepare datasets with covariate shift and corruption.

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
    - outlier_factor: Factor by which to multiply corrupted columns.

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
    X_train, X_test, y_train, y_test = train_test_split(
        X1, y1, test_size=0.7, random_state=42
    )
    
    # Generate the second dataset
    X2, y2 = generate_diabetes_data(n_samples2, noise_params2, coefficients2, seed)
    
    # Introduce outliers in the second dataset
    n_outliers = int(prop_outliers * len(X2))
    if columns_to_corrupt:
        idx_outliers = np.random.choice(X2.index, n_outliers, replace=False)
        for column in columns_to_corrupt:
            X2.loc[idx_outliers, column] *= outlier_factor
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

def convert_string_to_dict(input_str: str) -> Dict[str, float]:
    """
    Convert a hypothesis string to a dictionary.

    Parameters:
    - input_str: Input string containing hypotheses and probabilities.

    Returns:
    - result_dict: Dictionary mapping covariates to probabilities.
    """
    result_dict = {}
    lines = input_str.strip().split('\n')
    
    for line in lines:
        if ';' not in line:
            continue
        hypothesis_part, probability_part = line.split(';')
        try:
            covariate = hypothesis_part.split(': ')[1].strip('[]')
            probability_str = probability_part.split(': ')[1].strip('%')
            probability = float(probability_str) / 100
            result_dict[covariate] = probability
        except (IndexError, ValueError):
            continue
    
    return result_dict

def compute_entropy(distribution: Dict[str, float]) -> float:
    """
    Compute the entropy of a given distribution.

    Parameters:
    - distribution: Dictionary mapping items to their probabilities.

    Returns:
    - entropy: Calculated entropy value.
    """
    entropy = 0.0
    for prob in distribution.values():
        if prob > 0:
            entropy -= prob * math.log(prob)
    return entropy

def compute_kl_divergence(distribution_p: Dict[str, float], distribution_q: Dict[str, float]) -> float:
    """
    Compute the KL divergence between two distributions.

    Parameters:
    - distribution_p: First probability distribution.
    - distribution_q: Second probability distribution.

    Returns:
    - kl_divergence: Calculated KL divergence.
    """
    kl_divergence = 0.0
    for key, p_prob in distribution_p.items():
        q_prob = distribution_q.get(key, 0)
        if p_prob > 0 and q_prob > 0:
            kl_divergence += p_prob * math.log(p_prob / q_prob)
    return kl_divergence

def save_results(
    results_dict: Dict,
    filename: str
) -> None:
    """
    Save results to a pickle file.

    Parameters:
    - results_dict: Dictionary containing the results.
    - filename: Filename to save the results.
    """
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(results_dict, f)

def initialize_llm() -> H_LLM:
    """
    Initialize the Language Model (LLM) for hypothesis generation.

    Returns:
    - H: Initialized H_LLM instance.
    """
    config = get_openai_config()
    llm = AzureOpenAI(
        api_version=config["api_version"],
        azure_endpoint=config["api_base"],
        api_key=config["api_key"]
    )
    context = "The goal is to hypothesize concrete reasons for why the model has underperformed."
    H = H_LLM(config=config, llm=llm, context=context)
    return H

def run_corruption_experiment(
    model,
    H: H_LLM,
    corrupt_coefs: List[float],
    seeds: List[int],
    params: Dict
) -> List[Dict]:
    """
    Run experiments varying corruption coefficients.

    Parameters:
    - model: Machine learning model to train.
    - H: Initialized H_LLM instance.
    - corrupt_coefs: List of corruption coefficients (proportion of outliers).
    - seeds: List of random seeds.
    - params: Dictionary of experiment parameters.

    Returns:
    - results: List of result dictionaries.
    """
    results = []
    for corrupt_coef in corrupt_coefs:
        for seed in seeds:
            try:
                # Prepare the datasets with covariate shift and corruption
                X_train, y_train, X_test, y_test, shift_index = prepare_datasets_with_covariate_shift_and_corruption(
                    n_samples1=params['n_samples1'],
                    n_samples2=params['n_samples2'],
                    coefficients1=params['coefficients1'],
                    coefficients2=params['coefficients2'],
                    noise_params1=params['noise_params1'],
                    noise_params2=params['noise_params2'],
                    seed=seed,
                    prop_outliers=corrupt_coef,
                    columns_to_corrupt=params['columns_to_corrupt'],
                    outlier_factor=params['outlier_factor']
                )

                # Drop 'time' column if it exists
                X_train = X_train.drop(columns=['time'], errors='ignore')
                X_test = X_test.drop(columns=['time'], errors='ignore')

                # Split test data into batches
                X_test_batches = np.array_split(X_test, NUM_BATCHES_TEST)
                y_test_batches = np.array_split(y_test, NUM_BATCHES_TEST)

                # Fit the model on the training data
                model.fit(X_train, y_train)

                x_before = X_test_batches[0]
                y_before = y_test_batches[0]
                x_after = X_test_batches[-1]
                y_after = y_test_batches[-1]

                # Hypothesize issues with performance
                covariate_guesses = H.hypothesize_issues_with_performance(
                    x_before, x_after, y_before, y_after, model, H.context
                )
                hypotheses = H.summarize_probabilities(
                    covariate_guesses,
                    x_before,
                    x_after,
                    context="Your goal is to provide a summary of probabilities on likelihood of each of the covariates resulting in the model failing"
                )
                try:
                    items = convert_string_to_dict(hypotheses)
                    # Compute entropy and KL divergence
                    d_quality = compute_entropy(items)
                    kl_quality = compute_kl_divergence(params['true_probabilities'], items)
                except Exception as e:
                    logging.warning(f"Failed to compute metrics for corrupt_coef={corrupt_coef}, seed={seed}: {e}")
                    items = None
                    d_quality = None
                    kl_quality = None

                # Log the results
                result = {
                    'experiment_type': 'corruption',
                    'corrupt_coef': corrupt_coef,
                    'seed': seed,
                    'covariate_guesses': covariate_guesses,
                    'hypotheses': hypotheses,
                    'd_quality': d_quality,
                    'kl_quality': kl_quality
                }
                logging.info(result)
                print(result)

                # Save the results
                filename = f'results/quality_of_info_corrupt_{corrupt_coef}_seed_{seed}.pkl'
                save_results(result, filename)
                results.append(result)

            except Exception as e:
                error_message = f"Error for corrupt_coef={corrupt_coef}, seed={seed}: {e}"
                print(error_message)
                logging.error(error_message)
    return results

def run_outlier_experiment(
    model,
    H: H_LLM,
    outlier_factors: List[float],
    seeds: List[int],
    params: Dict
) -> List[Dict]:
    """
    Run experiments varying outlier factors.

    Parameters:
    - model: Machine learning model to train.
    - H: Initialized H_LLM instance.
    - outlier_factors: List of outlier factors to apply.
    - seeds: List of random seeds.
    - params: Dictionary of experiment parameters.

    Returns:
    - results: List of result dictionaries.
    """
    results = []
    for outlier_factor in outlier_factors:
        for seed in seeds:
            try:
                # Prepare the datasets with covariate shift and corruption
                X_train, y_train, X_test, y_test, shift_index = prepare_datasets_with_covariate_shift_and_corruption(
                    n_samples1=params['n_samples1'],
                    n_samples2=params['n_samples2'],
                    coefficients1=params['coefficients1'],
                    coefficients2=params['coefficients2'],
                    noise_params1=params['noise_params1'],
                    noise_params2=params['noise_params2'],
                    seed=seed,
                    prop_outliers=params['prop_outliers'],  # Fixed proportion
                    columns_to_corrupt=params['columns_to_corrupt'],
                    outlier_factor=outlier_factor
                )

                # Drop 'time' column if it exists
                X_train = X_train.drop(columns=['time'], errors='ignore')
                X_test = X_test.drop(columns=['time'], errors='ignore')

                # Split test data into batches
                X_test_batches = np.array_split(X_test, NUM_BATCHES_TEST)
                y_test_batches = np.array_split(y_test, NUM_BATCHES_TEST)

                # Fit the model on the training data
                model.fit(X_train, y_train)

                x_before = X_test_batches[0]
                y_before = y_test_batches[0]
                x_after = X_test_batches[-1]
                y_after = y_test_batches[-1]

                # Hypothesize issues with performance
                covariate_guesses = H.hypothesize_issues_with_performance(
                    x_before, x_after, y_before, y_after, model, H.context
                )
                hypotheses = H.summarize_probabilities(
                    covariate_guesses,
                    x_before,
                    x_after,
                    context="Your goal is to provide a summary of probabilities on likelihood of each of the covariates resulting in the model failing"
                )
                try:
                    items = convert_string_to_dict(hypotheses)
                    # Compute entropy and KL divergence
                    d_quality = compute_entropy(items)
                    kl_quality = compute_kl_divergence(params['true_probabilities'], items)
                except Exception as e:
                    logging.warning(f"Failed to compute metrics for outlier_factor={outlier_factor}, seed={seed}: {e}")
                    items = None
                    d_quality = None
                    kl_quality = None

                # Log the results
                result = {
                    'experiment_type': 'outlier',
                    'outlier_factor': outlier_factor,
                    'seed': seed,
                    'covariate_guesses': covariate_guesses,
                    'hypotheses': hypotheses,
                    'd_quality': d_quality,
                    'kl_quality': kl_quality
                }
                logging.info(result)
                print(result)

                # Save the results
                filename = f'results/quality_of_info_outlier_{outlier_factor}_seed_{seed}.pkl'
                save_results(result, filename)
                results.append(result)

            except Exception as e:
                error_message = f"Error for outlier_factor={outlier_factor}, seed={seed}: {e}"
                print(error_message)
                logging.error(error_message)
    return results

def main():
    """
    Main function to execute the experiments.
    """
    # Setup logging
    setup_logging()

    # Define experiment parameters
    experiment_params = {
        'n_samples1': 20_000,
        'n_samples2': 100_000,
        'coefficients1': [0.3, 0.0075, -0.01, 0.05, 0.04, -0.03, -0.02, -0.1],
        'coefficients2': [-0.3, -0.0075, 0.2, -0.05, -0.015, -0.001, 0.02, -2],
        'noise_params1': {'mean': 0, 'std': 0.2},
        'noise_params2': {'mean': 0, 'std': 0.2},
        'columns_to_corrupt': ['Age'],
        'outlier_factor': 2.0,
        'prop_outliers': 0.05,
        'true_probabilities': {
            'Age': 1,
            'HbA1c': 0,
            'FastingGlucose': 0,
            'BMI': 0,
            'BloodPressure': 0,
            'Cholesterol': 0,
            'Insulin': 0,
            'PhysicalActivity': 0
        }
    }

    # Initialize the Language Model
    H = initialize_llm()

    # Initialize the machine learning model
    model = LogisticRegression()

    # Define experiment variations
    corrupt_coefs = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    outlier_factors = [1, 1.05, 1.1, 1.2, 1.5, 2, 4, 5, 10, 20, 100, 1000]
    seeds = [42, 43, 44, 45, 46]

    # Run corruption experiments
    logging.info("Starting corruption experiments.")
    print("Starting corruption experiments...")
    corruption_results = run_corruption_experiment(
        model=model,
        H=H,
        corrupt_coefs=corrupt_coefs,
        seeds=seeds,
        params=experiment_params
    )
    logging.info("Completed corruption experiments.")
    print("Completed corruption experiments.")

    # Run outlier experiments
    logging.info("Starting outlier experiments.")
    print("Starting outlier experiments...")
    outlier_results = run_outlier_experiment(
        model=model,
        H=H,
        outlier_factors=outlier_factors,
        seeds=seeds,
        params=experiment_params
    )
    logging.info("Completed outlier experiments.")
    print("Completed outlier experiments.")

    # Optionally, save all results together
    all_results = {
        'corruption_experiments': corruption_results,
        'outlier_experiments': outlier_results
    }
    all_results_filename = 'results/all_experiment_results.pkl'
    save_results(all_results, all_results_filename)
    logging.info(f"All experiment results saved to {all_results_filename}")
    print(f"All experiment results saved to {all_results_filename}")

if __name__ == '__main__':
    main()
