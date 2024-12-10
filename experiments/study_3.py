import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from tqdm import tqdm
import matplotlib.pyplot as plt

from river import metrics, stream, tree, neighbors, naive_bayes, ensemble, linear_model, drift

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


# Experiment parameters
N_SAMPLES_1 = 50_000
N_SAMPLES_2 = 40_000
COEFFICIENTS_1 = [0.3, 0.0075, -0.01, 0.05, 0.04, -0.03, -0.02, -0.1]
COEFFICIENTS_2 = [-0.3, -0.0075, 0.2, -0.05, -0.015, -0.001, 0.02, -2]
NOISE_PARAMS_1 = {'mean': 0, 'std': 0.2}
NOISE_PARAMS_2 = {'mean': 0, 'std': 0.2}
COLUMNS_TO_CORRUPT = ['Age', 'BMI']
MODEL = LogisticRegression()
NUM_BATCHES_TEST = 400
PROP_OUTLIERS = 0.2
THRESHOLDS = [0.5, 0.75, 0.9, 1, 1.1, 1.2, 1.5, 2, 2.5, 3, 5, 10, 15, 20]
SEEDS = range(10)
WARM_START = 5


def generate_diabetes_data(n_samples, noise_distribution_params, coefficients, seed):
    """
    Generate a synthetic dataset simulating diabetes-related data.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    noise_distribution_params : dict
        Dictionary containing 'mean' and 'std' for noise distribution.
    coefficients : list of float
        Coefficients applied to generated features.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : pd.DataFrame
        Generated features.
    y : pd.Series
        Binary outcomes derived from logistic transformation.
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
        'HbA1c', 'FastingGlucose', 'Age', 'BMI', 'BloodPressure',
        'Cholesterol', 'Insulin', 'PhysicalActivity'
    ]

    data = pd.DataFrame(X, columns=variables)
    data['Outcome'] = outcomes
    return data.drop('Outcome', axis=1), data['Outcome']


def prepare_datasets_with_covariate_shift_and_corruption(
    n_samples1, n_samples2, coefficients1, coefficients2,
    noise_params1, noise_params2, seed,
    prop_outliers=0.2, columns_to_corrupt=None,
    outlier_factor=20
):
    """
    Prepare training and testing sets with artificial covariate shifts and corrupted outliers.

    Parameters
    ----------
    n_samples1 : int
        Size of the initial dataset.
    n_samples2 : int
        Size of the shifted dataset.
    coefficients1 : list of float
        Coefficients for initial data generation.
    coefficients2 : list of float
        Coefficients for shifted data generation.
    noise_params1 : dict
        Noise parameters for initial data.
    noise_params2 : dict
        Noise parameters for shifted data.
    seed : int
        Random seed.
    prop_outliers : float
        Proportion of outliers to introduce in the shifted dataset.
    columns_to_corrupt : list of str
        Columns to corrupt with outliers.
    outlier_factor : float
        Multiplicative factor for corruption.

    Returns
    -------
    X_train : pd.DataFrame
    y_train : pd.Series
    X_test : pd.DataFrame
    y_test : pd.Series
    shift_index : int
        Index where the shift begins in the test dataset.
    """
    X1, y1 = generate_diabetes_data(n_samples1, noise_params1, coefficients1, seed)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X1, y1, test_size=0.7, random_state=42
    )

    X2, y2 = generate_diabetes_data(n_samples2, noise_params2, coefficients2, seed)

    n_outliers = int(prop_outliers * len(X2))
    if columns_to_corrupt:
        idx_outliers = np.random.choice(X2.index, n_outliers, replace=False)
        for column in columns_to_corrupt:
            X2.loc[idx_outliers, column] *= outlier_factor
        y2.loc[idx_outliers] = np.random.binomial(1, 0.5, n_outliers)

    X_test = pd.concat([X_temp, X2], ignore_index=True)
    y_test = pd.concat([y_temp, y2], ignore_index=True)

    X_train['time'] = np.arange(len(X_train))
    X_test['time'] = np.arange(len(X_train), len(X_train) + len(X_test))

    shift_index = len(X_test) - len(X2)
    return X_train, y_train, X_test, y_test, shift_index


def save_results(results_dict, threshold, warm_start, seed):
    """
    Save experiment results to a file.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing experiment results.
    threshold : float
        Drift detection threshold value.
    warm_start : int
        Warm start value for the drift detector.
    seed : int
        Random seed used for reproducibility.
    """
    filename = f'results/results_threshold_{threshold}_warm_start_{warm_start}_seed_{seed}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)


# Configure logging
logging.basicConfig(
    filename='experiment_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create results directory if it does not exist
if not os.path.exists('results'):
    os.makedirs('results')

# Load the Large Language Model configuration
openai_config = get_openai_config()
llm = AzureOpenAI(
    api_version=openai_config["api_version"],
    azure_endpoint=openai_config["api_base"],
    api_key=openai_config["api_key"]
)

context = "The goal is to hypothesize concrete reasons for why the model has underperformed."
H = H_LLM(config=openai_config, llm=llm, context=context)

results_dict_outer = {}

# Main experiment execution
for threshold in THRESHOLDS:
    for seed in SEEDS:
        try:
            X_train, y_train, X_test, y_test, shift_index = prepare_datasets_with_covariate_shift_and_corruption(
                N_SAMPLES_1,
                N_SAMPLES_2,
                COEFFICIENTS_1,
                COEFFICIENTS_2,
                NOISE_PARAMS_1,
                NOISE_PARAMS_2,
                seed,
                PROP_OUTLIERS,
                COLUMNS_TO_CORRUPT
            )

            # Remove time column before modeling
            X_train = X_train.drop('time', axis=1)
            X_test = X_test.drop('time', axis=1)

            logging.info(f'Beginning experiment: seed={seed}, threshold={threshold}, warm_start={WARM_START}')

            info_dict = {
                'seed': seed,
                'threshold': threshold,
                'warm_start': WARM_START
            }

            drift_detector = drift.binary.DDM(drift_threshold=threshold, warm_start=WARM_START)

            # Run adaptive batch self-healing experiment
            t5, m5 = adaptive_batch_self_healing(
                H, MODEL, X_train, y_train, X_test, y_test, NUM_BATCHES_TEST, drift_detector
            )

            results_dict = {
                'info': info_dict,
                't5': t5,
                'm5': m5
            }
            results_dict_outer[(threshold, WARM_START, seed)] = results_dict

            # Save intermediate results
            save_results(results_dict, threshold, WARM_START, seed)

            logging.info(f'Completed experiment: seed={seed}, threshold={threshold}, warm_start={WARM_START}')

        except Exception as e:
            logging.error(
                f'Error during experiment: seed={seed}, threshold={threshold}, warm_start={WARM_START}: {e}'
            )
