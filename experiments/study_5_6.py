# Standard Library Imports
import os
import warnings
import hashlib
import re
import itertools
import pickle

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# OpenAI Imports
from openai import AzureOpenAI

# Pydantic Imports
from pydantic import BaseModel, PrivateAttr

# Project-Specific Imports
from openai_config import get_openai_config
from HLLM import H_LLM



# Import previously discovered values to consider for corruption
col_corrupt_list = [
    ['Age'],
    ['Age', 'HbA1c'],
    ['Age', 'HbA1c', 'FastingGlucose'],
    ['Age', 'HbA1c', 'FastingGlucose', 'BMI'],
    ['Age', 'HbA1c', 'FastingGlucose', 'BMI', 'BloodPressure'],
    ['Age', 'HbA1c', 'FastingGlucose', 'BMI', 'BloodPressure', 'Cholesterol'],
     ['Age', 'HbA1c', 'FastingGlucose', 'BMI', 'BloodPressure', 'Cholesterol', 'Insulin'],
     ['Age', 'HbA1c', 'FastingGlucose', 'BMI', 'BloodPressure', 'Cholesterol', 'Insulin', 'PhysicalActivity'],
]


from sklearn.model_selection import train_test_split

# Load the LLM (Azure Config object)
num_batches_test = 10
config = get_openai_config()
llm = AzureOpenAI(
    api_version=config["api_version"],
    azure_endpoint=config["api_base"],
    api_key=config["api_key"])

context = "The goal is to hypothesize concrete reasons for why the model has underperformed. "
H = H_LLM(config=config, llm=llm, context=context)
config = get_openai_config()


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
def prepare_datasets_with_covariate_shift_and_corruption(n_samples1, n_samples2, coefficients1, coefficients2, noise_params1, noise_params2, seed, prop_outliers=0.2, columns_to_corrupt=None):
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
    outlier_factor = 20
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


def prepare_datasets_with_covariate_shift_and_corruption_with_variability(n_samples1, n_samples2, coefficients1, coefficients2, noise_params1, noise_params2, seed, prop_outliers, columns_to_corrupt=None):
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
    outlier_factor = 20

    if columns_to_corrupt:
        for i, column in enumerate(columns_to_corrupt):
            prop_outliers_col = prop_outliers[i]
            n_outliers = int(prop_outliers_col * len(X2))
            idx_outliers = np.random.choice(X2.index, n_outliers, replace=False)
            #print(idx_outliers)
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

# Example usage:
n_samples1 = 100_000
n_samples2 = 100_000
coefficients1 = [0.3, 0.0075, -0.01, 0.05, 0.04, -0.03, -0.02, -0.1]
coefficients2 = [-0.3, -0.0075, 0.2, -0.05, -0.015, -0.001, 0.02, -2]
noise_params1 = {'mean': 0, 'std': 0.2}
noise_params2 = {'mean': 0, 'std': 0.2}
seed = 42
columns_to_corrupt = ['Age']

X_train, y_train, X_test, y_test, shift_index = prepare_datasets_with_covariate_shift_and_corruption(
    n_samples1, n_samples2, coefficients1, coefficients2, noise_params1, noise_params2, seed, 0, columns_to_corrupt)

X_train = X_train.drop('time', axis=1)
X_test = X_test.drop('time', axis=1)

# Supplementary functions

def train_and_evaluate_conditions(conditions, x_backtest, y_backtest, x_after, y_after):
    accuracies = []
    
    for condition in conditions:
        condition = "not (" + condition + ")"
        # Filter x_backtest and y_backtest using the current condition
        filtered_x_backtest = x_backtest.query(condition)
        filtered_y_backtest = y_backtest[filtered_x_backtest.index]
        
        # Train logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(filtered_x_backtest, filtered_y_backtest)
        
        # Evaluate the model on x_after
        y_pred = model.predict(x_after)
        accuracy = accuracy_score(y_after, y_pred)
        
        # Append the accuracy to the list
        accuracies.append(accuracy)
    
    return accuracies

def convert_queries(input_string):
    # Pattern to extract the queries
    pattern = re.compile(r"Query \d+:\s*(.*?)(?:;|\n|$)", re.DOTALL)
    matches = pattern.findall(input_string)
    
    cleaned_queries = []
    
    for query in matches:
        query = query.strip()  # Remove leading and trailing whitespace
        
        # Remove outer df[...] if present
        outer_df_pattern = re.compile(r"^df\[(?:\((.*)\)|(.+))\]$")
        match_outer = outer_df_pattern.match(query)
        if match_outer:
            # Use the first non-None group
            query = match_outer.group(1) or match_outer.group(2)
        
        # Remove any df['column'] or df["column"] inside the query
        df_column_pattern = re.compile(r"df\[['\"](.*?)['\"]\]")
        query = df_column_pattern.sub(r'\1', query)
        
        cleaned_queries.append(query)
    
    return cleaned_queries

# Queries to test out (given from previous input).

queries = ['HbA1c > 5.35) & (FastingGlucose > 86.09',
 'HbA1c > 5.35) & (Age > 32.18',
 'HbA1c > 5.35) & (BMI > 17.29',
 'HbA1c > 5.35) & (BloodPressure > 97.77',
 'HbA1c > 5.35) & (Cholesterol > 144.80',
 'HbA1c > 5.35) & (Insulin > 45.59',
 'HbA1c > 5.35) & (PhysicalActivity > 1.73',
 'FastingGlucose > 86.09) & (Age > 32.18',
 'FastingGlucose > 86.09) & (BMI > 17.29',
 'FastingGlucose > 86.09) & (BloodPressure > 97.77',
 'FastingGlucose > 86.09) & (Cholesterol > 144.80',
 'FastingGlucose > 86.09) & (Insulin > 45.59',
 'FastingGlucose > 86.09) & (PhysicalActivity > 1.73',
 'Age > 32.18) & (BMI > 17.29',
 'Age > 32.18) & (BloodPressure > 97.77',
 'Age > 32.18) & (Cholesterol > 144.80',
 'Age > 32.18) & (Insulin > 45.59',
 'Age > 32.18) & (PhysicalActivity > 1.73',
 'BMI > 17.29) & (BloodPressure > 97.77',
 'BMI > 17.29) & (Cholesterol > 144.80']

# Experiment with range of values corrupted
accs_dict = {}
cols_corrupt = col_corrupt_list[-1]
props_consider = [2, 5, 10, 20, 50, 75]
for i, prop in enumerate(props_consider): 
    prop_outliers = np.random.randint(1, prop, 8) / 100

    X_train, y_train, X_test, y_test, shift_index = prepare_datasets_with_covariate_shift_and_corruption_with_variability(
        n_samples1, n_samples2, coefficients1, coefficients2, noise_params1, noise_params2, seed, prop_outliers, cols_corrupt )
    X_train = X_train.drop('time', axis=1)
    X_test = X_test.drop('time', axis=1)

    # Load the LLM (Azure Config object)
    num_batches_test = 10
    config = get_openai_config()
    llm = AzureOpenAI(
        api_version=config["api_version"],
        azure_endpoint=config["api_base"],
        api_key=config["api_key"]
    )

    context = "The goal is to hypothesize concrete reasons for why the model has underperformed. "
    H = H_LLM(config=config, llm=llm, context=context)

    # Split the test data into batches
    X_test_batches = np.array_split(X_test, num_batches_test)
    y_test_batches = np.array_split(y_test, num_batches_test)
    model = LogisticRegression()


    x_before = X_test_batches[0]
    y_before = y_test_batches[0]
    x_after = X_test_batches[-1]
    y_after = y_test_batches[-1]
    x_backtest = X_test_batches[-2]
    y_backtest = y_test_batches[-2]
    # Fit the model on the training data
    model.fit(x_backtest, y_backtest)

    y_preds = model.predict(x_after)
    print(accuracy_score(y_preds, y_after))
    accuracies = train_and_evaluate_conditions(queries, x_backtest, y_backtest, x_after, y_after)
    accs_dict[i] = accuracies
    print(accuracies)


# Ensure the ./results/ directory exists
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)

# Define the file path for saving the pickle file
output_file_path = os.path.join(results_dir, "accs_dict.pkl")

# Export accs_dict as a pickle file
with open(output_file_path, "wb") as file:
    pickle.dump(accs_dict, file)

### --- backtesting window ----

accs_dict = {}
cols_corrupt = col_corrupt_list[-1]
prop = 5
batch_lengths = {}

for n_samples2 in [10_000_000]:
    prop_outliers = np.random.randint(1, prop, 8) / 100
    prop_outliers = np.ones(8) * 5 / 100

    X_train, y_train, X_test, y_test, shift_index = prepare_datasets_with_covariate_shift_and_corruption_with_variability(
        n_samples1, n_samples2, coefficients1, coefficients2, noise_params1, noise_params2, seed, prop_outliers, cols_corrupt )
    X_train = X_train.drop('time', axis=1)
    X_test = X_test.drop('time', axis=1)

    # Load the LLM (Azure Config object)
    num_batches_test = 10
    config = get_openai_config()
    llm = AzureOpenAI(
        api_version=config["api_version"],
        azure_endpoint=config["api_base"],
        api_key=config["api_key"]
    )

    context = "The goal is to hypothesize concrete reasons for why the model has underperformed. "
    H = H_LLM(config=config, llm=llm, context=context)

    # Split the test data into batches
    X_test_batches = np.array_split(X_test, num_batches_test)
    y_test_batches = np.array_split(y_test, num_batches_test)
    model = LogisticRegression()

print(f"Accuracy dictionary successfully saved to for corruption values{output_file_path}")


queries = ['FastingGlucose > 376.145108',
 'Insulin > 320.642677',
 'HbA1c > 21.553946',
 'Age > 187.805319',
 'BMI > 93.998780',
 'BloodPressure > 452.899287',
 'Cholesterol > 757.675355',
 'PhysicalActivity > 11.314583',
 '(HbA1c > 21.553946) & (FastingGlucose > 376.145108)',
 '(Age > 187.805319) & (BMI > 93.998780)',
 '(BloodPressure > 452.899287) & (Cholesterol > 757.675355)',
 '(Insulin > 320.642677) & (PhysicalActivity > 11.314583)',
 '(HbA1c > 21.553946) & (FastingGlucose > 376.145108) & (Age > 187.805319)',
 '(BMI > 93.998780) & (BloodPressure > 452.899287) & (Cholesterol > 757.675355)',
 '(Insulin > 320.642677) & (PhysicalActivity > 11.314583) & (HbA1c > 21.553946)',
 '(FastingGlucose > 376.145108) & (Age > 187.805319) & (BMI > 93.998780)',
 '(BloodPressure > 452.899287) & (Cholesterol > 757.675355) & (Insulin > 320.642677)',
 '(PhysicalActivity > 11.314583) & (HbA1c > 21.553946) & (FastingGlucose > 376.145108)',
 '(Age > 187.805319) & (BMI > 93.998780) & (BloodPressure > 452.899287)']



# Initialize accs_dict to store accuracies for different backtest sizes
accs_dict = {}

for backtest_size in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100_000]:
    # Sample x_backtest and y_backtest based on backtest_size
    if backtest_size <= len(X_test_batches[-2]):
        sample_indices = np.random.choice(len(X_test_batches[-2]), backtest_size, replace=False)
        x_backtest_sampled = X_test_batches[-2].iloc[sample_indices]
        y_backtest_sampled = y_test_batches[-2].iloc[sample_indices]
    else:
        # If backtest_size is larger than the available data, use the entire batch
        x_backtest_sampled = X_test_batches[-2]
        y_backtest_sampled = y_test_batches[-2]

    # Define the x_before, y_before, x_after, y_after for reference
    x_before = X_test_batches[0]
    y_before = y_test_batches[0]
    x_after = X_test_batches[-1]
    y_after = y_test_batches[-1]

    # Fit the model on the sampled backtest data
    model.fit(x_backtest_sampled, y_backtest_sampled)

    # Predict using the model on the x_after data
    y_preds = model.predict(x_after)

    # Calculate accuracies using the sampled backtest data
    accuracies = train_and_evaluate_conditions(queries, x_backtest_sampled, y_backtest_sampled, x_after, y_after)

    # Attach the accuracies to accs_dict with backtest_size as the key
    accs_dict[backtest_size] = accuracies

    # Print the accuracies for reference
    print(f"Backtest size: {backtest_size}, Accuracies: {accuracies}")

# Optionally, print the entire accs_dict to see all accuracies for different backtest sizes
print(accs_dict)

# Define the file path for saving the pickle file
output_file_path = os.path.join(results_dir, "accs_dict_backtesting_window.pkl")

# Export accs_dict as a pickle file
with open(output_file_path, "wb") as file:
    pickle.dump(accs_dict, file)

print(f"Accuracy dictionary successfully saved to for backtesting window: {output_file_path}")
