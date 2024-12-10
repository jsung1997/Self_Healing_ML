import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

def prepare_datasets_with_covariate_shift_and_corruption_with_variability(
    n_samples1, n_samples2, coefficients1, coefficients2, noise_params1, noise_params2, seed, prop_outliers, columns_to_corrupt=None):
    X1, y1 = generate_diabetes_data(n_samples1, noise_params1, coefficients1, seed)
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.7, random_state=42)
    X2, y2 = generate_diabetes_data(n_samples2, noise_params2, coefficients2, seed)
    
    outlier_factor = 20
    if columns_to_corrupt:
        for i, column in enumerate(columns_to_corrupt):
            prop_outliers_col = prop_outliers[i]
            n_outliers = int(prop_outliers_col * len(X2))
            idx_outliers = np.random.choice(X2.index, n_outliers, replace=False)
            X2.loc[idx_outliers, column] *= outlier_factor
            y2.loc[idx_outliers] = np.random.binomial(1, 0.5, n_outliers)
    
    X_test = pd.concat([X_test, X2], ignore_index=True)
    y_test = pd.concat([y_test, y2], ignore_index=True)
    X_train['time'] = np.arange(len(X_train))
    X_test['time'] = np.arange(len(X_train), len(X_train) + len(X_test))
    shift_index = len(X_test) - len(X2)
    return X_train.drop('time', axis=1), y_train, X_test.drop('time', axis=1), y_test, shift_index
