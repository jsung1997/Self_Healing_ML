import pandas as pd
import numpy as np
import os
from scipy.io import arff

def transform_dataset(df, n):
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        # Get the top n most frequent values
        top_n_values = df[col].value_counts().index[:n]
        
        # Replace less frequent values with 'Other'
        df[col] = df[col].apply(lambda x: x if x in top_n_values else f'{col}_Other')
    
    df = pd.get_dummies(df, drop_first=True)    
    return df

####################### LOAD airlines DATASET ########################
dir_path = '../data/imported_datasets/'
file_path = os.path.join(dir_path, 'airlines', 'airlines.arff')

# Load dataset
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

outcome_1 = df.Delay.unique()[0]
df['Outcome'] = df['Delay'] == outcome_1
assert df.Outcome.value_counts(normalize=True).min() > 0.1, 'Less than 10 percent samples belong to this group'

n = 10  # Define the number of top categories to keep
df_airlines = transform_dataset(df, n)
# Drop time, use index as a proxy
df_airlines = df_airlines.drop('Time', axis=1)
X_airlines = df_airlines.drop('Outcome', axis=1)
X_airlines = X_airlines.drop(columns=["Delay_b'1'"])
y_airlines = df_airlines.Outcome

####################### LOAD POKER DATASET ########################
dir_path = '../data/imported_datasets/'
file_path = os.path.join(dir_path, 'poker', 'poker-lsn.arff')

# Load dataset
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Change the outcome variable to an integer
df['Outcome'] = df['class'].map(lambda x: int(str(x)[2]))
df = df.drop('class', axis=1)

n = 10  # Define the number of top categories to keep
df_poker = transform_dataset(df, n)

X_poker = df_poker.drop('Outcome', axis=1)
poker_names = [
    "Rank_1",
    "Rank_2",
    "Rank_3",
    "Rank_4",
    "Rank_5",
    "Card1_Suit_Diamonds",
    "Card1_Suit_Hearts",
    "Card1_Suit_Spades",
    "Card2_Suit_Diamonds",
    "Card2_Suit_Hearts",
    "Card2_Suit_Spades",
    "Card3_Suit_Diamonds",
    "Card3_Suit_Hearts",
    "Card3_Suit_Spades",
    "Card4_Suit_Diamonds",
    "Card4_Suit_Hearts",
    "Card4_Suit_Spades",
    "Card5_Suit_Diamonds",
    "Card5_Suit_Hearts",
    "Card5_Suit_Spades"
]

X_poker.columns = poker_names
y_poker = df_poker.Outcome
# Randomly reshuffle columns
np.random.seed(42)
X_poker.columns = np.random.permutation(X_poker.columns)


# Create y-poker to a binary where 0 == 0, else is 1
y_poker = y_poker == 1

####################### LOAD weather DATASET ########################
dir_path = '../data/imported_datasets/'
file_path_x = os.path.join(dir_path, 'weather', 'NEweather_data.csv')
file_path_y = os.path.join(dir_path, 'weather', 'NEweather_class.csv')

X_weather = pd.read_csv(file_path_x, header=None)
y_weather = pd.read_csv(file_path_y, header=None)[0]
X_weather.columns = [f'Feature_{i}' for i in range(X_weather.shape[1])]

####################### LOAD Elec DATASET ########################
dir_path = '../data/imported_datasets/'
file_path_x = os.path.join(dir_path, 'Elec2', 'elec2_data.dat')
file_path_y = os.path.join(dir_path, 'Elec2', 'elec2_label.dat')

X_elec = pd.read_csv(file_path_x, header=None, delimiter=' ')
X_elec.columns = [f'Feature_{i}' for i in range(X_elec.shape[1])]
y_elec = pd.read_csv(file_path_y, header=None, delimiter=' ')[0] # Take the first col

# Read cov type dataset at covType, filename covType.arff

dir_path = '../data/imported_datasets/'
file_path = os.path.join(dir_path, 'covType', 'covType.arff')

# Load dataset
data, meta = arff.loadarff(file_path)

df = pd.DataFrame(data)

df['class'].value_counts(normalize=True).index[0]

# Get y outcome and evaluate whether item is the first index of the value counts
outcome_1 = df['class'].value_counts(normalize=True).index[0]
df['Outcome'] = df['class'] == outcome_1

# Get x,y splits by dropping outcome and class
X_cov = df.drop(['Outcome', 'class'], axis=1)
y_cov = df.Outcome

# Sample only continuous variables
X_cov = X_cov.select_dtypes(include=[np.number])

# Dictionary to store datasets
datasets = {
    'airlines': {
        'X': X_airlines,
        'y': y_airlines,
        'description': 'Airline delay prediction',
        'task': 'Binary classification to predict flight delays'
    },
    'poker': {
        'X': X_poker,
        'y': y_poker,
        'description': 'Poker hand prediction',
        'task': 'Multi-class classification to predict poker hand rankings'
    },
    'weather': {
        'X': X_weather,
        'y': y_weather,
        'description': 'Weather prediction in the Northeast US',
        'task': 'Binary classification to predict a specific weather condition'
    },
    'elec': {
        'X': X_elec,
        'y': y_elec,
        'description': 'Electricity pricing prediction',
        'task': 'Binary classification to predict electricity price movement (up/down)'
    },
    'covType': {
        'X': X_cov,
        'y': y_cov,
        'description': 'Forest cover type prediction',
        'task': 'Binary classification to predict a specific forest cover type'
    }
}

def load_five_datasets():
    return datasets
