import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 50)

train = pd.read_csv('inputs/train.csv')
test = pd.read_csv('inputs/test.csv')

# Label train and test data
train['Label'] = 'train'
test['Label'] = 'test'

# Concatenate train and test data
df = pd.concat([train, test])

# Split passenger ID into group number and number within the group
df[['GroupNumber', 'NumberWithinGroup']] = df['PassengerId'].str.split('_', expand=True)
df[['GroupNumber', 'NumberWithinGroup']] = df[['GroupNumber', 'NumberWithinGroup']].astype(int)

df['HomePlanetEuropa'] = np.where(df['HomePlanet'] == 'Europa', 1, 0)
df['HomePlanetEarth'] = np.where(df['HomePlanet'] == 'Earth', 1, 0)
df['HomePlanetMars'] = np.where(df['HomePlanet'] == 'Mars', 1, 0)

df['CryoSleepTrue'] = np.where(df['CryoSleep'], 1, 0)
df['CryoSleepMissing'] = np.where(df['CryoSleep'].isna(), 1, 0)

df[['CabinFirstLetter', 'CabinNumber', 'CabinLastLetter']] = df['Cabin'].str.split('/', expand=True)
df['CabinNumber'] = pd.to_numeric(df['CabinNumber'], errors='coerce')

df['CabinMissing'] = np.where(df['Cabin'].isna(), 1, 0)

df['CabinFirstLetterA'] = np.where(df['CabinFirstLetter'] == 'A', 1, 0)
df['CabinFirstLetterB'] = np.where(df['CabinFirstLetter'] == 'B', 1, 0)
df['CabinFirstLetterC'] = np.where(df['CabinFirstLetter'] == 'C', 1, 0)
df['CabinFirstLetterD'] = np.where(df['CabinFirstLetter'] == 'D', 1, 0)
df['CabinFirstLetterE'] = np.where(df['CabinFirstLetter'] == 'E', 1, 0)
df['CabinFirstLetterF'] = np.where(df['CabinFirstLetter'] == 'F', 1, 0)
df['CabinFirstLetterG'] = np.where(df['CabinFirstLetter'] == 'G', 1, 0)

# TODO: Use cabin number

df['CabinLastLetterP'] = np.where(df['CabinLastLetter'] == 'P', 1, 0)

df['DestinationTrappist'] = np.where(df['Destination'] == 'TRAPPIST-1e', 1, 0)
df['DestinationPSO'] = np.where(df['Destination'] == 'PSO J318.5-22', 1, 0)
df['DestinationCancri'] = np.where(df['Destination'] == '55 Cancri e', 1, 0)

# TODO: Bins for age

df['Toddler'] = np.where(df['Age'] < 6, 1, 0)
df['Child'] = np.where((df['Age'] >= 6) & (df['Age'] < 18), 1, 0)

df['VIPTrue'] = np.where(df['VIP'], 1, 0)
df['VIPMissing'] = np.where(df['VIP'].isna(), 1, 0)

df['Spending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
df['SpentMoney'] = np.where(df['Spending'] > 0, 1, 0)

# TODO: Bins for spending

df['NameLength'] = df['Name'].str.len()
df['NameMissing'] = np.where(df['Name'].isna(), 1, 0)
df[['FirstName', 'LastName']] = df['Name'].str.split(' ', expand=True)
# TODO: Use first and last names

# Drop unused columns
columns_to_drop = [
    'PassengerId',
    'HomePlanet',
    'CryoSleep',
    'Cabin',
    'Destination',
    'VIP',
    'Name',
    'CabinFirstLetter',
    'CabinLastLetter',
    'FirstName',
    'LastName',
]
df.drop(columns=columns_to_drop, inplace=True)

train = df[df['Label'] == 'train']
test = df[df['Label'] == 'test']

y = train['Transported']
X = train.drop(columns=['Transported', 'Label'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=188, stratify=y)
X_test = test.drop(columns=['Transported', 'Label'])

# Add group size feature (based on train only to avoid data leakage)
group_counts = X_train['GroupNumber'].value_counts()
X_train['GroupSize'] = X_train['GroupNumber'].map(group_counts)
X_val['GroupSize'] = X_val['GroupNumber'].map(group_counts)
X_test['GroupSize'] = X_test['GroupNumber'].map(group_counts)

# Fill missing values with mean
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_val = X_val.fillna(train_means)
X_test = X_test.fillna(train_means)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train)
X_val = pd.DataFrame(X_val)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_val = pd.DataFrame(y_val)

print(y_train.mean().round(4))

for column in train.columns:
    if train[column].max() == 1:  # binary columns only
        print(column, train['Transported'][train[column] == 1].mean().round(4), sum(train[column]))

X_train.to_csv('processed/X_train.csv', index=False)
X_val.to_csv('processed/X_val.csv', index=False)
X_test.to_csv('processed/X_test.csv', index=False)
y_train.to_csv('processed/y_train.csv', index=False)
y_val.to_csv('processed/y_val.csv', index=False)
