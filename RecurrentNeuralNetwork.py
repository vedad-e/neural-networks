import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

num_classes = 2

read_file = pd.read_excel("Podaci_primjer.xlsx")
read_file.to_csv(r'Podaci_primjer.csv')

dataset = pd.read_csv('Podaci_primjer.csv', names=['rb','i1','i2','i3','i4','u1','u2','u3','u4','u5',], header=None)

dataset[['i4']] = dataset[['i2']].fillna(value=dataset[['i2']].mean())
dataset[['u1']] = dataset[['u1']].fillna(value=dataset[['u1']].mean())
dataset[['u2']] = dataset[['u2']].fillna(value=dataset[['u2']].mean())
dataset[['u3']] = dataset[['u3']].fillna(value=dataset[['u3']].mean())

dataset_input = read_file.iloc[400:601, 5:8].values
dataset_output = read_file.iloc[400:601, 4].values

# U dataset_input su ulazi u1-u3 a u dataset_output je izlaz i4
print (dataset)

#Skaliranje podataka, kako bi im vrijednosti bile izmedju -1 i 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(dataset_input)
y = scaler.fit_transform(dataset_output.reshape(-1,1))

#Sada se splita dataset u trening set i test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ulazni podaci se moraju reshapeati
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print(X_train)

combinations = [
    {'units': 50, 'activation': 'relu', 'optimizer': 'adam',},
    {'units': 100, 'activation': 'relu', 'optimizer': 'adam',},
    {'units': 200, 'activation': 'relu', 'optimizer': 'adam',},
    {'units': 300, 'activation': 'relu', 'optimizer': 'adam',},
    {'units': 400, 'activation': 'relu', 'optimizer': 'adam',},
    {'units': 500, 'activation': 'relu', 'optimizer': 'adam',},
    {'units': 800, 'activation': 'relu', 'optimizer': 'adam',},
]

# Kreiranje liste koja ce se naknadno popuniti u petlji
result = [(r['units'], r['activation'], r['optimizer'], None) for r in combinations]

result = [(r['units'], r['activation'], r['optimizer'], None) for r in combinations]

# Stvarna test data
plt.plot(y_test, color = 'green')

#Definiranje RNN arhitekture
for i, r in enumerate(combinations):
    rnn = Sequential()
    rnn.add(SimpleRNN(units=r['units'], activation=r['activation'], input_shape=(1, 3)))
    rnn.add(Dense(1))
    rnn.compile(optimizer=r['optimizer'], loss='mean_squared_error')

    # Treniranje i obucavanje na osnovu dataseta
    rnn.fit(X_train, y_train, epochs=150, verbose=0)
    y_pred = rnn.predict(X_test)
    score = rnn.evaluate(X_test, y_test)

    # Sacuvamo dobijene paremetre u kreiranu result listu
    result[i] = (r['units'], r['activation'], r['optimizer'], score)

    # Vrsimo plot prediktovanog testa, sa razlicitim brojem unita
    plt.plot(y_pred, label=f"Combination {i}")
    
# Vrsimo plot svih podataka
plt.xlabel('Samples')
plt.ylabel('Value')
plt.legend(["Original"] + [f"Prediction {i}" for i, _ in enumerate(result)])
plt.show()
