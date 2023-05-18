import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning
import warnings

num_classes = 2

read_file = pd.read_excel(r'C:\Users\vedad\Desktop\Podaci_primjer.xlsx')
read_file.to_csv(r'Podaci_primjer.csv')

dataset = pd.read_csv('Podaci_primjer.csv', names=['rb','i1','i2','i3','i4','u1','u2','u3','u4','u5',], header=None)

dataset[['i2']] = dataset[['i2']].fillna(value=dataset[['i2']].mean())
dataset[['u1']] = dataset[['u1']].fillna(value=dataset[['u1']].mean())
dataset[['u2']] = dataset[['u2']].fillna(value=dataset[['u2']].mean())
dataset[['u3']] = dataset[['u3']].fillna(value=dataset[['u3']].mean())
dataset[['u4']] = dataset[['u4']].fillna(value=dataset[['u4']].mean())
dataset[['u5']] = dataset[['u5']].fillna(value=dataset[['u5']].mean())

dataset['target_class'] = dataset['i2'].apply(lambda x: 0 if x < 314 else 1)

# Izracunali smo srednju vrijednost naseg zadanog seta koja je 314, te smo dodijelili
# da je lambda 0 za vrijednosti ispod 314, i 1 za vrijednosti vece od 314
X = dataset.iloc [500:1008, 5:10].values
y = dataset.iloc [500:1008, -1].values

# U X su ulazi u1-u5 a u Y je izlaz i2
print (dataset)

#Sada se splita dataset u trening set i test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Skaliranje podataka, kako bi im vrijednosti bile izmedju -1 i 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)

#Pravljenje MLPa
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='sgd', learning_rate='constant', learning_rate_init=0.025, max_iter=1000)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
y_pred_class = (y_pred > 0.5)

#print(y_pred_class, y_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_class)
#print(cm)

score = mlp.score(X_test, y_test)
print(f"Test accuracy: {score}")


params = [
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "learning_rate_init": 0.01,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "learning_rate_init": 0.087,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "learning_rate_init": 3.277,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "learning_rate_init": 0.01,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "learning_rate_init": 0.087,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "learning_rate_init": 3.277,
    },
    {"solver": "adam", "learning_rate_init": 0.01},
    {"solver": "adam", "learning_rate_init": 0.087},
    {"solver": "adam", "learning_rate_init": 3.277},
]

labels = [
    "constant learning_rate 0.01",
    "constant learning_rate 0.087",
    "constant learning_rate 3.277",
    "inv-scaling learning_rate 0.01",
    "inv-scaling learning_rate 0.087",
    "inv-scaling learning_rate 3.277",
    "adaptive learning_rate 0.01",
    "adaptive learning_rate 0.087",
    "adaptive learning_rate 3.277",
]

plot_args = [
    {"c": "red", "linestyle": "-"},
    {"c": "green", "linestyle": "-"},
    {"c": "blue", "linestyle": "-"},
    {"c": "red", "linestyle": "--"},
    {"c": "green", "linestyle": "--"},
    {"c": "blue", "linestyle": "--"},
    {"c": "purple", "linestyle": "-."},
    {"c": "black", "linestyle": "-."},
    {"c": "orange", "linestyle": "-."},
]


def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)

    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 1000
    else:
        max_iter = 2000

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(hidden_layer_sizes=(11, 4), activation='relu')

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
            mlp.fit(X_train, y_train)

        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X_train, y_train))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
data_sets = [(X_train, y_train)]

for ax, data, name in zip(
    axes.ravel(), data_sets, ["MLP"]
):
    plot_on_dataset(*data, ax=ax, name=name)

fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
plt.show()
