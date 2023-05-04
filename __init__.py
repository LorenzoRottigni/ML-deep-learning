# Multi level Perceptron neutral network classifier algorithm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

n = 100

# random entry matrix
X = np.random.random(
    # 100 records, 5 features
    size=(n, 5)
)

# random target categories for classifier algoritmh
y = np.random.choise(['yes', 'no'], size=n)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MLPClassifier(
    # number of neural layers of compution between input layer and output layer
    hidden_layer_sizes=[
        # 1000 neurons on first hidden layer
        1000,
        # 500 neurons on second hidden layer
        500
    ],
    # by providing only 1 layer of 1 neuron, the model will go in overfitting because it hans't enough neurons to learn. ([1])
    # enable console logging
    verbose=2
)

model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

# ends in underfitting because the neural network
# focussed on training data that was totally random
# and wasnt able to find relations between features
