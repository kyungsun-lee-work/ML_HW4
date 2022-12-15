import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from dataGenerate import data_generate
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def elu(z,alpha):
	return z if z >= 0 else alpha*(np.exp(z) - 1)

def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def mlp(size):
    data = data_generate(size)

    x, y, z, label = data
    a = np.array([])
    b = np.array([])
    c = np.array([])
    d = np.array([])

    for labelCnt in range(4):
        x, y, z, label = data[labelCnt]
        label = np.full_like(x, label)
        print("now:", len(z), len(label))
    
        a = np.append(a, x)
        b = np.append(b, y)
        c = np.append(c, z)
        d = np.append(d, label)

    a = np.array(a).reshape(size, 1)
    b = np.array(b).reshape(size, 1)
    c = np.array(c).reshape(size, 1)
    X = np.concatenate((a, b, c), axis=1)

    
    # if iter < len(size) - 1:
    #     x_train[iter] = X
    #     y_train[iter] = d
    # else:
    #     x_test = X
    #     y_test = d

    # perform 10-fold cross-validation
    x_train, x_test, y_train, y_test = train_test_split(X, d, test_size=0.1, random_state=32)

    clf = [0 for _ in range(50)]
    best = 0
    best_layer = 0

    for layer in range(30):
        clf[layer] = MLPClassifier(hidden_layer_sizes=(layer+1,), random_state=5, verbose=False, learning_rate_init=0.01, activation='relu', max_iter=2000)
        clf[layer].fit(x_train, y_train)

        ypredict = clf[layer].predict(x_test)
        temp = accuracy_score(y_test, ypredict)
        print("Num of layer", layer, "-", temp)

        if (best <= temp):
            best = temp
            best_layer = layer

    clf = MLPClassifier(hidden_layer_sizes=(best_layer,), random_state=5, verbose=False, learning_rate_init=0.01, activation='relu', max_iter=2000)
    clf.fit(x_train, y_train)

    ypredict = clf.predict(x_test)

    print("best:", best_layer, "minimum classification error probability:", 1- best, "score:", 1 - accuracy_score(y_test, ypredict))

    cm = confusion_matrix(y_test, ypredict)
    #sns.heatmap(cm, center=True)
    print(cm)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=15)
    plt.ylabel('Actuals', fontsize=15)
    plt.title('Confusion Matrix %i samples' %size, fontsize=18)
    plt.show()

size = [100, 200, 500, 1000, 2000, 5000, 100000]
mlp(5000)
