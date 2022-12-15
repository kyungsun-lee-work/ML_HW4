import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from scipy.stats import multivariate_normal

def data_generate(num):
    numOfdataset = num
    myClass = [1,2,3,4]
    classPriors = [0.25, 0.25, 0.25, 0.25]
    data = []

    label = random.choices(myClass, classPriors, k=numOfdataset)
    #print(label, Counter(label))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    mean = [[0, 0, 0], [1, -1, 1], [-3, 1, 2], [2, 0, -2]]
    cov = [[[2, 0, 0], [0, 1, 0], [0, 0, 1]],  # diagonal covariance
            [[2, 2, 2], [1, 1, 1], [0, 0, 1]],
            [[2, 0, 2], [0, 1, 0], [0, 0, 2]],
            [[1, 0, 0], [0, 2, 0], [0, 0, 3]]]

    #generate a list of markers and another of colors 
    markers = ["o" , "v" , "^" , ","]
    colors = ['r','g','b','y','m', 'y', 'k']
    labels = ['class 1','class 2','class 3','class 4']

    for i in range(4):
        x, y, z = np.random.multivariate_normal(mean[i], cov[i], label.count(i+1)).T
        ax.scatter(x, y, z, label=labels[i], marker=markers[i], color=colors[i])
        data.append((x, y, z, i+1))

    #print(data)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Training datasets with %i samples' %num)
    plt.legend()
    plt.show()

    var = [0, 0, 0, 0]
    correct = 0
    wrong = 0

    for i in range(4):
        var[i] = multivariate_normal(mean[i], cov[i])

    for i in range(4):
        x, y, z, ans = data[i]
        for k in range(len(x)):
            max = float("-inf")
            for mm in range(4):
                if max < var[mm].pdf([x[k], y[k], z[k]]):
                    max = var[mm].pdf([x[k], y[k], z[k]])
                    predict = mm + 1

            if predict == ans:
                correct = correct + 1
            else:
                wrong = wrong + 1

    print("correct:", correct, ", wrong:", wrong, ", probability of error of theoretically optimal claasifier:", wrong/num*100.0)

    return data
