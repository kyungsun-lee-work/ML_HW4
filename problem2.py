import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

def generateGMM(num):
    numOfdataset = num
    components = [1,2,3,4]
    priors = [0.1, 0.2, 0.3, 0.4]

    label = random.choices(components, priors, k=numOfdataset)
    #print(label, Counter(label))

    # fig = plt.figure()

    mean = [[5, 0], [0, 4], [0, 0], [6, 4]]
    cov = [[[4, 0], [0, 2]],  # diagonal covariance
            [[1, 0], [0, 3]],
            [[2, 0], [0, 2]],
            [[3, 0], [0, 1]]]

    #generate a list of markers and another of colors 
    markers = ["o" , "v" , "^" , ","]
    colors = ['r','g','b','y','m', 'y', 'k']
    labels = ['component 1','component 2','component 3','component 4']

    a = np.array([])
    b = np.array([])

    for i in range(4):
        x, y = np.random.multivariate_normal(mean[i], cov[i], label.count(i+1)).T
        # plt.scatter(x, y, label=labels[i], marker=markers[i], color=colors[i])

        a = np.append(a, x)
        b = np.append(b, y)

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('GMM datasets with %i samples' %num)
    # plt.legend()
    # plt.show()

    return a, b

def selectGMM(data_num):
    a, b = generateGMM(data_num)

    # 10-fold cross-validation
    X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.1, random_state=0)

    Gmm_train = np.stack((X_train, y_train), axis=1)
    Gmm_test = np.stack((X_test, y_test), axis=1)

    gmm = [0, 0, 0, 0, 0, 0]
    result = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        gmm[i] = GaussianMixture(n_components=i+1, verbose=2, random_state=0).fit(Gmm_train)
        result[i] = gmm[i].score(Gmm_test, 2)
        print("Log-likelihood of GMM", i+1, ":", result[i])

    return result.index(max(result))

data_num = 10000
res = [0, 0, 0, 0, 0, 0]
component = [1,2,3,4,5,6]

for iter in range(100):
    ans = selectGMM(data_num)
    res[ans] = res[ans] + 1

plt.bar(component, res)

for i, v in enumerate(component):
    plt.text(v, res[i], res[i], fontsize = 9, color='blue', horizontalalignment='center', verticalalignment='bottom')

plt.xlabel('Components')
plt.ylabel('rate of get selected')
plt.title('Experiment 100 times in GMM with %i samples' %data_num)
plt.show()
