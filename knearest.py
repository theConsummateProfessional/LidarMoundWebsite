from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import asyncio
import time

async def read_files(path):
    y = []
    df = pd.read_csv(path)
    x_arr = df[["alphas", "theta", "phi"]].to_numpy()
    print(path)
    if ("Natural" in path):
        y.append(0)
    else:
        y.append(1)

    return x_arr, y

def testing(X_train, y_train, X_test, y_test, index, numNeighbors, train_acc, test_acc):
    print("Testing...", index)
    knn = KNeighborsClassifier(n_neighbors=numNeighbors)
    knn.fit(list(X_train), y_train)

    train_acc[index] = knn.score(list(X_train), y_train)
    test_acc[index] = knn.score(list(X_test), y_test)
    print(index, "done")

async def main(argv1):
    tasks = []
    start_time = time.time()
    for filename in os.listdir(argv1):
        f = os.path.join(argv1, filename)
        if (os.path.isfile(f) and ".csv" in f):
            name_of_mound = filename.replace(".csv", "")
            tasks.append(asyncio.ensure_future(read_files(f)))
    not_x = await asyncio.gather(*tasks)
    X = []
    y = []
    for val in not_x:
        X.append(val[0].flatten())
        y.append(val[1][0])
    print("--- %s seconds ---" % (time.time() - start_time))

    np_X = np.array(X, dtype=object)
    biggest_list_length = len(max(np_X, key=len))
    for i, x in enumerate(np_X):
        list_length = len(x)
        zeros_length = biggest_list_length - list_length
        np_X[i] = np.concatenate((x, np.zeros(zeros_length)), axis=None)

    X_train, X_test, y_train, y_test = tts(np_X, y, test_size=0.20, random_state=42)
    np.asarray(X_train)

    neighbors = np.arange(1, 9)

    print('\n____________________________________')
    print('         TESTING PHASE               ')
    print('____________________________________')
    train_acc = []
    test_acc = []
    start_time = time.time()
    for i, k in enumerate(neighbors):
        print("Testing...", i)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(list(X_train), y_train)

        train_acc.append(knn.score(list(X_train), y_train))
        test_acc.append(knn.score(list(X_test), y_test))
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.plot(neighbors, test_acc, label = 'Testing dataset Accuracy')
    plt.plot(neighbors, train_acc, label = 'Training dataset Accuracy')

    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))