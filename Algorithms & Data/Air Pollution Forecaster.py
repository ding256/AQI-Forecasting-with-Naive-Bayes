import pandas as pd

import numpy as np

from sklearn.naive_bayes import GaussianNB

data0 = pd.read_csv('WeatherData_Averages.csv', sep=',')

data = pd.DataFrame(data0)

for i in range(200,360):
    features_train = data.iloc[:i, 6:15]
    labels_train = data.iloc[:i, 18:]
    clf = GaussianNB()
    clf.fit(features_train, labels_train.values.ravel())
    #输出单个预测结果
    features_test = data.iloc[i+1:, 6:15]
    labels_test = data.iloc[i+1:, 18:]
    labels_test=labels_test.values.ravel()
    pred = clf.predict(features_test)
    cnt=0
    for j in range(len(pred)):
        if abs(pred[j] - labels_test[j]) < 50:
            cnt += 1
        #print(pred[i], ' ', labels_test[i])
    print(i,"Accuracy: ", (cnt * 100.0 / len(labels_test)), "%")