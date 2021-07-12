from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from read_data import read_cancer_data

data = read_cancer_data()
y = data[:,0]
X = data[:,1:9]
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,y)
scores = fit.scores_
for i in range(0,len(scores)):
    print(i+1,"   :   ",fit.scores_[i])

enc = preprocessing.OneHotEncoder()
X = data[:,[1]]
enc.fit(X)
X1 = enc.transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split( X1, y, test_size=0.30, random_state=79)
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
clf = SVC(kernel=kernel[0])
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(clf.score(X_test, y_test))