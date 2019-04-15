import pickle
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pickle_in = open("x.pickle","rb")
x = pickle.load(pickle_in)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
# print(x[1])
x =np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)

# Create Support Vector Classification object
Modle = svm.SVC(gamma=0.001, C =100)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21, stratify=y)

Modle.fit(X_train, y_train)
Score = Modle.score(X_test,y_test);

print(Score)