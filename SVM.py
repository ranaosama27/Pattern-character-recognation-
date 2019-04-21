
import os
import cv2
import random
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#
DataDir = "E:\Pattern-character-recognition\Dataset"
Categories = ["Zero","One","Two","Three","Four","Five","six","Seven","Eight","Nine","A-Capital","B-Capital","C-Capital","D-Capital","E-Capital","F-Capital","G-Capital","H-Capital","I-Capital",
              "J-Capital","K-Capital","L-Capital","M-Capital","N-Capital","O-Capital","P-Capital","Q-Capital","R-Capital","S-Capital","T-Capital","U-Capital","V-Capital","W-Capital","X-Capital","Y-Capital","Z-Capital",
              "a-Small","b-Small","c-Small","d-Small","e-Small","f-Small","g-Small","h-Small","i-Small","j-Small","k-Small","l-Small","m-Small","n-Small","o-Small","p-Small","q-Small","r-Small","s-Small","t-Small",
              "u-Small","v-Small","w-Small","x-Small","y-Small","z-Small"]
training_data = []
image_size = 50
x = []
y = []


def create_trainig_data():
    for category in Categories:
        path = os.path.join(DataDir, category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (image_size, image_size)).flatten()
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

random.shuffle(training_data)

create_trainig_data()
for feature, label in training_data:
    x.append(feature)
    y.append(label)

# Create Support Vector Classification object
Modle = svm.SVC(gamma=0.001, C =100) # لازم تغير ف البرامترز ديه 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21, stratify=y)

Modle.fit(X_train, y_train)
Score = Modle.score(X_test,y_test);

print(Score)
