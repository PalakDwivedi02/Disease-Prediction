import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

% matplotlibinline

#Reading the Data Set

DATA_PATH = "dataset/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)
disease_counts = data["prognosis"].value_counts()
temp_df=pd.DataFrame({"Disease": disease_counts.index, "Counts": disease_counts.values})
plt.figure(figsize=(18,8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
plt.xticks(rotation=90)
plt.show()
encoder= LabelEncoder()
data["prognosis"]=encoder.fit_transform(data["prognosis"])

#Splitting the data for training and testing the model

X = data.iloc[:,:-1]
y = data.iloc[:-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=24)
print(f"Train : {X_train.shape},{y_train.shape}")
print(f"Test : {X_test.shape},{y_test.shape}")

#Model Building

def cv_scoring(estimator,X,y):
    return accuracy_score(y,estimator.predict(X))

models={"SVC":SVC(), "GaussianNB":GaussianNB(), "RandomForest":RandomForestClassifier(random_state=18)}
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model,X,y,cv=10,n_jobs=-1,scoring=cv_scoring)

print("=="*30)
print(model_name)
print(f"scores : {scores}")
print(f"Mean Score: {np.mean(scores)}")

#Training and Testing SVM Classifier 

