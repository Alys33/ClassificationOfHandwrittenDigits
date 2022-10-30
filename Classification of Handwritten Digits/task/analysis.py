
# importing the modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np

(X_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
X = np.concatenate((X_train, x_test))[:6000]
y= np.concatenate((y_train, y_test))[:6000]

# reshaping of flatting  the features matrix
X = X.reshape(X.shape[0], X.shape[1] ** 2)


# print(f"Classes: {np.unique(y_train)}")
# print(f"Features' shape: {x_train.shape}")
# print(f"Target's shape: {y_train.shape}")
# print(f"min: {x_train.min()}, max: {x_train.max()}")

from sklearn.model_selection import train_test_split
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

df = pd.Series(y_train)
prop = df.value_counts(normalize=True)

# print(f'x_train shape: {X_train.shape}')
# print(f'x_test shape: {X_test.shape}')
# print(f'y_train shape: {y_train.shape}')
# print(f'y_test shape: {y_test.shape}')
# print("Proportion of samples per class in train set:")
# print(prop)

# instance des models

#prediction with the defaults parameters  for 4 different models 

def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    target_pred = model.predict(features_test)
    score = accuracy_score(target_test, target_pred)
    return round(score,3)

# knear = fit_predict_eval(KNeighborsClassifier(), X_train, X_test, y_train, y_test)
# decis = fit_predict_eval(DecisionTreeClassifier(random_state=40), X_train, X_test, y_train, y_test)
# logistic = fit_predict_eval(LogisticRegression(), X_train, X_test, y_train, y_test)
# rf = fit_predict_eval(RandomForestClassifier(random_state=40), X_train, X_test, y_train, y_test)
# algo = ["KNeighborsClassifier()", "DecisionTreeClassifier(random_state=40)", "LogisticRegression()", "RandomForestClassifier(random_state=40)"]
# algo2 = ["KNeighborsClassifier()", "DecisionTreeClassifier(random_state=40)", "LogisticRegression()", "RandomForestClassifier"]
# result = [knear, decis, logistic, rf]
# max_model = max(result)
# index = result.index(max_model)
# best_model = algo2[index]
#
# for model in range(len(result)):
#     print(f'Model: {algo[model]}\nAccuracy: {result[model]}\n')
# print(f"The answer to the question: {best_model} - {max_model}")


# Stage 4/5: Data preprocessing

from sklearn.preprocessing import Normalizer

normal = Normalizer()
tf_train = normal.fit(X_train)
X_train_norm = tf_train.transform(X_train)

tf_test = normal.fit(X_test)
X_test_norm = tf_test.transform(X_test)



# knear = fit_predict_eval(KNeighborsClassifier(), X_train_norm, X_test_norm, y_train, y_test)
# decis = fit_predict_eval(DecisionTreeClassifier(random_state=40), X_train_norm, X_test_norm, y_train, y_test)
# logistic = fit_predict_eval(LogisticRegression(), X_train_norm, X_test_norm, y_train, y_test)
# rf = fit_predict_eval(RandomForestClassifier(random_state=40), X_train_norm, X_test_norm, y_train, y_test)
# algo = ["KNeighborsClassifier()", "DecisionTreeClassifier(random_state=40)", "LogisticRegression()", "RandomForestClassifier(random_state=40)"]
# algo2 = ["KNeighborsClassifier()", "DecisionTreeClassifier(random_state=40)", "LogisticRegression()", "RandomForestClassifier"]
# result = [knear, decis, logistic, rf]
# max_model = max(result)
# index = result.index(max_model)
# best_model = algo2[index]
#
# for model in range(len(result)):
#     print(f'Model: {algo[model]}\nAccuracy: {result[model]}\n')
# # print(f"The answer to the question: {best_model} - {max_model}\n")
#
# print("The answer to the 1st question: yes\n")
# print("The answer to the 2nd question: KNeighborsClassifier-0.953, RandomForestClassifier-0.937")


# Stage 5/5: Hyperparameter Tuning

# data representation: X_train_norm, X_test_norm
# model : KNeighborsClassifier, RandomForestClassifier
from sklearn.model_selection  import GridSearchCV


random_f = RandomForestClassifier(random_state=40)
kn = KNeighborsClassifier()
param_kn = {
    'n_neighbors': [2, 3, 4, 5, 6, 7,8,9,10],
    'weights': ['uniform','distance'],
    'algorithm': ['auto','brute']
}

param_rf = {
    'n_estimators': [300, 400, 500,600, 700, 800],
    'max_features': ['sqrt','log2'],
    'class_weight': ['balanced','balanced_subsample']
}

# for random forest
# grid_rf = GridSearchCV(estimator=random_f, param_grid=param_rf, scoring="accuracy", n_jobs=-1)
#
# grid_rf.fit(X_train_norm, y_train)
#
# print(f"The best parameter for random forest is {grid_rf.best_estimator_}")
#
# #for KNeighborsClassifier
#
# grid_kn = GridSearchCV(estimator=kn, param_grid=param_kn, scoring="accuracy", n_jobs=-1)
# grid_kn.fit(X_train_norm, y_train)
#
# print(f"The best parameter for ra is {grid_kn.best_estimator_}")


# train with test sets
new_rf = RandomForestClassifier(class_weight='balanced_subsample', max_features='log2', n_estimators=700, random_state=40)
new_kn = KNeighborsClassifier(n_neighbors=4, weights='distance')
valu_rf = fit_predict_eval(new_rf, X_train_norm, X_test_norm, y_train, y_test)
valu_kn = fit_predict_eval(new_kn, X_train_norm, X_test_norm, y_train, y_test)

print("K-nearest neighbours algorithm\nbest estimator: KNeighborsClassifier(n_neighbors=4, weights='distance')")
print(f"accuracy: {valu_kn}")

print("Random forest algorithm\nbest estimator: RandomForestClassifier(class_weight='balanced_subsample', max_features='log2',n_estimators=700, random_state=40)")
print(f"accuracy: {valu_rf}")


