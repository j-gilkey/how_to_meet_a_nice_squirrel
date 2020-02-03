from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import squirrel_data_access
import resampling
import grid_search
import charts_and_things
import seaborn as sns

df = squirrel_data_access.create_df()
X_train, X_test,y_train,y_test = squirrel_data_access.get_train_test_split(df, 'friendly')

def plot_feature_importances(model, X_train):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.show()

def knn_model(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=3, leaf_size=20, weights='distance', algorithm='ball_tree')
    knn.fit(X_train, y_train)
    #y_pred_prob = knn.predict_proba(X_test)[:, 1]

    #generate_roc_curve(y_test, y_pred_prob)

    # clasPred = knn.predict(X_test)
    # #print(clasPred)
    # #print(accuracy_score(y_test, clasPred))
    #
    # cm = confusion_matrix(y_test,clasPred)
    #
    # charts_and_things.plot_confusion_matrix(cm, ['Aloof', 'Friendly'], title='KNN')

    param_grid = {
        'leaf_size': [10, 30],
        'n_neighbors': [1,3],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree','kd_tree'],
        'p': [2]
        }

    grid_search.grid_search(knn, param_grid, X_train, X_test, y_train, y_test)

#knn_model(X_train, X_test, y_train, y_test)

def generate_roc_curve(y_test, y_pred_prob):

    print('hello')

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC for KNN Squirrel Classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()

knn_model(X_train, X_test, y_train, y_test)
