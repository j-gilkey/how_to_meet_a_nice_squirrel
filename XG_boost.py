import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import squirrel_data_access
import resampling
import charts_and_things
import grid_search
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = squirrel_data_access.create_df()
X_train, X_test,y_train,y_test = squirrel_data_access.get_train_test_split(df, 'friendly')

def XGBoost(X_train, X_test, y_train, y_test):
    clf = XGBClassifier()
    clf.fit(X_train, y_train)

    clasPred = clf.predict(X_test)
    #print(clasPred)
    #print(accuracy_score(y_test, clasPred))

    cm = confusion_matrix(y_test,clasPred)

    charts_and_things.plot_confusion_matrix(cm, ['Aloof', 'Friendly'], title='XGBoost')

    param_grid = {
        'learning_rate': [0.1, 0.2],
        'max_depth': [2,5],
        'min_child_weight': [1, 3],
        'subsample': [0.5, 0.9],
        'n_estimators': [100,200],
    }

    #grid_search.grid_search(clf, param_grid, X_train, X_test, y_train, y_test)

XGBoost(X_train, X_test, y_train, y_test)

# grid_clf = GridSearchCV(clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
# grid_clf.fit(X_train, y_train)
#
# best_parameters = grid_clf.best_params_
#
# print('Grid Search found the following optimal parameters: ')
# for param_name in sorted(best_parameters.keys()):
#     print('%s: %r' % (param_name, best_parameters[param_name]))
#
# training_preds = grid_clf.predict(X_train)
# test_preds = grid_clf.predict(X_test)
# training_accuracy = accuracy_score(y_train, training_preds)
# test_accuracy = accuracy_score(y_test, test_preds)
#
# training_classified = classification_report(y_train, training_preds)
# test_classified = classification_report(y_test, test_preds)
#
# print(training_classified)
# print(test_classified)
