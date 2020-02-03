import pandas as pd
import squirrel_data_access
import resampling
import grid_search
import charts_and_things
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
import grid_search
import charts_and_things
#import pydotplus


df = squirrel_data_access.create_df()
X_train, X_test,y_train,y_test = squirrel_data_access.get_train_test_split(df, 'friendly')
#X_train, X_test,y_train,y_test = squirrel_data_access.get_train_test_split(df, 'approaches')

def plot_feature_importances(model, X_train):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.show()

def dec_tree(X_train, X_test,y_train,y_test):
    ctree=DecisionTreeClassifier(max_depth = 10)
    print(ctree.fit(X_train, y_train))

    clasPred = ctree.predict(X_test)
    print(clasPred)
    print(accuracy_score(y_test, clasPred))

    cm = confusion_matrix(y_test,clasPred)

    training_preds = ctree.predict(X_train)
    test_preds = ctree.predict(X_test)
    training_accuracy = accuracy_score(y_train, training_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    training_classified = classification_report(y_train, training_preds)
    test_classified = classification_report(y_test, test_preds)

    print(training_classified)
    print(test_classified)
    charts_and_things.plot_confusion_matrix(cm, ['Aloof', 'Friendly'],title='Decision Tree' )

    # print(training_classified)
    # print(test_classified)
    # print(cm)
    #plot_feature_importances(ctree, X_train)

#dec_tree(X_train, X_test,y_train,y_test)

def bagged_trees(X_train, X_test,y_train,y_test):
    bagged_tree =  BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=10),
                                 n_estimators=30)
    bagged_tree.fit(X_train, y_train)

    #print(bagged_tree.get_params().keys())

    clasPred = bagged_tree.predict(X_test)
    #print(clasPred)
    #print(accuracy_score(y_test, clasPred))

    cm = confusion_matrix(y_test,clasPred)

    charts_and_things.plot_confusion_matrix(cm, ['Aloof', 'Friendly'], title='Bagged Tree')

    param_grid = {
        'base_estimator__criterion':['gini'],
        'base_estimator__max_depth': [2,5],
        'base_estimator__max_features': [2,10],
        'n_estimators': [30,50]
    }

    grid_search.grid_search(bagged_tree, param_grid, X_train, X_test,y_train,y_test)

    # clasPred = bagged_tree.predict(X_test)
    #
    # print(bagged_tree.score(X_train, y_train))
    # print(bagged_tree.score(X_test, y_test))
    # cm = confusion_matrix(y_test,clasPred)
    # print(cm)

#bagged_trees(X_train, X_test,y_train,y_test)

def random_forest(X_train, X_test, y_train, y_test, max_depth = 5, n_estimators = 100):
    forest = RandomForestClassifier(n_estimators=n_estimators, max_depth= max_depth, max_features = 2)
    forest.fit(X_train, y_train)

    clasPred = forest.predict(X_test)
    #print(clasPred)
    #print(accuracy_score(y_test, clasPred))

    cm = confusion_matrix(y_test,clasPred)

    plot_feature_importances(forest, X_train)

    # charts_and_things.plot_confusion_matrix(cm, ['Aloof', 'Friendly'], title='Random Forest')

    # param_grid = {
    #     'criterion':['gini'],
    #     'max_depth': [2,12],
    #     'max_features':[2,10],
    #     'n_estimators': [90,150]
    # }
    #
    # grid_search.grid_search(forest, param_grid, X_train, X_test,y_train,y_test)

random_forest(X_train, X_test, y_train, y_test, max_depth = 12, n_estimators = 90)

# def grid_search(model, param_grid, X_train, X_test, y_train, y_test, scoring = 'accuracy'):
#     #naive to the input model but is dependent on global X, Y variables being declared in the script
#     grid_model = GridSearchCV(model, param_grid, scoring=scoring, cv=None, n_jobs=1)
#     #initialize grid model
#     grid_model.fit(X_train, y_train)
#     #fit it
#
#     best_parameters = grid_model.best_params_
#     print('Grid Search found the following optimal parameters: ')
#     for param_name in sorted(best_parameters.keys()):
#         print('%s: %r' % (param_name, best_parameters[param_name]))
#     #report optimum parameters
#
#     training_preds = grid_model.predict(X_train)
#     test_preds = grid_model.predict(X_test)
#     #create predictions
#
#     cm = confusion_matrix(y_test, test_preds)
#     print(cm)
#     #show confusion matrix
#
#     training_classified = classification_report(y_train, training_preds)
#     test_classified = classification_report(y_test, test_preds)
#     #assess them
#
#     print(training_classified)
#     print(test_classified)
#     #print results
#     return grid_model


#random_forest(X_train, X_test, y_train, y_test)

#bagged_trees(X_train, X_test,y_train,y_test)
