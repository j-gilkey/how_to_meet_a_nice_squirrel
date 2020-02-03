import pandas as pd
import squirrel_data_access
import seaborn as sns
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks


def remove_tomek(X, y):
    #tl = TomekLinks()
    tl = TomekLinks(sampling_strategy='all')
    X_res, y_res = tl.fit_resample(X, y)
    return X_res, y_res

# X_res, y_res = remove_tomek(X, y)
#
# print(Counter(y_res))
