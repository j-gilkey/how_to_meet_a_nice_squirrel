import pandas as pd
import squirrel_data_access
import seaborn as sns
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# df_exp = squirrel_data_access.create_exploration_df()
# df_exp_1 = df_exp[df_exp['friendly'] == 1]
# df_exp_2 = df_exp[df_exp['friendly'] == 0]
# print(df_exp.shape)
#df = df[(df['running'] == 1) | (df['chasing'] == 1)| (df['climbing'] == 1)| (df['foraging'] == 1)| (df['eating'] == 1)]
#print(active_df.shape)

df = squirrel_data_access.create_df()
df_1 = df[df['friendly'] == 1]
df_2 = df[df['friendly'] == 0]
#df_2 = df_2[df_2['indifferent'] == 0]


#print(df_1.sum())
#print(pd.concat([df_2.sum(),df_1.sum()], axis=1, sort=False))

#print(df_approach.combination_of_primary_and_highlight_color.unique())

def behavior_pie():
    labels = ['approaches', 'indifferent', 'runs_from']
    values = [170, 1380, 639]

def grid_plot(df):

    fig = plt.figure()
    plt.style.use('seaborn')
    g = sns.PairGrid(df)
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    plt.show()

#grid_plot(df)

def count_plot(df, to_plot):
    #takes a dataframe and a column to create a hist plot on
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    ax = sns.countplot(x=to_plot, data=df, hue='friendly', hue_order=[1,0])
    ax.set_xlabel('Log Salary')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
    #hist_serial.set_xlabel('Trees per sq Mile')
    plt.show()

#count_plot(df_exp, 'primary_fur_color')
#count_plot(df_exp, 'combination_of_primary_and_highlight_color')


#count_plot(df, 'age')

def count_plot_compare(df_1, df_2, to_plot):
    #takes a dataframe and a column to create a hist plot on
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    ax1 = sns.countplot(x=to_plot, data=df_1)
    ax1.set_xlabel('Log Salary')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40)
    ax2 = sns.countplot(x=to_plot, data=df_2)
    ax2.set_xlabel('Log Salary')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40)
    #hist_serial.set_xlabel('Trees per sq Mile')
    plt.show()

#count_plot_compare(df, df_approach, 'primary_fur_color')

def hist_plot(df, to_plot):
    #takes a dataframe and a column to create a hist plot on
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    hist_serial = sns.distplot(df[to_plot].astype(float), kde = False)
    # print(stats.kurtosis(list(df[to_plot])))
    # print(stats.skew(list(df[to_plot])))
    #hist_serial.set_xlabel('Log Salary')
    #hist_serial.set_xlabel('Trees per sq Mile')
    plt.show()

def multi_hist_plot(df_1, df_2, to_plot):
    #takes a dataframe and a column to create a hist plot on
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    hist_1 = sns.distplot(df_1[to_plot].astype(float), kde = False)
    hist_2 = sns.distplot(df_2[to_plot].astype(float), kde = False)
    # print(stats.kurtosis(list(df[to_plot])))
    # print(stats.skew(list(df[to_plot])))
    #hist_serial.set_xlabel('Log Salary')
    #hist_serial.set_xlabel('Trees per sq Mile')
    plt.show()

#multi_hist_plot(df_1, df_2, 'X')

#hist_plot(df_2, 'Y')

def scatter_plot(X, Y, df):
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    ax = sns.scatterplot(x=df[X], y=df[Y], data=df)
    plt.show()

#scatter_plot('X', 'Y', df)

def joint_plot(X, Y, df):
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    g = sns.jointplot(X, Y, data=df,
                  kind="kde", space=0,)
    plt.show()

#joint_plot('X', 'Y', df_2)


def heat_corr(df, name):
    fig = plt.figure()
    #plt.clear()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    corr = df.corr()
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
    ax.set_title(name)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

#heat_corr(df, 'Heat')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    bottom, top = plt.ylim()
    plt.ylim(bottom + 0.5, top - 0.5)
    plt.show()
