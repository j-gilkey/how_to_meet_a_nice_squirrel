import pandas as pd
import resampling
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

def bool_to_int(df, column_list):

    for c in column_list:
        df[c] = df[c].astype(int)
    return df

def make_dummies(df, column, set_prefix = 'dummy'):
    dummy = pd.get_dummies(df[column],prefix = set_prefix, drop_first=True)
    df = pd.concat([df, dummy], axis = 1)
    return df


def create_df():
    df = pd.read_csv('2018_Central_Park_Squirrel_Census_Squirrel_Data.csv' )
    columns = ['X', 'Y', 'unique_squirrel_id', 'hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location', 'running', 'chasing', 'climbing', 'eating',
           'foraging', 'kuks', 'quaas', 'moans', 'tail_flags',
           'tail_twitches', 'approaches', 'indifferent', 'runs_from' ]

    df = df.loc[:, columns]
    df = df.set_index('unique_squirrel_id')
    #df = df.drop(columns=['hectare', 'hectare_squirrel_number', 'combination_of_primary_and_highlight_color'])


    df = df.dropna()
    df = df[df['age'] != '?']
    #df = make_dummies(df, 'combination_of_primary_and_highlight_color', 'dummy_color')
    df = make_dummies(df, 'primary_fur_color', 'dummy_color')
    df = make_dummies(df, 'age', 'age')
    df = make_dummies(df, 'location', 'location')
    df = make_dummies(df, 'shift', 'shift')
    #df = make_dummies(df, 'hectare', 'hectare')
    #print(df[df['runs_from'] == 1].shape)
    #df = df[(df['approaches'] == 1) | (df['indifferent'] == 0)]
    column_list = ['running', 'chasing', 'climbing', 'eating',
                    'foraging', 'kuks', 'quaas', 'moans', 'tail_flags',
                    'tail_twitches', 'approaches', 'indifferent', 'runs_from']

    bool_to_int(df, column_list)

    #print(df.shape)
    df = df[(df['approaches'] == 1) | (df['indifferent'] == 1) | (df['runs_from'] == 1)]
    #print(df.shape)

    #print(df.sum())

    df['friendly'] =  df['runs_from'].replace({0:1, 1:0})

    #print(df.head)

    #df = df.drop(columns=['hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location'])
    #df = df.drop(columns=['X', 'Y','hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location'])

    #df = df.drop(columns=['shift', 'age', 'primary_fur_color', 'location', 'approaches', 'indifferent', 'runs_from'])
    df = df.drop(columns=['hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location', 'approaches', 'indifferent', 'runs_from'])
    #df = df.drop(columns=['hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location', 'indifferent', 'runs_from'])


    return df

#create_df()

labels = ['approaches', 'indifferent', 'runs_from']
values = [170, 1380, 639]

def create_exploration_df():
    df = pd.read_csv('2018_Central_Park_Squirrel_Census_Squirrel_Data.csv' )
    columns = ['X', 'Y', 'unique_squirrel_id', 'hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location', 'running', 'chasing', 'climbing', 'eating',
           'foraging', 'kuks', 'quaas', 'moans', 'tail_flags',
           'tail_twitches', 'approaches', 'indifferent', 'runs_from' ]

    df = df.loc[:, columns]
    df = df.set_index('unique_squirrel_id')


    df = df.dropna()
    df = df[df['age'] != '?']

    column_list = ['running', 'chasing', 'climbing', 'eating',
                    'foraging', 'kuks', 'quaas', 'moans', 'tail_flags',
                    'tail_twitches', 'approaches', 'indifferent', 'runs_from']

    bool_to_int(df, column_list)

    df['friendly'] =  df['runs_from'].replace({0:1, 1:0})

    #print(df.head)

    #df = df.drop(columns=['hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location'])
    #df = df.drop(columns=['X', 'Y','hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location'])
    #df = df.drop(columns=['hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location', 'approaches', 'indifferent', 'runs_from'])
    #df = df.drop(columns=['hectare', 'shift', 'hectare_squirrel_number', 'age', 'primary_fur_color', 'combination_of_primary_and_highlight_color', 'location', 'indifferent', 'runs_from'])


    return df


def get_train_test_split(df, y_column_name):
    X = df.drop(columns=[y_column_name])
    y = df[y_column_name]
    print(y.shape)
    X, y = resampling.remove_tomek(X, y)
    #remove tomek pairs
    print(y.shape)
    X_train, X_test,y_train,y_test = train_test_split(X, y, test_size = 0.4, random_state = 123)
    x_y_dict ={'X_train' : X_train, 'X_test' : X_test, 'y_train': y_train, 'y_test' : y_test}
    #print(x_y_dict)
    return(X_train, X_test,y_train,y_test)

# df = create_df()
# df_no_flee = df[df['friendly'] == 1]
# print(df_no_flee.shape)
# df_flee = df[df['friendly'] == 0]
# print(df_flee.shape)
