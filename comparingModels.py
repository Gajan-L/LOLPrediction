from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dataProcess import data_Process
from models import AdaBoost, Logistic_Regression, NaiveBayes, XGBoost, RandomForest, SVM, NeuralNetwork


def comparing_models(X_train, X_test, y_train, y_test):
    AdaBoost(X_train, X_test, y_train, y_test)
    Logistic_Regression(X_train, X_test, y_train, y_test)
    NaiveBayes(X_train, X_test, y_train, y_test)
    XGBoost(X_train, X_test, y_train, y_test)
    RandomForest(X_train, X_test, y_train, y_test)
    SVM(X_train, X_test, y_train, y_test)
    NeuralNetwork(X_train, X_test, y_train, y_test)


file_dir = './input/'
df = data_Process(file_dir)

df_1 = df[['matchid', 'player', 'name', 'team_role', 'win']]
df_1 = df_1.pivot(index='matchid', columns='team_role', values='name')
df_1 = df_1.reset_index()
df_1 = df_1.merge(df[df['player'] == 1][['matchid', 'win']], left_on='matchid', right_on='matchid', how='left')
df_1 = df_1[df_1.columns.difference(['matchid'])]
df_1 = df_1.rename(columns={'win': 'T1 win'})
df_1 = df_1.dropna()

y_1 = df_1['T1 win']
X_1 = df_1[df_1.columns.difference(['T1 win'])]
le = preprocessing.LabelEncoder()

# dataset converted from label to integer category
# label string to numeric
le_t = X_1.apply(le.fit)
X_t_1 = X_1.apply(le.fit_transform)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_t_1, y_1, random_state=0)


# dataset converted from label to binary category
enc = preprocessing.OneHotEncoder()
enc_t = enc.fit(X_t_1)
X_t_2 = enc_t.transform(X_t_1)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_t_2, y_1, random_state=0)


# dataset converted from label to binary category (with platform id, season id)
df_2 = df[['matchid', 'player', 'name', 'team_role', 'win']]

df_2 = df_2.pivot(index='matchid', columns='team_role', values='name')
df_2 = df_2.reset_index()
df_2 = df_2.merge(df[df['player'] == 1][['matchid', 'win', 'platformid', 'seasonid', 'version']],
                  left_on='matchid', right_on='matchid', how='left')
df_2 = df_2[df_2.columns.difference(['matchid', 'version'])]
df_2 = df_2.rename(columns={'win': 'T1 win'})
df['name'].unique()
df_2 = df_2.dropna()

y_2 = df_2['T1 win']
X_2 = df_2[df_2.columns.difference(['T1 win'])]
le_t = X_2.apply(le.fit)
X_t_3 = X_2.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc_t = enc.fit(X_t_3)
X_t_4 = enc_t.transform(X_t_3)


# split train & test, exclude last row (our curious comp)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_t_4, y_2, random_state=0)

print('\nAccuracy on dataset converted from label to integer category:')
comparing_models(X_train_1, X_test_1, y_train_1, y_test_1)
print('\nAccuracy on dataset converted from label to binary category:')
comparing_models(X_train_2, X_test_2, y_train_2, y_test_2)
print('\nAccuracy on dataset converted from label to binary category (with platform id, season id):')
comparing_models(X_train_4, X_test_4, y_train_4, y_test_4)