from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dataProcess import data_Process
from models import NeuralNetwork

file_dir = 'D:/Project/ASD Project/input/'
df = data_Process(file_dir)

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

le = preprocessing.LabelEncoder()
y_2 = df_2['T1 win']
X_2 = df_2[df_2.columns.difference(['T1 win'])]
le_t = X_2.apply(le.fit)
X_t_3 = X_2.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc_t = enc.fit(X_t_3)
X_t_4 = enc_t.transform(X_t_3)


# split train & test
X_train, X_test, y_train, y_test = train_test_split(X_t_4, y_2, random_state=0)

# train using Neural Network
model = NeuralNetwork(X_train, X_test, y_train, y_test)

# Save model
model.save("Neural Network Model.h5")
print('Training model has been saved! ')