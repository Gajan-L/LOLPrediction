from keras.models import load_model
from sklearn import preprocessing
import pandas as pd


# load processed data
data = pd.read_csv('D:/Project/ASD Project2/DataFrame.csv', index_col=0)
print(data)
X = data[data.columns.difference(['T1 win'])]
le = preprocessing.LabelEncoder()

# test the game we want to predict
s = pd.Series(['Varus', 'Lulu', 'Gragas', 'Ryze', 'Yasuo',
               'Xayah', 'Janna', 'Jarvan IV', 'Malzahar', 'Gnar',
               'NA1', 8], index=data[data.columns.difference(['T1 win'])].columns, name='s')
# print(s)
X = X.append(s)
# print(X)
le_t = X.apply(le.fit)
X_1 = X.apply(le.fit_transform)
# print(X_1)
enc = preprocessing.OneHotEncoder()
enc_t = enc.fit(X_1)
X_2 = enc_t.transform(X_1)
# print(X_2[-1])

# loading model
model = load_model('D:/Project/ASD Project2/Neural Network Model.h5')
# predicting win rate
win_rate = model.predict_proba(X_2[-1])[0][1]
print(win_rate)

