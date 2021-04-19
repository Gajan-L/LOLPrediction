from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# AdaBoost
def AdaBoost(X_train, X_test, y_train, y_test):
    clf_ada = AdaBoostClassifier(n_estimators=1000).fit(X_train, y_train)
    acc_test_ada = clf_ada.score(X_test, y_test)
    print('Test accuracy of AdaBoost : {}'.format(acc_test_ada))
    return clf_ada


# Logistic Regression
def Logistic_Regression(X_train, X_test, y_train, y_test):
    clf_lr = LogisticRegression(random_state=0).fit(X_train, y_train)
    acc_test_lr = clf_lr.score(X_test, y_test)
    print('Test accuracy of Logistic Regression : {}'.format(acc_test_lr))
    return clf_lr


# Naive Bayes Classifier
def NaiveBayes(X_train, X_test, y_train, y_test):
    clf_bnb = BernoulliNB().fit(X_train, y_train)
    acc_test_bnb = clf_bnb.score(X_test, y_test)
    print('Test accuracy of Naive Bayes : {}'.format(acc_test_bnb))
    return clf_bnb


# XGBoost
def XGBoost(X_train, X_test, y_train, y_test):
    clf_xgb = xgboost.XGBClassifier(num_class=2, silent=1, eta=0.1, max_depth=10, gamma=0.01, subsample=0.75,
                                    objective='multi:softmax', random_state=0).fit(X_train, y_train)
    acc_test_xgb = clf_xgb.score(X_test, y_test)
    print('Test accuracy of XGBoost : {}'.format(acc_test_xgb))
    return clf_xgb


# Random Forest
def RandomForest(X_train, X_test, y_train, y_test):
    clf_rf = RandomForestClassifier(n_estimators=1000, n_jobs=2, random_state=0).fit(X_train, y_train)
    acc_test_rf = clf_rf.score(X_test, y_test)
    print('Test accuracy of Random Forest : {}'.format(acc_test_rf))
    return clf_rf


# Support Vector Machine
def SVM(X_train, X_test, y_train, y_test):
    clf_svm = svm.SVC().fit(X_train, y_train)
    acc_test_svm = clf_svm.score(X_test, y_test)
    print('Test accuracy of SVM : {}'.format(acc_test_svm))
    return clf_svm


# Neural Network
def NeuralNetwork(X_train, X_test, y_train, y_test):
    batch_size = 128
    nb_classes = 2
    nb_epoch = 308
    X_train = X_train.astype('float32')
    X_test = X_test.astype('int64')
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print(X_train.shape)
    model = Sequential()
    model.add(Dense(1500, input_shape=X_train.shape, activation='hard_sigmoid'))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=nb_epoch,
                        verbose=0, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy of Neural Network : {}'.format(score[1]))
    return model