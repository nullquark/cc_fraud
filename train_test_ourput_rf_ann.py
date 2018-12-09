from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals.joblib import dump
import numpy as np
import data
import report

# setup
np.random.seed(0)

# get data, test data is under and over sampled with SMOTEENN
x_train_rs, x_test, y_train_rs, y_test = data.parse_data_SMOTEENN('./creditcard.csv')

# train the RF model and output the results
rf_model = RandomForestClassifier(criterion='gini', n_estimators=100)
rf_model.fit(x_train_rs, y_train_rs.ravel())

pred_train = rf_model.predict(x_train_rs)
pred_test = rf_model.predict(x_test)

report.classification_report(y_train_rs, y_test, pred_train, pred_test)

#dump(rf_model, 'rf.joblib')

# train the ANN and output the results
mlp_model = MLPClassifier(activation='logistic', alpha=0.001, hidden_layer_sizes=(50, 25, 12), learning_rate='adaptive', max_iter=1500, solver='adam')
mlp_model.fit(x_train_rs, y_train_rs.ravel())

pred_train = mlp_model.predict(x_train_rs)
pred_test = mlp_model.predict(x_test)

report.classification_report(y_train_rs, y_test, pred_train, pred_test)

#dump(mlp_model, 'ann.joblib')