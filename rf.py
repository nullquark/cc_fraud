from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import dump, load
import numpy as np
import data
import report

# setup
test_data_saved = False
np.random.seed(0)

# data.pasrse_data_SMOTEENN save results to *_SMOTEENN.joblib files, deserialize from disk to save time if they exist
if (test_data_saved):
    x_train_rs = load('x_resample_SMOTEENN.joblib')
    x_test = load('x_test_SMOTEENN.joblib')
    y_train_rs = load('y_resample_SMOTEENN.joblib')
    y_test = load('y_test_SMOTEENN.joblib')
else:
    x_train_rs, x_test, y_train_rs, y_test = data.parse_data_SMOTEENN('./creditcard.csv')

# train and predict, train using over and under sampled data with SMOTEENN
rf_model = RandomForestClassifier(criterion='gini', n_estimators=100)
rf_model.fit(x_train_rs, y_train_rs.ravel())

pred_train = rf_model.predict(x_train_rs)
pred_test = rf_model.predict(x_test)

# output results and plot ROCAUC
report.classification_report(y_train_rs, y_test, pred_train, pred_test)

# serialize the trained model for later use
dump(rf_model, 'rf.joblib')