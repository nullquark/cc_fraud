from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import parallel_backend
import numpy as np
import data

# explicitly set random seed to help parallelization
np.random.seed(0)

# use naive oversampling for grid search
x_train_rs, x_test_rs, y_train_rs, y_test_rs = data.parse_data_random_oversample('./creditcard.csv')

params = {
    'n_estimators' : np.arange(100, 250, 50),
    'criterion' : ['gini', 'entropy']
}

# perform grid search with 5 fold cross validation, use all available cores
rf_model = GridSearchCV(RandomForestClassifier(n_jobs=-1, class_weight='balanced'), params, n_jobs=-1, cv=5, verbose=2)
with parallel_backend('threading'):
    rf_model.fit(x_train_rs, y_train_rs.ravel())

# output the best model params and score
print('best score: {0:.6f}'.format(rf_model.best_score_))
print('best params: ')
print(rf_model.best_params_)