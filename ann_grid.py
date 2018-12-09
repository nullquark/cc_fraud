from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import parallel_backend
import numpy as np
import data

# explicitly set random seed to help parallelization
np.random.seed(0)

# use naive oversampling for grid search
x_train_rs, x_test_rs, y_train_rs, y_test_rs = data.parse_data_random_oversample('./creditcard.csv')

params = {
    'hidden_layer_sizes' : [(100, 50, 25), (75, 35, 15), (50, 25, 12)],
    'activation' : ['logistic', 'tanh'],
    'solver' : ['sgd', 'adam'],
    'alpha' : 10. ** -np.arange(3, 6),
    'learning_rate' : ['invscaling', 'adaptive'],
    'max_iter' : [1000, 1500, 2000]
}

# perform grid search with 5 fold cross validation, use all available cores
mlp_model = GridSearchCV(MLPClassifier(), params, n_jobs=-1, cv=5, verbose=2)
with parallel_backend('threading'):
    mlp_model.fit(x_train_rs, y_train_rs.ravel())

# output the best model params and score
print('best score: {0:.6f}'.format(mlp_model.best_score_))
print('best params: ')
print(mlp_model.best_params_)