import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.joblib import load

"""
I'm using SciKit learn example code to plot a confusion matrix in an asthetic fashion. See:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.PuRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout() 
    # end of SciKit learn example code

class_names = ['non-fraud', 'fraud']

# assumes these files are available
x_train_rs = load('x_resample_SMOTEENN.joblib')
x_test = load('x_test_SMOTEENN.joblib')
y_train_rs = load('y_resample_SMOTEENN.joblib')
y_test = load('y_test_SMOTEENN.joblib')

# assumes the model is available
rf_model = load('rf.joblib')
rf_model.fit(x_train_rs, y_train_rs.ravel())
rf_pred_test = rf_model.predict(x_test)
rf_cm = metrics.confusion_matrix(y_test, rf_pred_test)

plt.figure()
plot_confusion_matrix(rf_cm, class_names)

# assumes the model is available
mlp_model = load('ann.joblib')
mlp_model.fit(x_train_rs, y_train_rs.ravel())
mlp_pred_test = mlp_model.predict(x_test)
mlp_cm = metrics.confusion_matrix(y_test, mlp_pred_test)

plt.figure()
plot_confusion_matrix(mlp_cm, class_names)

plt.show()