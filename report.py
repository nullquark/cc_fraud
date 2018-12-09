from sklearn import metrics
import matplotlib.pyplot as plt

"""
Output a classification report and ROCAUC plot

Args:
    y_train: np.array of training data ground truth
    y_test: np.array of testing data ground truth
    pred_train: np.array of predicted class labels during training
    pret_test: np.array of predicted class labels during testing
"""
def classification_report(y_train, y_test, pred_train, pred_test):
    # print several metrics to stdout
    print("Training Accuracy: {0:.6f}\n".format(metrics.accuracy_score(y_train, pred_train)))
    print("Testing Accuracy: {0:.6f}\n".format(metrics.accuracy_score(y_test, pred_test)))
    print("Confusion Matrix\n")
    print("{0}\n".format(metrics.confusion_matrix(y_test, pred_test)))
    print("Classification Report")
    print(metrics.classification_report(y_test, pred_test))

    # calculate ROC and ROCAUC curve
    fpr, tpr, _ = metrics.roc_curve(y_test, pred_test)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    # plot ROCAUC curve
    plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'.format(roc_auc), color='deeppink', linewidth=4)

    # plot dashed line from origin to (1,1)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    
    # set up axes and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc="lower right")

    plt.show()
