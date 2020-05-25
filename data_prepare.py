
#################################################
import numpy as np
from sklearn import metrics
#################################################

#################################################
######性能指标
def get_metrics(true_labels, predicted_labels):
    print('Accuracy:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        2))
    print('Precision:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        2))
    print('Recall:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        2))
    print('F1-measure:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        2))
##################################################