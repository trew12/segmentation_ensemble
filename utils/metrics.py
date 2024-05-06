import numpy as np


def compute_confusion_matrix(real, pred):
    diff = pred - real
    sum = pred + real
    FP = np.sum(diff == 1)
    FN = np.sum(diff == 255)
    TP = np.sum(sum == 2)
    TN = np.sum(sum == 0)
    return FP, FN, TP, TN
