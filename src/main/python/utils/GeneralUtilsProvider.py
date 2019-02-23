import os

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def remove_if_exists(to_remove):
    if os.path.exists(to_remove):
        os.remove(to_remove)


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y_test)

    truth = label_binarizer.transform(y_test)
    pred = label_binarizer.transform(y_pred)
    return roc_auc_score(truth, pred, average=average)
