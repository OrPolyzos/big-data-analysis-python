import os

import numpy
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

from utils.LoggerProvider import LoggerProvider


class PipelineWrapper(object):

    def __init__(self, classifier_name, feature_name, train_df, test_df, pipeline):
        self._logger = LoggerProvider.get_instance().get_logger("{0}_{1}_{2}".format(self.__class__.__name__, classifier_name, feature_name).replace(" ", ""))
        self._scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        self._classifier_name = classifier_name
        self._feature_name = feature_name
        self._pipeline = pipeline
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(train_df["Category"].tolist())
        self._train = {
            "data": train_df["Content"].tolist(),
            "target_names": list(self._label_encoder.classes_),
            "target": list(self._label_encoder.transform(train_df["Category"]))
        }
        self._test = {
            "id": test_df["Id"].tolist(),
            "data": test_df["Content"].tolist(),
            "target_names": [],
            "target": []
        }

        self._metrics = {
            "classifier": self._classifier_name,
            "feature": self._feature_name,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f_measure": 0,
            "auc": 0

        }
        self.metrics_line = None

    def cross_validate(self, number_of_folds=10, export_file=None):
        scores = cross_validate(self._pipeline, self._train["data"], self._train["target"], cv=number_of_folds, scoring=self._scoring)
        self._metrics["accuracy"] = numpy.mean(scores["test_accuracy"])
        self._metrics["precision"] = numpy.mean(scores["test_precision_macro"])
        self._metrics["recall"] = numpy.mean(scores["test_recall_macro"])
        self._metrics["f_measure"] = 2 * (self._metrics["precision"] * self._metrics["recall"]) / (self._metrics["precision"] + self._metrics["recall"])

        self._logger.info("{0}CFV Metrics: Accuracy: {1}, Precision: {2}, Recall: {3}, F-Measure: {4}".format(
            number_of_folds, self._metrics["accuracy"], self._metrics["precision"], self._metrics["recall"], self._metrics["f_measure"]
        ))
        if export_file is not None:
            if not os.path.exists(export_file):
                with open(export_file, "w+") as metrics_file:
                    metrics_file.write("classifier\tfeature\taccuracy\tprecision\trecall\tf-measure\tauc\n")

            with open(export_file, "a+") as metrics_file:
                metrics_file.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(
                    self._metrics["classifier"],
                    self._metrics["feature"],
                    self._metrics["accuracy"],
                    self._metrics["precision"],
                    self._metrics["recall"],
                    self._metrics["f_measure"],
                    self._metrics["auc"]))
            self._logger.info("Wrote metrics to file: {0}".format(export_file))
        return self._metrics

    def fit(self):
        self._pipeline.fit(self._train["data"], self._train["target"])

    def predict(self, export_file=None):
        self._test["target"] = self._pipeline.predict(self._test["data"])
        self._test["target_names"] = self._label_encoder.inverse_transform(self._test["target"])
        if export_file is not None:
            with open(export_file, "w+") as predictions_file:
                predictions_file.write("Test_Document_ID\tPredicted_Category\n")
                for i in range(len(self._test["id"])):
                    predictions_file.write("{0}\t{1}\n".format(self._test["id"][i], self._test["target_names"][i]))
                self._logger.info("Wrote predictions to file: {0}".format(export_file))
        return self._test
