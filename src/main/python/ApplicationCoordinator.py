import os
from pprint import pprint

import numpy
import pandas
from matplotlib import pyplot
from matplotlib.legend_handler import HandlerLine2D
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from service.DataProvider import DataProvider
from service.DuplicateResolver import DuplicateResolver
from service.PipelineWrapper import PipelineWrapper
from service.WordCloudProvider import WordCloudProvider
from utils.ConfigurationProvider import ConfigurationProvider
from utils.GeneralUtilsProvider import remove_if_exists, multiclass_roc_auc_score
from utils.LoggerProvider import LoggerProvider

pandas.options.mode.chained_assignment = None


def get_shingles(text, char_ngram=5):
    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))


def jaccard(set_a, set_b):
    set_intersection = set_a & set_b
    set_union = set_a | set_b
    return float(len(set_intersection)) / float(len(set_union))


class ApplicationCoordinator(object):

    def __init__(self):
        self._resources = os.path.join(os.path.dirname(__file__), "../resources")
        self._cfg_file = self._resources + "/configs.cfg"

        self._cfg_provider = ConfigurationProvider(self._cfg_file)
        self._logger_provider = LoggerProvider(self._cfg_provider.get_property("logs", "log_level", 20))

        self._logger = self._logger_provider.get_logger(self.__class__.__name__)

        self._train_set = self._cfg_provider.get_property("datasets", "train_set", self._resources + "/datasets/train_set.csv")
        self._train_set_provider = DataProvider(self._train_set)

        self._test_set = self._cfg_provider.get_property("datasets", "test_set", self._resources + "/datasets/test_set.csv")
        self._test_set_provider = DataProvider(self._test_set)

    def _create_word_cloud_provider(self):
        width = self._cfg_provider.get_property("word_cloud", "width", "400")
        height = self._cfg_provider.get_property("word_cloud", "height", "200")
        min_font_size = self._cfg_provider.get_property("word_cloud", "min_font_size", "4")
        max_font_size = self._cfg_provider.get_property("word_cloud", "max_font_size", "50")
        max_words = self._cfg_provider.get_property("word_cloud", "max_words", "200")
        background_color = self._cfg_provider.get_property("word_cloud", "background_color", "white")
        return WordCloudProvider(width, height, min_font_size, max_font_size, max_words, background_color)

    def run(self):
        self._logger.info('*************************************')
        self._logger.info('Program started')
        self._logger.info('PID: {0}'.format(os.getpid()))

        # self._create_word_cloud()
        # self._find_duplicates()
        self._classify()
        # self._explore_classification_params()

    def _create_word_cloud(self):
        self._logger.info("Creating word cloud")
        word_cloud_provider = self._create_word_cloud_provider()
        text = self._train_set_provider.extract_content_from_top_category()
        word_cloud_provider.generate_and_show(text)

    def _find_duplicates(self):
        self._logger.info("Finding duplicates")
        ids, contents = self._train_set_provider.extract_ids_with_column("Content")
        duplicate_resolver = DuplicateResolver(ids, contents)
        similarity_threshold = float(self._cfg_provider.get_property("deduplication", "similarity_threshold", 0.7))
        export_path = self._cfg_provider.get_property("deduplication", "export_path", None)
        duplicate_resolver.find_duplicates(similarity_threshold, export_path)

    def _classify(self):
        self._logger.info("Starting classification")
        metrics_output_csv = self._cfg_provider.get_property("classification", "metrics_output_csv", "metrics_output_csv")
        remove_if_exists(metrics_output_csv)

        fold_number = int(self._cfg_provider.get_property("classification", "fold_number", 10))

        predictions_output_csv = self._cfg_provider.get_property("classification", "predictions_output_csv", None)
        remove_if_exists(predictions_output_csv)

        prep_train = self._train_set_provider.get_data_frame()[["Content", "Category"]]
        prep_test = self._test_set_provider.get_data_frame()[["Id", "Content"]]

        pipeline_wrappers = [
            PipelineWrapper("SVM", "BoW", prep_train, prep_test,
                            Pipeline([
                                ('vect', CountVectorizer(stop_words="english")),
                                ('clf', SGDClassifier())
                            ])),
            PipelineWrapper("SVM", "SVD", prep_train, prep_test,
                            Pipeline([
                                ('vect', CountVectorizer(stop_words="english")),
                                ('tfidf', TfidfTransformer()),
                                ('svd', TruncatedSVD()),
                                ('clf', SGDClassifier())
                            ])),
            PipelineWrapper("Random Forest", "BoW", prep_train, prep_test,
                            Pipeline([
                                ('vect', CountVectorizer(stop_words="english")),
                                ('clf', RandomForestClassifier())
                            ])),
            PipelineWrapper("Random Forest", "SVD", prep_train, prep_test,
                            Pipeline([
                                ('vect', CountVectorizer(stop_words="english")),
                                ('tfidf', TfidfTransformer()),
                                ('svd', TruncatedSVD()),
                                ('clf', RandomForestClassifier())
                            ])),
            PipelineWrapper("Custom Random Forest", "SVD", prep_train, prep_test,
                            Pipeline([
                                ('vect', CountVectorizer(stop_words="english")),
                                ('tfidf', TfidfTransformer()),
                                ('svd', TruncatedSVD()),
                                ('clf', RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=5, min_samples_leaf=2, max_features='auto', bootstrap=True))
                            ]))
        ]

        for wrapper in pipeline_wrappers:
            wrapper.cross_validate(fold_number, export_file=metrics_output_csv)
            wrapper.fit()

        pipeline_wrappers[-1].predict(export_file=predictions_output_csv)

    def _explore_classification_params(self):
        train_df = self._train_set_provider.get_data_frame()
        train_df.pop("Id")
        train_df.pop("Title")
        train_df.pop("RowNum")

        count_vectorizer = CountVectorizer(stop_words="english")
        train_data = count_vectorizer.fit_transform(train_df["Content"])

        train_labels = train_df.pop("Category")
        label_encoder = LabelEncoder()
        label_encoder.fit(train_labels)
        train_labels_ids = label_encoder.transform(train_labels)

        x_train, x_test, y_train, y_test = train_test_split(
            train_data,
            train_labels_ids,
            test_size=0.25
        )
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        print("No arguments score: {0}".format(multiclass_roc_auc_score(y_test, rf.predict(x_test))))

        n_estimators = [1, 40, 60, 100, 150, 300, 600, 700, 800]
        train_results = []
        test_results = []
        for estimator in n_estimators:
            rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
            rf.fit(x_train, y_train)
            train_results.append(multiclass_roc_auc_score(y_train, rf.predict(x_train)))
            test_results.append(multiclass_roc_auc_score(y_test, rf.predict(x_test)))

        line1, = pyplot.plot(n_estimators, train_results, "b", label="TrainAUC")
        line2 = pyplot.plot(n_estimators, test_results, "r", label="TestAUC")
        pyplot.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        pyplot.ylabel("AUC score")
        pyplot.xlabel("n_estimators")
        pyplot.show()

        max_depths = numpy.linspace(10, 100, num=10)
        train_results = []
        test_results = []
        for max_depth in max_depths:
            rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, n_jobs=-1)
            rf.fit(x_train, y_train)
            train_results.append(multiclass_roc_auc_score(y_train, rf.predict(x_train)))
            test_results.append(multiclass_roc_auc_score(y_test, rf.predict(x_test)))

        line1, = pyplot.plot(max_depths, train_results, "b", label="TrainAUC")
        line2 = pyplot.plot(max_depths, test_results, "r", label="TestAUC")
        pyplot.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        pyplot.ylabel("AUC score")
        pyplot.xlabel("max_depth")
        pyplot.show()

        max_features = list(range(1, train_df.shape[0], 25))
        train_results = []
        test_results = []
        for max_feature in max_features:
            rf = RandomForestClassifier(max_features=max_feature)
            rf.fit(x_train, y_train)
            train_results.append(multiclass_roc_auc_score(y_train, rf.predict(x_train)))
            test_results.append(multiclass_roc_auc_score(y_test, rf.predict(x_test)))

        line1, = pyplot.plot(max_features, train_results, "b", label="TrainAUC")
        line2 = pyplot.plot(max_features, test_results, "r", label="TestAUC")
        pyplot.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        pyplot.ylabel("AUC score")
        pyplot.xlabel("max_features")
        pyplot.show()

        rf = RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=5, min_samples_leaf=2, max_features='auto', bootstrap=True)
        rf.fit(x_train, y_train)
        print("Auc score: {0}".format(multiclass_roc_auc_score(y_test, rf.predict(x_test))))

        pprint(rf.get_params())

        # Number of trees in random forest
        n_estimators = [int(x) for x in numpy.linspace(start=10, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in numpy.linspace(10, 100, num=10)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        pprint(random_grid)
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=10)
        rf_random.fit(train_data, train_labels_ids)

        print(rf_random.best_params_)


if __name__ == "__main__":
    ApplicationCoordinator().run()

