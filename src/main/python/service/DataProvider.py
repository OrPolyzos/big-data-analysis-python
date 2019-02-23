import os

import pandas


class DataProvider(object):
    def __init__(self, csv_file, separator="\t"):
        if len(csv_file) <= 0 or not os.path.exists(csv_file):
            raise IOError("Failed to find .csv file: {0}".format(csv_file))
        self._csv_file = csv_file
        self._data_frame = pandas.read_csv(self._csv_file, sep=separator)

    def get_data_frame(self):
        return self._data_frame

    def extract_content_from_top_category(self):
        top_category = sorted(self._data_frame.groupby(['Category']), key=lambda item: len(item[1]), reverse=True)[0][0]
        whole_content = " ".join(self._data_frame.loc[self._data_frame["Category"] == top_category]["Content"])
        return whole_content

    def extract_ids_with_column(self, extra_column):
        ids = []
        extra_columns = []
        for index, row in self._data_frame.iterrows():
            ids.append(row["Id"])
            extra_columns.append(row[extra_column])
        return ids, extra_columns
