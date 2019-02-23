import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class DuplicateResolver(object):

    def __init__(self, ids, contents):
        self._ids = ids
        self._contents = contents
        self._tf_id_vectorizer = TfidfVectorizer().fit_transform(self._contents)

    def find_duplicates(self, similarity_threshold=0.7, export_to=None):
        duplicates = []
        for i_doc in range(0, len(self._contents)):
            first_document_id = self._ids[i_doc]
            cosine_similarities = list(linear_kernel(self._tf_id_vectorizer[i_doc:i_doc + 1], self._tf_id_vectorizer).flatten())
            for j_doc in range(0, len(self._contents)):
                second_document_id = self._ids[j_doc]
                if first_document_id != second_document_id and cosine_similarities[j_doc] >= similarity_threshold:
                    duplicates.append((first_document_id, second_document_id, cosine_similarities[j_doc]))
        if export_to is not None:
            with pandas.option_context('display.precision', 2):
                pandas.DataFrame(duplicates, columns=['ID_1', 'ID_2', 'Similarity']).to_csv(str(export_to), sep="\t")

        return duplicates
