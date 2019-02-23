from matplotlib import pyplot
from wordcloud import WordCloud


class WordCloudProvider(object):
    def __init__(self, width, height, min_font_size, max_font_size, max_words, background_color):
        self._width = int(width)
        self._height = int(height)
        self._min_font_size = int(min_font_size)
        self._max_font_size = int(max_font_size)
        self._max_words = int(max_words)
        self._background_color = str(background_color)
        self._word_cloud = self._create_word_loud()

    def get_word_cloud(self):
        return self._word_cloud

    def generate_and_show(self, text, should_show=True):
        if text is None or len(text) == 0:
            raise Exception("Failed to create Word Cloud, cause of empty data")
        generated_word_cloud = self._word_cloud.generate(text)
        if should_show:
            pyplot.imshow(generated_word_cloud, interpolation='bilinear')
            pyplot.axis("off")
            pyplot.show()
        return generated_word_cloud

    def _create_word_loud(self):
        return WordCloud(width=self._width,
                         height=self._height,
                         min_font_size=self._min_font_size,
                         max_font_size=self._max_font_size,
                         max_words=self._max_words,
                         background_color=self._background_color)
