
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

import warnings

warnings.simplefilter("ignore", DeprecationWarning)


class TopicModelling():

    numberTopics = 10
    numberWords = 5

    def _initVectorizer(self, stopWords : str ='english') -> CountVectorizer:
        return CountVectorizer(stop_words=stopWords)

    # Create and fit the LDA model
    def _initLDA(self, numTopics : int):
        return LDA(n_components=numTopics)
        # lda.fit(counData)

    def _loadCsv(self, csv_path : str) -> pd.DataFrame:
        # User Input
        return pd.read_csv(csv_path)

    def _info(self, dataFrame : pd.DataFrame):
        return dataFrame.info()

    @classmethod
    def _groupBy(cls, dataFrame : pd.DataFrame, by : str):
        # User Input
        return dataFrame.groupby(by=by)

    def _textProcess(self, dataFrame : pd.DataFrame, columnName : str, customFunc : str) -> pd.DataFrame:
        # TODO : Add flexibility
        dataFrame[columnName] = dataFrame[columnName].map(eval(customFunc))
        return dataFrame

    @classmethod
    def _createWordCloud(csl, array : pd.Series or list) -> bytes:
        combinedString = ' '.join(array)
        wCloud = WordCloud().generate(combinedString)
        # wCloud.to_image()
        # TODO : return image bytes
        return wCloud

    def _plot10MostCommonWords(self, count_data, count_vectorizer):
        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts += t.toarray()[0]

        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

        plt.bar(x_pos, counts, align='center')
        plt.xticks(x_pos, words, rotation=90)
        plt.xlabel('words')
        plt.ylabel('counts')
        plt.title('10 most common words')
        plt.show()

    def _printTopics(self, model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    # 1_plot10MostCommonWords(_initVectorizer().fit(papers['title_processed']), _initVectorizer())
    # # Print the topics found by the LDA model
    # print("Topics found via LDA:")
    # _printTopics(_initLDA(numberTopics), _initVectorizer(), numberWords)


