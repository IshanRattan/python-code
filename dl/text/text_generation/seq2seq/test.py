

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

text = pd.read_csv(
    '/Users/ishanrattan/Desktop/Study/github/python-code/dl/text_generation/seq2seq/datasets/sentences.csv')
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(text['sentence'])

cv = CountVectorizer()
c_vector = cv.fit_transform(text['sentence'])
# print(c_vector)

# Add in the rest of the arguments
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))

    # Transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i] :zipped[i] for i in vector[vector_index].indices})

    # Sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
# print(return_weights(vectorizer.get_feature_names(), vectorizer.vocabulary_, vector, 8, 3))


def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
        # Call the return_weights function and extend filter_list
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)

    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)


# Call the function to get the list of word indices
filtered_words = words_to_filter(vectorizer.get_feature_names(), vectorizer.vocabulary_, vector, 3)

# Filter the columns in text_tfidf to only those in filtered_words
filtered_text = vector[:, list(filtered_words)]
print(filtered_text)