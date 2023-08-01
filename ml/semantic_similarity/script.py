
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class StringSimilarity():

    """### Build your own pairs for a particular use case. ###
       1) First build '_PAIRS' (): identify the target phrases and imagine possible combinations in which a user can phrase these.
       2) For example: Here I wanted to check whether a user wanted to sort in ascending order or descending order.
       3) A user can use different phrases for doing the sort and the phrase can be anywhere in the sentence so it is hard to write rules.
          Hence, I picked the anchor words: "largest to smallest", "lowest to highest" etc.
       4) For your use case you can pick your anchor phrases and put them together in a list of tuples, as given below"""


    _PAIRS = [("largest to smallest", True), ("smallest to largest", False),
             ("highest to lowest", True), ("lowest to highest", False),
             ("biggest to smallest", True), ("smallest to biggest", False),
             ("ascending order", True), ("descending order", False),
             ("increasing order", True), ("decreasing order", False)]


    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", device='cuda:0'):
        """
            Using "sentence-transformers/all-MiniLM-L6-v2" for creating text embeddings.
         """
        self._language_model = SentenceTransformer(model, device=device)


    def _check_sort_order(self, query: str) -> list:
        """
        :param query: Input string which the user wanted to sort.
        :return: Top 2 likely answers.
        """

        def _fetch_result(res: list) -> list:
            """
            :param res: Input list of top 2 matches based on cosine similarity.
            :return: Top 2 answers based on edit-distance.
            """
            return sorted(res, key=lambda x: x[2])

        results = []
        for pair in self._PAIRS:
            embeddings = self._language_model.encode([query, pair[0]])
            score = cosine_similarity([embeddings[0]], embeddings[1:])
            distance = levenshtein_distance(query, pair[0])
            results.append((pair, score[0][0], distance))

        # Sorting based on cosine similarity
        results = sorted(results, key=lambda x: x[1])
        print("\033[91m" + "Query: " + "\033[0m" + query)
        print("\033[92m" + "Likely answer --> " + "\033[0m" + f"{_fetch_result(results[-2:])[0]}", "\n")
        return _fetch_result(results[-2:])


sim = StringSimilarity(device='cpu')
sim._check_sort_order("Sort these values from lowest to highest 23432 3463 23523 6345752 25325")
sim._check_sort_order("Give me the sort order for these values starting from lowest to the highest?")
sim._check_sort_order("Let's arrange some numbers but go from biggest first and to lowest in the end.")

# Query: Sort these values from lowest to highest 23432 3463 23523 6345752 25325

#             Pair                       Cosine                       Edit-Distance
#     'lowest to highest'               0.6165699                          54
#     'highest to lowest'               0.6244998                          59