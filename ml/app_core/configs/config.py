
from topic_extraction.script import TopicModelling


_TASKS = ['', 'EDA', 'Regression', 'Classification', 'Clustering']
_MODELS = {'EDA' : TopicModelling,
           'Regression' : None}