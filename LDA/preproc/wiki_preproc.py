from preproc.nltk_preproc import NLTK_Preproc
import pandas as pd
import numpy as np
from collections import Counter
import pickle

# # hotel
# dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/prepare/source.csv'
# save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/prepare/source.pkl'

# restaurant
dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/yelp/prepare/wiki.csv'
save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/yelp/prepare/source.pkl'

# # car
# dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/prepare/source.csv'
# save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/prepare/source.pkl'

data = pd.read_csv(dir_, encoding='utf-8')
data.dropna(inplace=True)
wikis = data['article']
result = []
topic2wiki = {}
preproc = NLTK_Preproc()

print(f'Total number of review: {len(wikis)}')
for i,wiki in enumerate(wikis):
    try:
        result.append(preproc.preprocessing_nltk(wiki))
    except:
        print(f'Error occurred on {i}th wiki')
        pass

for topic in np.unique(data['topic'].tolist()):
    topic2wiki[topic] = []

for topic, wiki_preproc in zip(data['topic'], result):
    topic2wiki[topic].append(wiki_preproc)

final_result = {}

for topic in topic2wiki.keys():
    words = []
    for wiki in topic2wiki[topic]:
        words += wiki
    final_result[topic] = Counter(words)
pickle.dump(final_result,open(save_dir, 'wb'))
