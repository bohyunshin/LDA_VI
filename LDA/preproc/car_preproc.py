import pandas as pd
import numpy as np
import pickle
from preproc.nltk_preproc import NLTK_Preproc

data = pd.read_csv('/Users/shinbo/Desktop/metting/LDA/meeting materials/21.05.03/preproc_all_data.csv')
review_number = data['review_number']
preproc = data['preproc']
concat = {}
for i in np.unique(review_number.tolist()):
    concat[i] = ''

for number, review in zip(review_number,preproc):
    concat[number] += ' ' + review

result = []
preproc = NLTK_Preproc()

print(f'Total number of review: {len(concat.keys())}')
for i,review in enumerate(list(concat.values())):
    try:
        result.append(preproc.preprocessing_nltk(review))
    except AttributeError:
        pass

    if i % 1000 == 0:
        print(f'{i} reviews finished')

dir_ = '/Users/shinbo/Desktop/metting/LDA/meeting materials/21.05.03/preproc.pkl'
pickle.dump(result, open(dir_,'wb'))