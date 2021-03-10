from preproc.nltk_preproc import NLTK_Preproc
import pandas as pd
import pickle

dir_ = '/Users/shinbo/Desktop/metting/LDA/0. data/hotel data/3. third data.csv'
data = pd.read_csv(dir_)
reviews = data['reviews.text']
result = []
preproc = NLTK_Preproc()

print(f'Total number of review: {len(reviews)}')
for i,review in enumerate(reviews):
    try:
        result.append(preproc.preprocessing_nltk(review))
    except AttributeError:
        pass

    if i % 1000 == 0:
        print(f'{i} reviews finished')


save_dir = '/Users/shinbo/Desktop/metting/LDA/0. data/hotel data/third_data_preproc_pos.pkl'
pickle.dump(result, open(save_dir,'wb'))
