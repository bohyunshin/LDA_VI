from preproc.nltk_preproc import NLTK_Preproc
import pandas as pd
import pickle

dir_ = '/Users/shinbo/Desktop/metting/LDA/meeting materials/21.05.13/yelp_restaurant_sampled.csv'
data = pd.read_csv(dir_)
reviews = data['text']
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


save_dir = '/Users/shinbo/Desktop/metting/LDA/meeting materials/21.05.13/yelp_proproc.pkl'
pickle.dump(result, open(save_dir,'wb'))
