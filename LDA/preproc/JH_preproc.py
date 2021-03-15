from preproc.nltk_preproc import NLTK_Preproc
import pandas as pd
import pickle

dir_ = '/Users/shinbo/Desktop/metting/LDA/0. data/JH_data/JH_data.csv'
data = pd.read_csv(dir_, encoding='euc-kr')
reviews = data['review.entry']
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


save_dir = '/Users/shinbo/Desktop/metting/LDA/0. data/JH_data/JH_data_cleaned.pkl'
pickle.dump(result, open(save_dir,'wb'))
