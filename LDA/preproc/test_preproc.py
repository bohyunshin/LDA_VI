from preproc.nltk_preproc import NLTK_Preproc
import pandas as pd
import pickle

dir_ = '/Users/shinbo/Desktop/metting/LDA/0. data/hotel data/first_data.csv'
data = pd.read_csv(dir_, encoding='euc-kr')
reviews = data['review'].tolist()
result = []
preproc = NLTK_Preproc()

print(f'Total number of review: {len(reviews)}')
for i,review in enumerate(reviews):
    result.append(preproc.preprocessing_nltk(review))

    if i % 1000 == 0:
        print(f'{i} reviews finished')


save_dir = '/Users/shinbo/Desktop/metting/LDA/0. data/hotel data/first_data_preproc.pkl'
pickle.dump(result, open(save_dir,'wb'))
