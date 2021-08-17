from preproc.nltk_preproc import NLTK_Preproc
import pandas as pd
import pickle

dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/prepare/hotel data.csv'
data = pd.read_csv(dir_, encoding='utf-8')
reviews = data['review']
save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/preproc.pkl'
word2stemming = {
    'paid':'pay', 'offered':'offer', 'serviced':'service', 'solved':'solve',
    'beverages':'beverage', 'coffees':'coffee', 'fruits':'fruit',
    'cafes':'cafe', 'restaurants':'restaurant', 'bathrooms':'bathroom', 'towels':'towel',
    'welcoming':'welcome'
}

result = []
preproc = NLTK_Preproc()

print(f'Total number of review: {len(reviews)}')
for i,review in enumerate(reviews):
    try:
        review = preproc.preprocessing_nltk(review)
        review = ' '.join(review)
        for word, stemming in word2stemming.items():
            review = review.replace(word, stemming)
        result.append(review.split(' '))
    except AttributeError:
        print(f'error on {i}')
        pass

    if i % 1000 == 0:
        print(f'{i} reviews finished')

pickle.dump(result, open(save_dir,'wb'))
