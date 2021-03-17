import pickle
from sklearn.feature_extraction.text import CountVectorizer
from evaluation import TopicCoherence

type = 'first_data'
dir_ = f'/Users/shinbo/Desktop/metting/LDA/0. data/hotel data/preproc/{type}_preproc.pkl'
hotel = pickle.load(open(dir_, 'rb'))

data = pickle.load(open(dir_, 'rb'))
stop_words = ['hotel','stay','room','waikiki']
for i in range(len(data)):
    data[i] = [w for w in data[i] if w not in stop_words]

data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

features_name = cv.get_feature_names()

dir_ = f'../../../model_lda/{type}/DMM_result_{type}_stop_words.pkl'
mod = pickle.load(open(dir_, 'rb'))
lam = mod.components_

TC = TopicCoherence(lam, features_name, 20)
coherence = TC.cal_coherence(data)
print(coherence)