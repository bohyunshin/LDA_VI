import pickle
from sklearn.feature_extraction.text import CountVectorizer
from WORKING_LDA.WORKING_DMM_lda import LDA_VI

# #run my LDA model
# dir_ = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed_corpus.pickle"
type = 'first_data'
dir_ = f'/Users/shinbo/Desktop/metting/LDA/0. data/hotel data/preproc/{type}_preproc.pkl'
hotel = pickle.load(open(dir_, 'rb'))

# for ordinary lda
alpha = 5
eta = 0.1
maxIter = 500
maxIterDoc =100
threshold = 0.01
random_state = 42

K = 4
data = pickle.load(open(dir_, 'rb'))
stop_words = ['hotel','stay','room','waikiki']
for i in range(len(data)):
    data[i] = [w for w in data[i] if w not in stop_words]

data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

cv.get_feature_names()


lda = LDA_VI(alpha=alpha,eta=0.1,K=K)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)

save_dir = f'../../../model_lda/{type}/DMM_result_{type}_stop_words.pkl'
pickle.dump(lda, open(save_dir, 'wb'))