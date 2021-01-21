import pickle
from sklearn.feature_extraction.text import CountVectorizer
from WORKING_DMM_lda import LDA_VI

# #run my LDA model
# dir_ = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed_corpus.pickle"
dir_ = 'preprocessed_review.pickle'
hotel = pickle.load(open(dir_, 'rb'))

# for ordinary lda
alpha = 5
eta = 0.1
K = 4
maxIter = 1000
maxIterDoc =100
threshold = 0.01
random_state = 42

data = pickle.load(open(dir_, 'rb'))
data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

lda = LDA_VI(alpha=alpha,eta=eta,K=K)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)

save_dir = '../../model_lda/DMM_result.pkl'
pickle.dump(lda, open(save_dir, 'wb'))