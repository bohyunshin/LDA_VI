import pickle
from sklearn.feature_extraction.text import CountVectorizer
from online_lda import LDA_VI


# #run my LDA model
dir_ = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed_corpus.pickle"
# dir_ = 'preprocessed_review.pickle'

# for ordinary lda
alpha = 5
eta = 0.1
K = 10
maxIter = 200
maxIterDoc =100
threshold = 10
random_state = 42

data = pickle.load(open(dir_, 'rb'))
data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

lda = LDA_VI(alpha=alpha,eta=eta,K=K)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)