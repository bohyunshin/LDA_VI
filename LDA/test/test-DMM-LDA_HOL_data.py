import pickle
from sklearn.feature_extraction.text import CountVectorizer
from WORKING_LDA.WORKING_DMM_lda import LDA_VI

# #run my LDA model

# for ordinary lda
alpha = 4
eta = 0.1
maxIter = 500
maxIterDoc =100
threshold = 0.01
random_state = 42

K = 4
dir_ = '/Users/shinbo/Desktop/metting/LDA/0. data/JH_data/JH_data_cleaned.pkl'
data = pickle.load(open(dir_, 'rb'))
stop_words = ['good','great','go','get','place']
for i in range(len(data)):
    data[i] = [w for w in data[i] if w not in stop_words]

data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()


lda = LDA_VI(alpha=alpha,eta=0.1,K=K)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)

save_dir = f'/Users/shinbo/Desktop/metting/LDA/0. data/JH_data/result_{K}_topics.pkl'
pickle.dump(lda, open(save_dir, 'wb'))