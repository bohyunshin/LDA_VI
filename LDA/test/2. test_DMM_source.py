import pickle
from sklearn.feature_extraction.text import CountVectorizer
from WORKING_LDA.WORKING_DMM_lda_source import LDA_VI

#run my LDA model
# # for hotel data
# dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/prepare/preproc.pkl'
# K = 4
# stop_words = ['hotel','stay','room','waikiki']
# source_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/prepare/source.pkl'
# save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/model/DMM_source_result.pkl'


# for restaurant data
dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/yelp/prepare/proproc.pkl'
K = 5
stop_words = ['the','and','to','it','be','have','in','for','of','this','that',
              'go','get','place','time','come',
              'take','make','back','really','say','one','even','well','look',
              'back','also']
source_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/yelp/prepare/source.pkl'
save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/yelp/model/DMM_source_result.pkl'

# # for car data
# dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/prepare/preproc.pkl'
# K = 4
# stop_words = ['the','and','to','it','be','have','in','for','of','this','that']
# source_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/prepare/source.pkl'
# save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/model/DMM_source_result.pkl'

# for ordinary lda
alpha = 50/K
eta = 0.01
maxIter = 500
maxIterDoc =100
threshold = 1
random_state = 42

data = pickle.load(open(dir_, 'rb'))
source = pickle.load(open(source_dir, 'rb'))

for i in range(len(data)):
    data[i] = [w for w in data[i] if w not in stop_words]

data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

lda = LDA_VI(alpha=alpha,eta=eta,K=K,source=source,whether_source=True,_lambda=1)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)
pickle.dump(lda, open(save_dir, 'wb'))