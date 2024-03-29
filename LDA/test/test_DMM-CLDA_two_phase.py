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
maxIter = 5000
maxIterDoc =100
threshold = 0.01
random_state = 42

eta_seed = 400
eta_not_seed = 0.0001
# set seed words
seed_words = dict()
seed_words['price'] = ["price", "prices", "priced", "fee", "fees", "cost", "value", "money", "pay",
                             "expensive", "charge", "pricey", "paid", "cheaper", "rate", "reasonably", "cheap"]
seed_words['service'] = ["staff", "helpful", "service", "friendly", "pool", "desk",
                              "valet", "welcoming", "parking", "offered", "offer", "manager",
                              "weclome", "serviced", "service", "help", "courteous", "check", "towels",
                              "solved", "solve", "offers", 'serve']

seed_words['food'] = ['food', 'coffee','water','beverage','beverages','breakfast','luanch','dinner','tea',
                           'fruit','fruits','starbucks','cafe','cafes','coconut','restaurant', 'restaurants', 'starbucks','drink']
seed_words['accomodation'] = ['bathrooms','bathroom','lobby','elevator','towel','balcony',
                                    'location','far','close','front','stay']

K = len(seed_words.keys())+1
data = pickle.load(open(dir_, 'rb'))
data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()
two_phase_words = ['hotel','stay','room','waikiki']

lda = LDA_VI(alpha=alpha,eta_seed=eta_seed, eta=0.1,
             eta_not_seed=eta_not_seed, K=K,
             seed_words=seed_words, confirmatory=True,
             two_phase_words=two_phase_words)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)

save_dir = f'../../../model_lda/{type}/CDMM_result_{type}_two_phase_400.pkl'
pickle.dump(lda, open(save_dir, 'wb'))