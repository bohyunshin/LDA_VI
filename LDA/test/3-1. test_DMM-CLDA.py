import pickle
from sklearn.feature_extraction.text import CountVectorizer
from WORKING_LDA.WORKING_DMM_lda import LDA_VI

#run my LDA model
dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/prepare/preproc.pkl'
stop_words = ['hotel','stay','room','waikiki']
save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/model/CDMM_result.pkl'


eta_seed = 800
eta_not_seed = 0.0001
# set seed words
seed_words = dict()
# seed_words['price'] = ["price", 'fee','cost','value','money','pay','expensive','charge','pricey',
#                        'cheaper','cheap','rate','reasonably','reasonable']
# seed_words['service'] = ["service", "staff", "helpful", "service", "friendly", "pool", "desk",
#                               "valet", "welcome", "parking", "offer", "manager",
#                               "help", "courteous", "check", "towel",
#                               "solve", 'serve']
# seed_words['food'] = ['food', 'coffee','water','beverage','breakfast','lunch','dinner','tea',
#                            'fruit','cafe','coconut','restaurant','drink','starbucks']
# seed_words['accomodation'] = ['accomodation','bathroom','lobby','elevator','towel','balcony',
#                                     'location','far','close','stay']

seed_words['price'] = ['pricing','price','rate','value','reasonably','reasonable','unreasonable','budget',
                       'conscience','cost','affordable',
                       'bid','overprice','money','expensive','overrate','promotion','cheap']
seed_words['food'] = ['food','restaurant','hamburger','lunch','dinner','breakfast',
                      'meal','pickle','steakhouse','desserts','sushi','chicken',
                      'delicious']
seed_words['drink'] = ['drink','cocktail','beer','beverage','bar','alcoholic','drinking',
                       'lemonade','wine','teas','tangueray','nibble','lattes','tonic']
seed_words['service'] = ['service','staff','customer','dedication',
                         'friendly','attention','attentive','vallet','truely',
                         'cleanliness','housekeep'
                         ]

K = len(seed_words.keys())
alpha = 50/K
eta = 0.01
maxIter = 500
maxIterDoc =100
threshold = 0.01
random_state = 42

data = pickle.load(open(dir_, 'rb'))
for i in range(len(data)):
    data[i] = [w for w in data[i] if w not in stop_words]

data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

lda = LDA_VI(alpha=alpha,eta_seed=eta_seed, eta=eta,
             eta_not_seed=eta_not_seed, K=K,
             seed_words=seed_words, confirmatory=True)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)

pickle.dump(lda, open(save_dir, 'wb'))