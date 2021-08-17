import pickle
from sklearn.feature_extraction.text import CountVectorizer
from WORKING_LDA.WORKING_DMM_lda import LDA_VI

#run my LDA model
dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/yelp/prepare/proproc.pkl'
stop_words = ['the','and','to','it','be','have','in','for','of','this','that',
              'go','get','place','time','come',
              'take','make','back','really','say','one','even','well','look',
              'back','also']
save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/yelp/model/CDMM_result.pkl'

eta_seed = 800
eta_not_seed = 0.0001
# set seed words
seed_words = dict()
# seed_sets = pd.read_csv('/Users/shinbo/Desktop/metting/LDA/meeting materials/21.05.13/seed_words.csv')
# for col in seed_sets.columns:
#     seed_words[col] = seed_sets[col].tolist()
seed_words['price'] = ['price','pricing','reasonable','reasonably','value','cheap','cheaper',
                       'affordable','tax','cost','expensive','pricey','fair',]
seed_words['food'] = ['restaurant','jajangmyun','brushetta','saltado','meal','michelin',
                      'pubs','fajitas','salsas','buffet']
seed_words['drink'] = ['drink','drinks','beer','bartender','alcoholic','refill','cocktail','margarita',
                       'beverage','thirst','soju','pomegranate','alcohol','bar','blender']
seed_words['service'] = ['service','customer','mismanage','staff','friendly','courteous',
                         'inattentive','attentive','prompt','speedy','waiter','cleanliness',
                         'receptive','communicator','interpersonal','unresponsive']
seed_words['ambience'] = ['ambience','atmoshpere','decor','minimalist','parisian',
                          'romantic','homely','nostalgic','whimsical','cozy','shabby',
                          'dimly','music','decorative']

# for ordinary lda
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