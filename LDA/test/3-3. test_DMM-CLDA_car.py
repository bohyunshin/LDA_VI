import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from WORKING_LDA.WORKING_DMM_lda import LDA_VI

#run my LDA model
dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/prepare/preproc.pkl'
stop_words = ['the','and','to','it','be','have','in','for','of','this','that']
save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/model/CDMM_result.pkl'

eta_seed = 800
eta_not_seed = 0.0001
# set seed words
seed_words = dict()
# seed_sets = pd.read_csv('/Users/shinbo/Desktop/metting/LDA/meeting materials/21.05.03/seedword.csv')
# for col in seed_sets.columns:
#     seed_words[col] = seed_sets[col].tolist()
seed_words['interior/exterior'] = ['interior','exterior','inside','saddle',
                                   'style','design','exquisite','glossy',
                                   'color','appearance','elegant']
seed_words['seat'] = ['passenger','seating','seat','legroom','comfortable','unfortable',
                      'lumbar','setback','chair','backseat','headrest','occupant',
                      'setback']
seed_words['engine'] = ['turbo','cylinder','transmission','injection','motor',
                        'horsepower','turbocharger','acceleration']
seed_words['technology'] = ['feature','electronic','tech','gadget','musthave',
                            'hightech','equipment','technology','software','hardwere']
# seed_words['driving_experience'] = []


# for ordinary lda
K = len(seed_words.keys())
alpha = 50/K
eta = 0.01
maxIter = 5000
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