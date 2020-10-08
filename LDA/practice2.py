from online_lda import LDA_VI

dir_ = 'preprocessed_review.pickle'
K=5
lda = LDA_VI(dir_, 5, 0.1, K)
lda.train(0.01,1000)
# lda.train(threshold=.01, max_iter=1000, max_iter_doc=1000)