import pickle
import pandas as pd
import numpy as np

class TopicCoherence:
    def __init__(self, lam, features_name ,top_n):
        self.K = lam.shape[0]
        self.top_n_dict = {}
        for k in range(self.K):
            tmp = pd.DataFrame()
            tmp['lambda'] = lam[k,:]
            tmp['words'] = features_name
            tmp.sort_values(by='lambda', ascending=False, inplace=True)
            top_n_words = tmp['words'].tolist()[:top_n]
            self.top_n_dict[k] = top_n_words

    def UMass_metric(self, words, data):
        coherence = 0
        for w1 in words:
            v_l = 0
            for doc in data:
                if w1 in doc:
                    v_l += 1

            for w2 in [w for w in words if w != w1]:
                v_m_v_l = 0
                for doc in data:
                    if w1 in doc and w2 in doc:
                        v_m_v_l += 1
                coherence += np.log( (v_m_v_l + 1) / v_l)
        return coherence

    def cal_coherence(self, data):
        coherence = 0
        for k in self.top_n_dict:
            words = self.top_n_dict[k]
            coherence += self.UMass_metric(words, data)
        return coherence

