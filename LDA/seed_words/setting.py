import gensim
import pandas as pd

def seed_words(topics, save_dir):
    w2v_dir = '/Users/shinbo/Desktop/metting/LDA/paper/word_embedding/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'
    word2vec_model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
        w2v_dir, binary=True
    )

    seed_words = pd.DataFrame()
    for topic in topics:
        a = word2vec_model.most_similar(topic, topn = 500)
        seed_words[topic] = [i[0] for i in a]
    seed_words.to_csv(save_dir,index=False)

if __name__ == '__main__':
    # hotel data
    topics = ['price','service','food','drink','accomodation']
    save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/prepare/seed_words.csv'
    seed_words(topics, save_dir)

    # yelp data
    topics = ['price', 'service', 'food', 'drink', 'ambience']
    save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/yelp/prepare/seed_words.csv'
    seed_words(topics, save_dir)