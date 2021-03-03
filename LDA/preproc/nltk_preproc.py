import re
import nltk

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet

stop_words = open('corenlp_stopwords.txt', 'r').read().split('\n')
stop_words = stop_words + stopwords.words('english')

class NLTK_Preproc:
    def __init__(self):
        pass

    def get_wordnet_pos(self, treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self, tokens):
        lemmatizer = WordNetLemmatizer()
        pos_tag = nltk.pos_tag(tokens)
        lemmas = [lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag)) for (word, pos_tag) in pos_tag]
        return lemmas

    def preprocessing_nltk(self, text, n_grams=[], stop_words=stop_words):
        text = text.lower()
        text = ' '.join([i for i in text.split(' ') if '@' not in i])
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》•©*⁎’–]', ' ', text)
        text = re.sub('±', '', text)
        text = re.sub('  ', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub('  ', ' ', text)
        text = re.sub('\n', ' ', text)
        text = re.sub('  ', ' ', text)
        text = re.sub('[(){}]', '', text)
        text = re.sub('[;:]', '', text)
        text = re.sub('[0-9]', '', text)

        text = text.strip()

        # filtering n_grams (concatenate words by - not to be tokenized)
        for gram in n_grams:
            gram_r = gram.replace(' ', '-')
            text = text.replace(gram, gram_r)
            stop_words.append(gram_r)

        tokens = word_tokenize(text)
        tokens = [i for i in tokens if i not in stop_words]

        lemma_pos_token = self.pos_tag(tokens)

        final_tokens = [i for i in lemma_pos_token if i not in stop_words]

        return final_tokens
