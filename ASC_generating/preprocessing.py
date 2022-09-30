
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs

def parseSentence(line):
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem

def preprocess_train(domain):
    f = codecs.open('datasets/{}/train.txt'.format(domain), 'r', 'utf-8')
    out = codecs.open('preprocessed_data/{}/train.txt'.format(domain), 'w', 'utf-8')

    for line in f:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out.write(' '.join(tokens)+'\n')



def preprocess(domain):
    preprocess_train(domain)
   

if __name__ == '__main__':
    preprocess('restaurant')


