import xml.etree.ElementTree
path_base = r"../datasets/raw/"
# import  pandas as pd
import  pickle as pk
import string
import  csv, re
from tqdm import tqdm
import json

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stops = stopwords.words('english')
stops.extend(['us', "didnt",'im', 'make','today', 'feel', 'sometimes', 'ive' 'whatever','never','although','anyway','get','always', 'usually', 'want', 'go','would', 'one'])
strong_sentiment_words=[]
# strong_sentiment_words=['great', 'bad', 'awesome', 'perfect', 'good', 'cute', 'weird', 'fine', 'nice']
from sklearn.feature_extraction.text import TfidfVectorizer


def cleaning (text):
    # text=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    words_to_repere = ["it's", "don't", "'", "isn't", "you've", "'s"]
    for w in words_to_repere:
        text = text.replace(w, ' ' + w + ' ')
    text = re.sub(r'[^\w\s]', '', text)
    # if 'but' in text:
    #     text = text.split('but')[-1]
    return text

def generate_semeval14_training(train=True):
    file_src= 'Restaurants_Test_Gold.xml'
    file_save= 'test'
    if train:
        file_src='Restaurants_Train_v2.xml'
        file_save = 'train'
    data = {}
    e = xml.etree.ElementTree.parse(path_base+'semeval/' + file_src).getroot()
    reviews = e.findall('sentence')
    for review in (reviews):
        text = review.find('text').text
        aspect_term = []
        aspect_polarity = []
        options = review.findall('aspectCategories')
        for option in (options):
            suboptions = option.findall('aspectCategory')
            for suboption in (suboptions):
                aspect = suboption.get("category")
                polarity = suboption.get("polarity")
                if aspect=='anecdotes/miscellaneous':
                    # continue
                    aspect='anecdotes'
                aspect_term.append(aspect)
                aspect_polarity.append(polarity)
        #

        # if aspect =='anecdotes':continue
        text= text.strip().lower()
        text= text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join([w for w in word_tokenize(text) if w not in stops])

        if len(aspect_term)>1:
            continue
        if aspect == 'anecdotes': continue
        # print(text, ':----> ', aspect_term)
        # data.append((text, aspect_term))
        try:
            data[' '.join(aspect_term)]+=' '+text
        except:
            data[' '.join(aspect_term)]= text
        # for _, a in enumerate(aspect_term):
        #     data.append([text, a, aspect_polarity[_]])


    pk.dump(data, open('../datasets/semeval/{}.pk'.format(file_save), 'wb'))
    # print(len(data))



def process_sentihood():
    categories = {'general','price', 'transit-location', 'safety'}
    # pk.dump(categories, open('../datasets/sentihood/categories.pk', 'wb'))
    for phase in ['train', 'test']:
        data_to_save = dict()
        with open('../datasets/raw/sentihood/sentihood-{}.json'.format(phase),'r') as f :
            data= json.load(f)
            for d in data:
                aspect_category = [ac['aspect'] for ac in d['opinions']]
                aspect_category= list(set(aspect_category))
                if not len(aspect_category) or len(aspect_category)>1 or not len(set(aspect_category).intersection(categories)):
                    continue
                text = d['text']
                text= text.replace('LOCATION1','').replace('LOCATION2','')
                text = text.strip().lower()
                text = text.translate(str.maketrans('', '', string.punctuation))
                text = ' '.join([w for w in word_tokenize(text) if w not in stops])
                try:
                    data_to_save[' '.join(aspect_category)] += ' ' + text
                except:
                    data_to_save[' '.join(aspect_category)] = text
        pk.dump(data_to_save, open('../datasets/sentihood/{}.pk'.format( phase), 'wb'))
        print(data_to_save.keys())
        f.close()


def process_sentihood_tf_idf():
    categories = {'general','price', 'transit-location', 'safety'}
    for phase in ['train', 'test']:
        data_to_save=dict()
        corpus = list()
        aspect = []
        with open('../datasets/raw/sentihood/sentihood-{}.json'.format(phase),'r') as f :
            data= json.load(f)
            for d in data:
                aspect_category = [ac['aspect'] for ac in d['opinions']]
                aspect_category = list(set(aspect_category))
                aspect.append(aspect_category)
                text = d['text']
                text = text.replace('LOCATION1', '').replace('LOCATION2', '')
                text = text.strip().lower()
                text = ' '.join([w for w in word_tokenize(text) if w not in stops+strong_sentiment_words])
                text = text.translate(str.maketrans('', '', string.punctuation))
                corpus.append(text)

        f.close()
        vectorizer = TfidfVectorizer()
        tf_idf = vectorizer.fit_transform(corpus)
        for i in range(len(corpus)):
            aspect_category = aspect[i]
            text = corpus[i]
            if not len(aspect_category) or len(aspect_category) > 1 or not len(
                    set(aspect_category).intersection(categories)):
                continue
            vocab = dict(zip(vectorizer.get_feature_names(), tf_idf.toarray()[i]))
            print(text)
            text = ' '.join([w for w in text.split() if w in vocab and vocab.get(w)>.4])
            print(text)
            print(aspect_category)
            print('------------')

            try:
                data_to_save[' '.join(aspect_category)] += ' ' + text
            except:
                data_to_save[' '.join(aspect_category)] = text

        pk.dump(data_to_save, open('../datasets/sentihood/{}.pk'.format(phase), 'wb'))
        print(data_to_save.keys())
        f.close()
def process_semeval_tf_idf():

    for phase, file_src in zip(['train', 'test'], ['Restaurants_Train_v2.xml','Restaurants_Test_Gold.xml']):
        data_to_save=dict()
        corpus = list()
        aspect = []

        e = xml.etree.ElementTree.parse(path_base + 'semeval/' + file_src).getroot()
        reviews = e.findall('sentence')
        for review in (reviews):
            text = review.find('text').text
            aspect_term = []
            options = review.findall('aspectCategories')
            for option in (options):
                suboptions = option.findall('aspectCategory')
                for suboption in (suboptions):
                    aspect_term.append(suboption.get("category"))

            aspect.append(aspect_term)
            text = text.strip().lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ' '.join([w for w in word_tokenize(text) if w not in stops+strong_sentiment_words])
            corpus.append(text)


        vectorizer = TfidfVectorizer()
        tf_idf = vectorizer.fit_transform(corpus)
        for i in tqdm(range(len(corpus))):
            aspect_category = aspect[i]
            text = corpus[i]
            if not len(aspect_category) or len(aspect_category) > 1 or 'anecdotes/miscellaneous' in aspect_category :
                continue

            vocab = dict(zip(vectorizer.get_feature_names(), tf_idf.toarray()[i]))
            # print(text)
            text = ' '.join([w for w in text.split() if w in vocab and vocab.get(w)>.4])
            # print(text)
            # print(aspect_category)
            # print('------------')

            try:
                data_to_save[' '.join(aspect_category)] += ' ' + text
            except:
                data_to_save[' '.join(aspect_category)] = text

        pk.dump(data_to_save, open('../datasets/semeval/{}.pk'.format(phase), 'wb'))
        # print(data_to_save.keys())
        # # f.close()



if __name__ == '__main__':

    process_sentihood_tf_idf()
    process_semeval_tf_idf()
   


