import json
import re
import gensim
import pickle as pk
# print(pk.__version__)
import os
import xml.etree.ElementTree
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stops = stopwords.words('english')
stops.extend(['us', "didnt",'im','couldnt', 'make','today', 'feel', 'sometimes', 'ive' 'whatever','never','although','anyway','get','always', 'usually', 'want', 'go','would', 'one'])
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse

def semeval(threshold=.2):
    model = gensim.models.Word2Vec.load(r'embeddings/restaurant.bin')
    path_base= '../datasets/raw/semeval/'
    categories= pk.load(open('../datasets/semeval/categories.pk', 'rb'))
    category_seed_words= pk.load(open('../datasets/semeval/categories_seeds.pk', 'rb'))
    categories.append('anecdotes')
    for phase, file_src,  in zip(['train', 'test'], ['Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml']):
        data_to_save =[]
        corpus = list()
        aspect = []

        e = xml.etree.ElementTree.parse(path_base + file_src).getroot()
        reviews = e.findall('sentence')
        for review in tqdm(reviews):
            text = review.find('text').text
            aspect_category = []
            aspect_polarity =dict()
            options = review.findall('aspectCategories')
            for option in (options):
                suboptions = option.findall('aspectCategory')
                for suboption in (suboptions):
                    current_aspect =suboption.get("category").replace('anecdotes/miscellaneous', 'anecdotes')
                    aspect_category.append(current_aspect)
                    aspect_polarity[current_aspect]=suboption.get("polarity")

            # aspect.append(aspect_term)
            text = text.strip().lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text_w_stops = [w for w in word_tokenize(text) if w not in stops]
            corpus.append(text)


            category_representatives={a:[] for a in aspect_category}

            for a in aspect_category:
                if a =='anecdotes':
                    category_representatives[a].append('anecdotes')
                    continue

                representatives=[]
                for w in text_w_stops:
                    if w not in model.wv or  w in representatives :
                        continue
                    seed_vec = np.array([model.wv[w_] for w_ in category_seed_words.get(a) if w_ in model.wv])
                    score = np.max(cosine_similarity(seed_vec, np.expand_dims(model.wv[w], axis=0)), axis=0)[0]

                    if score > threshold:
                        representatives.append(w)
                representatives = list(set(representatives).difference(set(categories)))

                category_representatives[a]= representatives

            current_data= dict()
            current_data['text']= text
            asp_tem =[]

            for aspect in categories:

                if aspect in category_representatives:
                    temp={'category':aspect, 'polarity':aspect_polarity.get(aspect), 'auxiliary':list(set(category_representatives.get(aspect)))}
                else:
                    # temp = {'category': aspect, 'polarity': 'none', 'auxiliary': [aspect] }
                    temp = {'category': aspect, 'polarity': 'none', 'auxiliary': [] }
                asp_tem.append(temp)
            current_data['aspect']=asp_tem
            data_to_save.append(current_data)
        print(len(data_to_save))
        with open('../datasets/semeval/bert_{}.json'.format(phase), 'w') as f :
            json.dump(data_to_save, f, indent=3)
        f.close()

def sentihood(threslod=.3):
    model = gensim.models.Word2Vec.load(r'D:\data\domain_specific_embedding\restaurant/restaurant.bin')
    category_seed_words = pk.load(open('../datasets/sentihood/categories_seeds.pk', 'rb'))
    # categories.append('anecdotes')
    categories = {'general', 'price', 'transit-location', 'safety'}
    for phase in ['train', 'test', 'dev']:
        data_to_save = []
        with open('../datasets/raw/sentihood/sentihood-{}.json'.format(phase), 'r') as f:
            data = json.load(f)
            for d in tqdm(data):
                aspects = d['opinions']
                text = d['text']
                # text = text.replace('/', ' ')
                text_w_stops = text.replace('LOCATION1', '').replace('LOCATION2', '')
                text_w_stops = text_w_stops.strip().lower()
                text_w_stops = text_w_stops.translate(str.maketrans('', '', string.punctuation))
                text_w_stops = [w for w in word_tokenize(text_w_stops) if w not in stops]

                category_representatives={a['aspect']:[] for a in aspects if a['aspect'] in categories}
                aspect_category=[a['aspect'] for a in aspects if a['aspect'] in categories]
              
                aspect_polarity={a['target_entity']+'#'+a['aspect']: a['sentiment'] for a in aspects if a['aspect'] in categories}
                for a in aspect_category:

                    representatives=[]
                    for w in text_w_stops:
                        if w not in model.wv or  w in representatives :
                            continue
                        # print(a)
                        seed_vec = np.array([model.wv[w_] for w_ in category_seed_words.get(a) if w_ in model.wv])
                        score = np.max(cosine_similarity(seed_vec, np.expand_dims(model.wv[w], axis=0)), axis=0)[0]

                        if score > threslod:
                            representatives.append(w)
                    category_representatives[a]= representatives

                current_data= dict()
                current_data['text']= text.replace('LOCATION1', 'LOCATION - 1 -').replace('LOCATION2', 'LOCATION - 2 -').lower()
                asp_tem =[]
                loc=dict()
                for lc, v in zip(['LOCATION1', 'LOCATION2'], ['location - 1 -', 'location - 2 -']):
                    if lc in text:
                        text = text.replace(lc, lc+' ')
                        loc[lc] = v

                aspect_with_loc_dist=[]
                # compute distance
                if len(loc)>1:

                    text=  re.sub(r'[^\w\s]', ' ', text.strip().lower()).split()

                    for aspect in categories:
                        if aspect not in  category_representatives:
                            continue
                        aux = category_representatives.get(aspect)
                        aux_loc={lc:[] for lc in loc.keys() }
                        for w in aux:
                            if w not in text:
                                continue
                            for lc in loc.keys():
                                aux_loc[lc].append(abs(text.index(w)- text.index(lc.lower())))
                        if not len(aux_loc['LOCATION1']) :
                            continue
                        if np.mean(aux_loc['LOCATION1']) <np.mean(aux_loc['LOCATION2']) :
                            aspect_with_loc_dist.append(('LOCATION1', aspect))
                        else:
                            aspect_with_loc_dist.append(('LOCATION2', aspect))
                for lc,v in loc.items():
                    for aspect in categories:

                        if aspect in category_representatives and aspect_polarity.get(lc+'#'+aspect) in ['Positive', 'Negative']:

                            if (lc, aspect) in aspect_with_loc_dist or len(loc)==1:
                                aux = category_representatives.get(aspect)
                            else:
                                # aux=[aspect]
                                aux=[]
                            aux = list(set(aux))
                            temp={'category':aspect, 'polarity':aspect_polarity.get(lc+'#'+aspect), 'auxiliary':aux, 'target':v}
                        else:
                            temp = {'category': aspect, 'polarity': 'None','auxiliary': [] , 'target':v}
                        asp_tem.append(temp)
                current_data['aspect']=asp_tem
                data_to_save.append(current_data)

        with open('../datasets/sentihood/bert_{}.json'.format(phase), 'w') as f :
            json.dump(data_to_save, f, indent=3)
        f.close()





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval', type=str, help='semeval, sentihood', required=True)

    opt = parser.parse_args()
    if opt.dataset =='semeval':
        semeval()
    else:
        sentihood()


if __name__ == '__main__':
    main()
