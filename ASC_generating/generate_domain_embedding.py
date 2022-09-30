
import gensim
import codecs

class Sentence(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    source = 'preprocessed_data/{}/train.txt,formate(domain)
    model_file = 'embeddings/{}.bin' % (domain)
    sentences = Sentence(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4)
    model.save(model_file)

if __name__ == '__main__':
    print ('Pre-training word embeddings ...')
    main('restaurant')
  
