import  json
import argparse
from nltk.corpus import stopwords
stops = stopwords.words('english')
stops.extend(['us', "didnt",'im','couldnt','even', 'shouldnt','ive','make','today', 'feel', 'sometimes', 'ive' 'whatever','never','although','anyway','get','always', 'usually', 'want', 'go','would', 'one'])
stops.extend(['location - 1 -', 'location - 2 -'])


from nltk.parse.corenlp import CoreNLPDependencyParser,CoreNLPParser
from tqdm import tqdm

def extract_opnion_words(dataset='semeval'):
    #
    from stanfordnlp.server import CoreNLPClient


    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'depparse'], timeout=60000, memory='16G') as client:
        files={'semeval':[ 'train', 'test'],'sentihood':[ 'train', 'test', 'dev'] }
        aspect_categories={'semeval':[ 'price', 'food', 'service', 'ambience'],'sentihood':[ 'price', 'test', 'dev'] }
        for subFile in files.get(dataset):
            with open('../datasets/{0}/bert_{1}_{2}.json'.format(dataset, subFile,'')) as f :
                data= json.load(f)
                new_data=[]
                for d in tqdm(data):
                    text, aspects = d.get('text'), d.get('aspect')
                    ann = client.annotate(text)
                    dp_rel = get_parse(ann)

                    for i in range(len(aspects)):
                        current_aspect = aspects[i]
                        auxiliary= current_aspect['auxiliary'].copy()
                        if current_aspect['category'] in text:
                            auxiliary.append(aspects[i]['category'])
                        opnions = []
                        for w in auxiliary:
                            if w not in text or w in opnions:
                                continue
                            # extract its modifiers
                            for rel in dp_rel:
                                l, m, r = rel
                                candidates = [l, r]
                                if m in ['nsubj', 'amod', 'advmod', 'ccomp', 'compound'] and w in candidates:
                                    del candidates[candidates.index(w)]
                                    opnions.extend(candidates)

                        opnions= list(set(opnions).difference(set(stops+auxiliary+aspect_categories.get(dataset))))
                        opnions= sort_auxiliary(text, opnions)
                        aspects[i]['opinions']= opnions

                    new_data.append({'text': text, 'aspect':aspects})
                f.close()
                with open('../../datasets/{0}/bert_{1}.json'.format(dataset, subFile), 'w') as f:
                    json.dump(new_data, f, indent=3)
                f.close()
def extract_opnion_words_sentihood(dataset='sentihood'):
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    files={'semeval':[ 'train', 'test'], 'sentihood':[ 'train', 'test', 'dev']}

    for subFile in files.get(dataset):
        with open('../datasets/{0}/bert_{1}.json'.format(dataset, subFile)) as f :
            data= json.load(f)
            new_data=[]
            for d in tqdm(data):
                text, aspects = d.get('text'), d.get('aspect')
                dp_tree, = parser.raw_parse(text)
                dp_rel = list(dp_tree.triples())




                for i in range(len(aspects)):
                    opnions = []
                    current_aspect = aspects[i]
                    auxiliary= current_aspect['auxiliary'].copy()
                    if current_aspect['category'] in text:
                        auxiliary.append(aspects[i]['category'])
                    # auxiliary= set(auxiliary)
                    opnions = []
                    for w in auxiliary:
                        if w not in text or w in opnions:
                            continue
                        # print(w)
                        # extract its modifiers
                        for rel in dp_rel:
                            l, m, r = rel
                            candidates = [l[0], r[0]]
                            if m in ['nsubj', 'amod', 'advmod', 'ccomp', 'compound'] and w in candidates:
                                del candidates[candidates.index(w)]
                                opnions.extend(candidates)
                    opnions= list(set(opnions).difference(set(stops)))
                    opnions= sort_auxiliary(text, opnions)
                    aspects[i]['opinions']= opnions
                new_data.append({'text': text, 'aspect':aspects})
            f.close()
            with open('../../datasets/{0}/bert_{1}.json'.format(dataset, subFile), 'w') as f:
                json.dump(new_data, f, indent=3)
            f.close()

def sort_auxiliary( text_a, text_b):
        # text_b=[w for w in text_b if len(w)>2]
        text_a = text_a.split()
        arr = [text_a.index(w) if w in text_a else len(text_a) for w in text_b]
        arr = sorted(arr)
        return [text_a[k] if k !=  len(text_a) else ' '.join(set(text_b).difference(set(text_a))) for k in arr]


def get_parse(ann):

    sentence = ann.sentence[0]

    # get the dependency parse of the first sentence
    dependency_parse = sentence.enhancedDependencies
    # get a dictionary associating each token/node with its label
    token_dict = {}
    for i in range(0, len(sentence.token)):
        token_dict[sentence.token[i].tokenEndIndex] = sentence.token[i].word

    # get a list of the dependencies with the words they connect
    list_dep = []
    for i in range(0, len(dependency_parse.edge)):
        source_node = dependency_parse.edge[i].source
        source_name = token_dict[source_node]

        target_node = dependency_parse.edge[i].target
        target_name = token_dict[target_node]

        dep = dependency_parse.edge[i].dep

        list_dep.append((source_name, dep, target_name))
    return list_dep



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval', type=str, help='semeval, sentihood', required=True)

    opt = parser.parse_args()
    extract_opnion_words(dataset=opt.dataset)

if __name__ == '__main__':
    main()




