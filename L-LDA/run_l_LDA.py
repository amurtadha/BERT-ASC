import random
import sys
import pickle as pk
sys.path.append('../')
import model.labeled_lda as llda
import argparse
from nltk.corpus import stopwords
stops = stopwords.words('english')
stops.extend(['us', "didnt",'im', 'make','today', 'feel', 'sometimes', 'ive' 'whatever','never','although','anyway','get','always', 'usually', 'want', 'go','also','would', 'one', 'theres'])

def train(dataset):
    n_iter= 400

    labeled_documents_train  = pk.load(open('../datasets/{}/train.pk'.format(dataset), 'rb'))
    labeled_documents_train = [(v, k.split()) for k, v in labeled_documents_train.items()]

    labeled_documents_test = pk.load(open('../datasets/{}/test.pk'.format(dataset), 'rb'))
    labeled_documents_test = [(v, k.split()) for k, v in labeled_documents_test.items()]


    llda_model = llda.LldaModel(labeled_documents=labeled_documents_train, alpha_vector=0.01)
    print(llda_model)


    while True:
        print("iteration %s sampling..." % (llda_model.iteration + 1))
        llda_model.training(1)
        print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
        print("delta beta: %s" % llda_model.delta_beta)
        if llda_model.iteration > n_iter:
        # if llda_model.is_convergent(method="beta", delta=0.01):
            break

    # update
    print("before updating: ", llda_model)
    update_labeled_documents =random.sample(labeled_documents_test, k=2)
    llda_model.update(labeled_documents=update_labeled_documents)
    print("after updating: ", llda_model)

    # train again
    # llda_model.training(iteration=10, log=True)
    while True:
        print("iteration %s sampling..." % (llda_model.iteration + 1))
        llda_model.training(1)
        print("after 1 iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
        print("delta beta: %s" % llda_model.delta_beta)
        # if llda_model.is_convergent(method="beta", delta=0.01):
        if llda_model.iteration >n_iter*2:
            break
    save_model_dir = "../datasets/{}".format(dataset)
    # llda_model.save_model_to_dir(save_model_dir, save_derivative_properties=True)
    llda_model.save_model_to_dir(save_model_dir)
def inference(dataset, n = 20):
    save_model_dir  = "../datasets/{}".format(dataset)
    llda_model = llda.LldaModel()
    llda_model.load_model_from_dir(save_model_dir, load_derivative_properties=False)

    labeled_documents_test = pk.load(open('../datasets/{}/test.pk'.format(dataset), 'rb'))
    labeled_documents_test = [(v, k.split()) for k, v in labeled_documents_test.items()]

    document=random.sample(labeled_documents_test, k=1)[0][0]
    # print(document)
    topics = llda_model.inference(document=document, iteration=100, times=10)
    # print(topics)
    categories= pk.load(open('../datasets/{}/categories.pk'.format(dataset), 'rb'))
    for c in categories:
        if c == 'anecdotes':
            continue
        print("Top-15 terms of topic : ",c,  llda_model.top_terms_of_topic(c, n, False))
    categories_seed={}
    for c in categories:
        if c=='anecdotes':
            continue
        seeds=llda_model.top_terms_of_topic(c, n, False)
        seeds=[s for s in seeds if s not in categories ]
        seeds.append(c)
        # if c in uae:
        #     seeds.extend(uae.get(c))
        seeds=list(set(seeds).difference(set(stops)))
        categories_seed[c]=seeds
    pk.dump(categories_seed, open('../datasets/{}/categories_seeds.pk'.format(dataset), 'wb'))

def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval', type=str, help='semeval, sentihood', required=True)
    opt = parser.parse_args()
    train(opt.dataset)
    inference(opt.dataset)
if __name__ == '__main__':
    main()
