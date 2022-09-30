import random
import sys
import pickle as pk
sys.path.append('../')
import model.labeled_lda as llda
def train():
    # initialize data
    labeled_documents_train = [("example example example example example"*10, ["example"]),
                         ("test llda model test llda model test llda model"*10, ["test", "llda_model"]),
                         ("example test example test example test example test"*10, ["example", "test"]),
                         ("good perfect good good perfect good good perfect good "*10, ["positive"]),
                         ("bad bad down down bad bad down"*10, ["negative"])]


    labeled_documents_train  = pk.load(open('data/semeval/train.pk', 'rb'))
    labeled_documents_test  = pk.load(open('data/semeval/test.pk', 'rb'))
    labeled_documents_train=[(v,k.split()) for k,v in labeled_documents_train.items()]
    labeled_documents_test=[(v,k.split()) for k,v in labeled_documents_test.items()]
    
    llda_model = llda.LldaModel(labeled_documents=labeled_documents_train, alpha_vector=0.01)
    print(llda_model)

    # training
    # llda_model.training(iteration=10, log=True)
    while True:
        print("iteration %s sampling..." % (llda_model.iteration + 1))
        llda_model.training(1)
        print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
        print("delta beta: %s" % llda_model.delta_beta)
        if llda_model.iteration > 600:
        # if llda_model.is_convergent(method="beta", delta=0.01):
            break

    # update
    print("before updating: ", llda_model)
    update_labeled_documents =random.sample(labeled_documents_test, k=5)
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
        if llda_model.iteration >1200:
            break

    # inference
    # note: the result topics may be different for difference training, because gibbs sampling is a random algorithm
    document = "example llda model example example good perfect good perfect good perfect" * 100
    document=random.sample(labeled_documents_test, k=1)[0][0]
    print(document)
    topics = llda_model.inference(document=document, iteration=100, times=10)
    print(topics)

   
    save_model_dir = "data/model"
    llda_model.save_model_to_dir(save_model_dir)

    # load from disk
    llda_model_new = llda.LldaModel()
    llda_model_new.load_model_from_dir(save_model_dir, load_derivative_properties=False)
   
    print("Top-5 terms of topic 'food': ", llda_model.top_terms_of_topic("food", 10, False))
    print("Top-5 terms of topic 'price': ", llda_model.top_terms_of_topic("price", 10, False))
    print("Top-5 terms of topic 'ambience': ", llda_model.top_terms_of_topic("ambience",10, False))
    print("Top-5 terms of topic 'service': ", llda_model.top_terms_of_topic("service", 10, False))
    print("Top-5 terms of topic 'anecdotes/miscellaneous': ", llda_model.top_terms_of_topic("anecdotes/miscellaneous", 10, False))
    # print("Doc-Topic Matrix: \n", llda_model.theta)
    # print("Topic-Term Matrix: \n", llda_model.beta)
def inference():
   
    save_model_dir = "data/model"

    # load from disk
    llda_model = llda.LldaModel()
    llda_model.load_model_from_dir(save_model_dir)
   
    n= 15
    print("Top-5 terms of topic 'food': ", llda_model.top_terms_of_topic("food", n, False))
    print("Top-5 terms of topic 'price': ", llda_model.top_terms_of_topic("price", n, False))
    print("Top-5 terms of topic 'ambience': ", llda_model.top_terms_of_topic("ambience", n, False))
    print("Top-5 terms of topic 'service': ", llda_model.top_terms_of_topic("service", n, False))
    print("Top-5 terms of topic 'anecdotes/miscellaneous': ",
          llda_model.top_terms_of_topic("anecdotes/miscellaneous", n, False))


if __name__ == '__main__':
    # train()
    inference()
