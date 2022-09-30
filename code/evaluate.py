import json
import logging
import argparse

import os
import sys
import random
import numpy
import pandas as pd

from torch.utils.data.sampler import  WeightedRandomSampler
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from data_utils import ABSATokenizer, ABSADataset_absa_bert_semeval_json, ABSADataset_absa_bert_sentihood_json
from layers.optimization  import BertAdam
from evaluation import *
import torch.nn.functional as F
import  numpy as np
import copy
from  tqdm import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


from MyModel import BERT_ASC

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        self.model = BERT_ASC.from_pretrained(opt.pt_model, num_labels=opt.lebel_dim)
        self.model.to(self.opt.device)

        tokenizer = ABSATokenizer.from_pretrained(opt.pt_model)
        if self.opt.dataset=='semeval':
            self.testset = ABSADataset_absa_bert_semeval_json(opt.dataset_file['test'], tokenizer, opt)
        else:
            self.testset = ABSADataset_absa_bert_sentihood_json(opt.dataset_file['test'], tokenizer, opt)

        logger.info(' test {}'.format( len(self.testset)))
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))



    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        score = []
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):

                t_sample_batched = [b.to(self.opt.device) for b in t_sample_batched]
                input_ids, token_type_ids, attention_mask, labels = t_sample_batched

                logits = self.model(input_ids, token_type_ids, attention_mask, labels=None)
                score.append(F.softmax(logits, dim=-1).detach().cpu().numpy())

                n_correct += (torch.argmax(logits, -1) == labels).sum().item()
                n_total += len(logits)

                if t_targets_all is None:
                    t_targets_all = labels
                    t_outputs_all = logits
                else:
                    t_targets_all = torch.cat((t_targets_all, labels), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, logits), dim=0)

        return t_targets_all.cpu().numpy(), torch.argmax(t_outputs_all, -1).cpu().numpy(), np.concatenate(score, axis=0)

    def run(self):

        testset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['bert_segments_ids'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['input_mask'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['label'] for f in self.testset], dtype=torch.long))


        test_data_loader = DataLoader(dataset=testset, batch_size=self.opt.eval_batch_size, shuffle=False)

        best_model_path = 'state_dict/{0}/{0}_.bm'.format(self.opt.dataset, str(self.opt.seed))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))
        self.model.eval()

        y_true, y_pred, score = self._evaluate_acc_f1(test_data_loader)

        if self.opt.dataset=='semeval':
            aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
            sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
            sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
            sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
            logger.info('>> P: {:.4f} , R: {:.4f} , F: {:.4f} '.format(aspect_P, aspect_R, aspect_F))
            logger.info('>> 4 classes acc: {:.4f} '.format(sentiment_Acc_4_classes))
            logger.info('>> 3 classes acc: {:.4f} '.format(sentiment_Acc_3_classes))
            logger.info('>> 2 classes acc: {:.4f} '.format(sentiment_Acc_2_classes))
        else:
            aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
            aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
            aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
            logger.info(())
            logger.info('>> aspect_strict_Acc: {:.4f} , aspect_Macro_F1: {:.4f} , aspect_Macro_AUC: {:.4f} '.format(aspect_strict_Acc, aspect_Macro_F1, aspect_Macro_AUC))
            logger.info('>> sentiment_Acc: {:.4f} '.format(sentiment_Acc))
            logger.info('>> sentiment_Macro_AUC: {:.4f} '.format(sentiment_Macro_AUC))









def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval', type=str,  choices=['semeval','sentihood'], help='semeval, sentihood', required=True)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='try 5e-5, 2e-5')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.001, type=float)
    parser.add_argument('--warmup_proportion', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=5, type=int, help='')
    parser.add_argument("--train_batch_size", default=32,type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument('--log_step', default=50, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=120, type=int)
    parser.add_argument('--lebel_dim', default=5, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--save_model', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    opt = parser.parse_args()

    if opt.dataset=='sentihood':
        opt.lebel_dim =3

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    dataset_files = {
        'train': '../datasets/{}/bert_train.json'.format( opt.dataset),
        'test': '../datasets/{}/bert_test.json'.format( opt.dataset),
        'val': '../datasets/{}/bert_dev.json'.format( opt.dataset)
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }



    opt.pt_model =r'plm/pt/'
    print(opt.pt_model)
    opt.model_class = ABSATokenizer
    opt.dataset_file = dataset_files
    opt.inputs_cols = ['text_bert_indices', 'bert_segments_ids', 'input_mask', 'label']
    opt.initializer = initializers[opt.initializer]

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':

    main()
