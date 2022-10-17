
import logging
import argparse
import os
import sys
import random
import numpy
from transformers import AdamW
from torch.utils.data.sampler import  WeightedRandomSampler
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from data_utils import ABSATokenizer, ABSADataset_absa_bert_semeval_json, ABSADataset_absa_bert_sentihood_json
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
            self.trainset = ABSADataset_absa_bert_semeval_json(opt.dataset_file['train'], tokenizer, opt)
            self.testset = ABSADataset_absa_bert_semeval_json(opt.dataset_file['test'], tokenizer, opt)
            assert 0 <= opt.valset_ratio < 1
            if opt.valset_ratio > 0:
                valset_len = int(len(self.trainset) * opt.valset_ratio)
                self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
            else:
                self.valset = self.testset
        else:
            self.trainset = ABSADataset_absa_bert_sentihood_json(opt.dataset_file['train'], tokenizer, opt)
            self.testset = ABSADataset_absa_bert_sentihood_json(opt.dataset_file['test'], tokenizer, opt)
            self.valset =self.testset
            


        logger.info('train {0}: dev {1}: test {2}'.format(len(self.trainset), len(self.valset), len(self.testset)))
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


    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def _train(self, optimizer, train_data_loader, val_data_loader, t_total):
        max_val_f1 = 0
        global_step = 0
        loss_total= 0
        setp_total= 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))

            self.model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                optimizer.zero_grad()
                sample_batched= [b.to(self.opt.device) for b in sample_batched]
                input_ids, token_type_ids, attention_mask, labels= sample_batched
                loss= self.model(input_ids, token_type_ids, attention_mask, labels)
                loss.backward()
                with torch.no_grad():
                    loss_total+= loss.item()
                    setp_total+=len(labels)
                lr_this_step = self.opt.learning_rate * self.warmup_linear(global_step / t_total, self.opt.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

                optimizer.step()
                global_step += 1


                if global_step % self.opt.log_step == 0:
                    y_true, y_pred, score = self._evaluate_acc_f1(val_data_loader)
                    if self.opt.dataset  == 'semeval':
                        aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
                        sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
                        sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
                        sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
                        max_per= sentiment_Acc_4_classes
                    else:
                        aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
                        aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
                        aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
                        max_per=aspect_strict_Acc

                    if max_per > max_val_f1:
                        max_val_f1 = max_per
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')

                        path = copy.deepcopy(self.model.state_dict())
                        if self.opt.dataset == 'semeval':
                            logger.info('')
                            logger.info('>> P: {:.4f} , R: {:.4f} , F: {:.4f} '.format(aspect_P, aspect_R, aspect_F))
                            logger.info('>> 2 classes acc: {:.4f} '.format(sentiment_Acc_2_classes))
                            logger.info('>> 3 classes acc: {:.4f} '.format(sentiment_Acc_3_classes))
                            logger.info('>> 4 classes acc: {:.4f} '.format(sentiment_Acc_4_classes))
                        else:
                            logger.info('')
                            logger.info('>> aspect_strict_Acc: {:.4f} , aspect_Macro_F1: {:.4f} , aspect_Macro_AUC: {:.4f} '.format(aspect_strict_Acc, aspect_Macro_F1, aspect_Macro_AUC))
                            logger.info('>> sentiment_Acc: {:.4f} '.format(sentiment_Acc))
                            logger.info('>> sentiment_Macro_AUC: {:.4f} '.format(sentiment_Macro_AUC))
                    self.model.train()
            logger.info(" epoch : {0}, training loss: {1} ".format(str(epoch), loss_total/setp_total  ))

        return path

    def _evaluate_acc_f1_(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['label'].to(self.opt.device)

                _, t_outputs,_,_ = self.model(t_inputs)


                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],
                              average='macro')
        return acc, f1

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        score = []
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):

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

    def make_weights_for_balanced_classes(self, labels, nclasses, fixed=False):
        if fixed:
            weight = [0] * len(labels)
            if nclasses == 3:
                for idx, val in enumerate(labels):
                    if val == 0:
                        weight[idx] = 0.2
                    elif val == 1:
                        weight[idx] = 0.4
                    elif val == 2:
                        weight[idx] = 0.4
                return weight
            else:
                for idx, val in enumerate(labels):
                    if val == 0:
                        weight[idx] = 0.2
                    else:
                        weight[idx] = 0.4
                return weight
        else:
            count = [0] * nclasses
            for item in labels:
                count[item] += 1
            weight_per_class = [0.] * nclasses
            N = float(sum(count))
            for i in range(nclasses):
                weight_per_class[i] = N / float(count[i])
            weight = [0] * len(labels)
            for idx, val in enumerate(labels):
                weight[idx] = weight_per_class[val]
            return weight
    def run(self):


        all_label_ids= torch.tensor([f['label'] for f in self.trainset], dtype=torch.long)
        self.trainset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.trainset], dtype=torch.long), torch.tensor([f['bert_segments_ids'] for f in self.trainset], dtype=torch.long), torch.tensor([f['input_mask'] for f in self.trainset], dtype=torch.long), all_label_ids)

        if self.opt.dataset == "semeval":
            sampler_weights = self.make_weights_for_balanced_classes(all_label_ids, 5)
        else:
            sampler_weights = self.make_weights_for_balanced_classes(all_label_ids, 3)
        train_sampler = WeightedRandomSampler(sampler_weights, len(self.trainset), replacement=True)


        
        train_data_loader= DataLoader(dataset=self.trainset, batch_size=self.opt.train_batch_size,sampler=train_sampler)

        self.testset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['bert_segments_ids'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['input_mask'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['label'] for f in self.testset], dtype=torch.long))

        self.valset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.valset], dtype=torch.long),
                                     torch.tensor([f['bert_segments_ids'] for f in self.valset], dtype=torch.long),
                                     torch.tensor([f['input_mask'] for f in self.valset], dtype=torch.long),
                                     torch.tensor([f['label'] for f in self.valset], dtype=torch.long))

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.eval_batch_size, shuffle=False)

        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.eval_batch_size, shuffle=False)

        num_train_steps = int(len(train_data_loader) * self.opt.num_epoch)
        

        
        t_total = num_train_steps
        optimizer= self.opt.optimizer(self.model.parameters(), lr=self.opt.learning_rate,
                                            weight_decay=self.opt.l2reg)
       
        best_model_path = self._train(optimizer, train_data_loader, val_data_loader, t_total)
        self.model.load_state_dict(best_model_path)
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

        
        if self.opt.save_model:
            path = 'state_dict/{0}/{0}.bm'.format( self.opt.dataset, str(self.opt.seed))
            torch.save(self.model.state_dict(), path)





def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval',  choices=['semeval','sentihood'], type=str, required=True)
    parser.add_argument('--learning-rate', default=3e-5, type=float, help='try 5e-5, 2e-5')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.001, type=float)
    parser.add_argument('--warmup-proportion', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=5, type=int, help='')
    parser.add_argument("--train-batch-size", default=32,type=int, help="Total batch size for training.")
    parser.add_argument("--eval-batch-size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument('--log-step', default=50, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=120, type=int)
    parser.add_argument('--lebel-dim', default=5, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--pt-model', default='activebus/BERT-PT_rest', type=str)
    parser.add_argument('--save_model', default=0, type=int)
    parser.add_argument('--device', default='cuda:5', type=str, help='e.g. cuda:0')
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

    logger.info(opt.pt_model)
    opt.optimizer = AdamW
    opt.model_class = ABSATokenizer
    opt.dataset_file = dataset_files
    opt.inputs_cols = ['text_bert_indices', 'bert_segments_ids', 'input_mask', 'label']
    opt.initializer = torch.nn.init.xavier_uniform_
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':

    main()
