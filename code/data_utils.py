
import json
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import unicodedata
import six




class ABSADataset_absa_bert_semeval_json(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt
        all_data = []
        with open(fname) as f :
            dataa= json.load(f)
        f.close()
        for i in tqdm(range(len(dataa))):
            text = dataa[i]['text']
            aspects = dataa[i]['aspect']
            text_=text

            for current_aspect in aspects:
                category = current_aspect['category']
                auxiliary = current_aspect['auxiliary']
                opinions = current_aspect['opinions']

                auxiliary.append(category)

                auxiliary.extend(opinions)

                auxiliary= list(set(auxiliary))

                auxiliary=self.sort_auxiliary(text_, auxiliary)

                label = current_aspect['polarity']
                label = {a: _ for _, a in enumerate(['positive', 'neutral', 'negative', 'conflict', 'none'])}.get(label)

              
                auxiliary ='what is the sentiment of ' + auxiliary
                example = tokenizer.encode_plus(text, auxiliary,add_special_tokens=True,  truncation = True,   padding = 'max_length',max_length=self.opt.max_seq_len,  return_token_type_ids=True)
                data = {
                    'text': text_,
                    'text_bert_indices': np.asarray(example['input_ids'], dtype='int64'),
                    'bert_segments_ids': np.asarray(example['token_type_ids'], dtype='int64'),
                    'input_mask': np.asarray(example['attention_mask'], dtype='int64'),
                    'label': label,
                }

                all_data.append(data)


        self.data = all_data
    def sort_auxiliary(self, text_a, text_b):

        text_a = text_a.split()
        arr = [text_a.index(w) if w in text_a else len(text_b) for w in text_b]
        arr = sorted(arr)
        return ' '.join([text_a[k] if k !=  len(text_b) else ' '.join(set(text_b).difference(set(text_a))) for k in arr])
  
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
class ABSADataset_absa_bert_sentihood_json(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt
        all_data = []
        # dataa = pd.read_csv(fname)
        with open(fname) as f :
            dataa= json.load(f)
        f.close()
        for i in tqdm(range(len(dataa))):

            text = dataa[i]['text']
            aspects = dataa[i]['aspect']
            text_= text
         

            for current_aspect in aspects:
                category = current_aspect['category']
                target = current_aspect['target']
                auxiliary = current_aspect['auxiliary']
                opinions = current_aspect['opinions']
                auxiliary.extend(opinions)
                auxiliary= self.sort_auxiliary(text_,auxiliary)



                auxiliary=target + ' ' + category + ' ' + ' '.join(auxiliary)

                label = current_aspect['polarity']

                label = {a: _ for _, a in enumerate(['None', 'Positive', 'Negative'])}.get(label)

                assert  label != None

                auxiliary = 'what is the sentiment of ' + auxiliary
                example = tokenizer.encode_plus(text, auxiliary, add_special_tokens=True, truncation=True,
                                                padding='max_length', max_length=self.opt.max_seq_len,
                                                return_token_type_ids=True)
                data = {
                    'text_bert_indices': np.asarray(example['input_ids'], dtype='int64'),
                    'bert_segments_ids': np.asarray(example['token_type_ids'], dtype='int64'),
                    'input_mask': np.asarray(example['attention_mask'], dtype='int64'),
                    'label': label,
                }

                all_data.append(data)


        self.data = all_data

    def sort_auxiliary(self, text_a, text_b):

        text_a = text_a.split()
        arr = [text_a.index(w) if w in text_a else len(text_b) for w in text_b]
        arr = sorted(arr)
        return ' '.join([text_a[k] if k !=  len(text_b) else ' '.join(set(text_b).difference(set(text_a))) for k in arr])
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
