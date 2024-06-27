
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
            text = self.convert_to_unicode(text)
            text_=text
            text=tokenizer.tokenize(text.strip().lower())

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

                auxiliary= self.convert_to_unicode(auxiliary)

                # auxiliary =category
                auxiliary = tokenizer.tokenize('what is the sentiment of '+auxiliary.strip().lower())

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")

                segment_ids.append(1)
                for token in text:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                for token in auxiliary:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)


                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                input_ids = input_ids[:self.opt.max_seq_len]
                input_mask = input_mask[:self.opt.max_seq_len]
                segment_ids = segment_ids[:self.opt.max_seq_len]
                while len(input_ids) < self.opt.max_seq_len:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)


                input_ids = np.asarray(input_ids, dtype='int64')
                input_mask = np.asarray(input_mask, dtype='int64')
                segment_ids = np.asarray(segment_ids, dtype='int64')


                data = {
                    'text': text_,
                    'text_bert_indices': input_ids,
                    'bert_segments_ids': segment_ids,
                    'input_mask': input_mask,
                    'label': label,
                }

                all_data.append(data)


        self.data = all_data
    def sort_auxiliary(self, text_a, text_b):

        text_a = text_a.split()
        arr = [text_a.index(w) if w in text_a else len(text_b) for w in text_b]
        arr = sorted(arr)
        return ' '.join([text_a[k] if k !=  len(text_b) else ' '.join(set(text_b).difference(set(text_a))) for k in arr])
    def convert_to_unicode(self,text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")
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
            text = self.convert_to_unicode(text)
            text_= text
            text=tokenizer.tokenize(text.strip().lower())

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



                auxiliary= self.convert_to_unicode(auxiliary)
                auxiliary = tokenizer.tokenize(auxiliary.strip().lower())

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in auxiliary:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in text:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                input_ids = input_ids[:self.opt.max_seq_len]
                input_mask = input_mask[:self.opt.max_seq_len]
                segment_ids = segment_ids[:self.opt.max_seq_len]
                while len(input_ids) < self.opt.max_seq_len:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)


                input_ids = np.asarray(input_ids, dtype='int64')
                input_mask = np.asarray(input_mask, dtype='int64')
                segment_ids = np.asarray(segment_ids, dtype='int64')


                data = {
                    'text_bert_indices': input_ids,
                    'bert_segments_ids': segment_ids,
                    'input_mask': input_mask,
                    'label': label,
                }

                all_data.append(data)


        self.data = all_data

    def sort_auxiliary(self, text_a, text_b):

        text_a = text_a.split()
        arr = [text_a.index(w) if w in text_a else len(text_b) for w in text_b]
        arr = sorted(arr)
        return ' '.join([text_a[k] if k !=  len(text_b) else ' '.join(set(text_b).difference(set(text_a))) for k in arr])
    def convert_to_unicode(self,text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
