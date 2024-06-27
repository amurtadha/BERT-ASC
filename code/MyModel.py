import torch
import torch.nn as nn
from transformers import   AutoConfig, AutoModelForSequenceClassification



class BERT_ASC_vanila(nn.Module):

    def __init__(self, args, hidden_size=256):
        super(BERT_ASC_vanila, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        config.num_labels = args.lebel_dim
        self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.to('cuda')
        # layers = [nn.Linear(config.hidden_size, hidden_size), nn.ReLU(), nn.Dropout(.3),
        #           nn.Linear(hidden_size, args.lebel_dim)]

        # layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        # self.classifier = nn.Sequential(*layers)

    def forward(self,  input_ids, token_type_ids=None, attention_mask=None, labels=None, return_attention=False):
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        return outputs['logits']
