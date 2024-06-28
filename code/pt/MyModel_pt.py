import torch
import torch.nn as nn

from layers.modeling_custom import BertModel, BertPreTrainedModel, BertLayer, BertPooler

class GRoIE(nn.Module):
    def __init__(self, count, config, num_labels):
        super(GRoIE, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pooler = BertPooler(config)
        self.pre_layers = torch.nn.ModuleList()
        self.loss_fct = torch.nn.ModuleList()
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        for i in range(count):
            self.pre_layers.append(BertLayer(config))
            self.loss_fct.append(torch.nn.CrossEntropyLoss(ignore_index=-1))

    def forward(self, layers, attention_mask, labels):
        losses = []
        logitses = []
        for i in range(self.count):
            layer = self.pre_layers[i](layers[-i-1], attention_mask)
            layer = self.pooler(layer)
            logits = self.classifier(layer)
            if labels is not None:
                loss = self.loss_fct[i](logits.view(-1, self.num_labels), labels.view(-1))
                losses.append(loss)
            logitses.append(logits)
        if labels is not None:
            total_loss = torch.sum(torch.stack(losses), dim=0)
        else:
            total_loss = torch.Tensor(0)
        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return total_loss, avg_logits


class BERT_ASC(BertPreTrainedModel):
    def __init__(self, config, num_labels=3):
        super(BERT_ASC, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=True)
        # self.groie = GRoIE(4, config, num_labels)
        self.groie = GRoIE(4, config, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, return_attention=False):
        attention, layers, _, mask = self.bert(input_ids, token_type_ids,
                                                        attention_mask=attention_mask,
                                                        output_all_encoded_layers=True)
        loss, logits = self.groie(layers, mask, labels)
        if labels is not None:
            return loss
        else:
            if return_attention:
                return attention, logits
            else:
                return logits
