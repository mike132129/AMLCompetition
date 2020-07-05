from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
import pdb
from torch.autograd import Variable

class modified_bert_for_class(nn.Module):
    def __init__(self, model):
        super(modified_bert_for_class, self).__init__()
        self.model = model
        self.cls_linear_1 = nn.Linear(768, 300)
        self.cls_linear_2 = nn.Linear(300, 1)
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):

        bert_output = self.model(input_ids=input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids)
        pool_output = self.dropout_1(bert_output[1])
        cls_logits = self.cls_linear_1(pool_output)
        cls_logits = self.dropout_2(cls_logits)
        cls_logits = self.cls_linear_2(cls_logits)
        cls_logits = cls_logits.squeeze(-1)

        if labels == None: # when predicting
            return cls_logits

        pos_weight = torch.FloatTensor([1.5]).to(torch.device('cuda:0'))
        criterion = BCEWithLogitsLoss(pos_weight=pos_weight) # proportion of positive sample is 7

        cls_loss = criterion(cls_logits.view(-1, 1), labels.view(-1, 1))

        output = (cls_loss, cls_logits) 
        return output

class modified_bert_for_tag(nn.Module):
    def __init__(self, model):
        super(modified_bert_for_tag, self).__init__()
        self.model = model
        self.name_linear1 = nn.Linear(768, 200)
        self.name_linear2 = nn.Linear(200, 2)
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):

        bert_output = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        sequence_output = bert_output[0]
        name_logits = self.dropout_1(sequence_output)
        name_logits = self.name_linear1(name_logits)
        name_logits = self.dropout_2(name_logits)
        name_logits = self.name_linear2(name_logits)

        start_logits, end_logits = name_logits.split(1, dim=-1)

        # Squeeze
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if labels == None: # when predicting
            return name_logits

        start_target = labels[0]
        end_target = labels[1]
        
        pos_wgt = torch.FloatTensor([200]).to(torch.device('cuda:0'))
        loss_fct = BCEWithLogitsLoss(pos_weight=pos_wgt)

        start_loss = loss_fct(start_logits, start_target.float())
        end_loss = loss_fct(end_logits, end_target.float())

        output = ((start_loss, end_loss), name_logits)
        return output
