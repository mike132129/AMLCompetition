import numpy as np
import pdb
from argparse import ArgumentParser
from dataset import AMLDataset_For_BCL, pad_for_BCL, AMLDataset_For_Tag_Name, pad_for_Tag_Name
from dataset import get_split, create_token_type, create_mask
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from module import modified_bert_for_class, modified_bert_for_tag
import torch
import logging
import json
from keras.preprocessing.sequence import pad_sequences

torch.manual_seed(11320)
logging.basicConfig(level=logging.INFO)

def parse():
    parser = ArgumentParser(description="train")
    parser.add_argument('--train_binary_class', action='store_true', default=False)
    parser.add_argument('--predict_binary_class', action='store_true', default=False)
    parser.add_argument('--bert', action='store_true', default=False)
    parser.add_argument('--ernie', action='store_true', default=False)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--train_tag_name', action='store_true', default=False)
    parser.add_argument('--predict_competition', action='store_true', default=False)
    args = parser.parse_args()
    return args

def train_binary_classification(dataset, model, class_model):
    print('total {} data'.format(len(dataset)))
    trainset, validset = torch.utils.data.random_split(dataset, [3000, 806])

    trainloader = DataLoader(trainset, batch_size=4, collate_fn=pad_for_BCL, shuffle=True)
    validloader = DataLoader(validset, batch_size=30, collate_fn=pad_for_BCL, shuffle=False)
    epochs = 30
    total_steps = len(trainloader) * epochs
    optimizer = AdamW(class_model.parameters(), lr=5e-6, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    for epoch in range(epochs):

        model.train()
        class_model.train()
        total_loss = 0.0

        for step, data in tqdm(enumerate(trainloader)):
            if step % 300 == 0 and not step == 0:
                print('BCELoss: {}'.format(total_loss/step))

            tensors = [t.to(torch.device('cuda:0')) for t in data if t is not None]
            input_ids, token_type_ids, attention_mask, labels = tensors[0], tensors[1], tensors[2], tensors[3]
            
            loss, logits = class_model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=labels.float()
                                    )
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(class_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            class_model.zero_grad()
            model.zero_grad()

        model.eval()
        class_model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0.0
            predict = []
            true = []

            for step, data in enumerate(tqdm(validloader)):

                tensors = [t.to(torch.device('cuda:0')) for t in data if t is not None]
                input_ids, token_type_ids, attention_mask, labels = tensors[0], tensors[1], tensors[2], tensors[3]
                loss, logits = class_model(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        labels=labels.float()
                                        )
                total_loss += loss.item()
                prediction = logits.sigmoid().round().cpu()
                correct += (prediction == labels.cpu().detach()).sum().item()
                total += prediction.size(0)

            print('Epoch: {}, Accuracy: {}'.format(epoch, correct/total))

        torch.save(class_model.state_dict(), './model/bert-for-classification-epoch-0702-%s.bin' % epoch)

def predict_binary_classification(dataset, model, class_model, tokenizer):
    testloader = DataLoader(dataset, batch_size=30, collate_fn=pad, shuffle=False)

    class_model.load_state_dict(torch.load(args.load_model, map_location=lambda storage, loc: storage))
    correct = 0
    total = 0
    inaccurate = []
    with torch.no_grad():
        for data in tqdm(testloader):
            tensors = [t.to(torch.device('cuda:0')) for t in data if t is not None]
            input_ids, token_type_ids, attention_mask, labels = tensors[0], tensors[1], tensors[2], tensors[3]
            loss, logits = class_model(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        labels=labels.float()
                                        )
            
            prediction = logits.sigmoid().round().cpu()
            logging.debug(prediction)
            correct += (prediction == labels.cpu().detach()).sum().item()
            total += prediction.size(0)

        print(' Accuracy: {}'.format(correct/total))

def train_tag_criminal_name(dataset, model, tag_model):
    print(len(dataset))
    trainset, validset = torch.utils.data.random_split(dataset, [440, 101])

    # tag_model.load_state_dict(torch.load('./model/bert-for-tagging-epoch-0704-4.bin', map_location=lambda storage, loc: storage))

    trainloader = DataLoader(trainset, batch_size=4, collate_fn=pad_for_Tag_Name, shuffle=True)
    validloader = DataLoader(validset, batch_size=30, collate_fn=pad_for_Tag_Name, shuffle=False)
    epochs = 20
    total_steps = len(trainloader) * epochs
    optimizer = AdamW(tag_model.parameters(), lr=5e-6, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    for epoch in range(epochs):

        model.train()
        tag_model.train()
        total_loss = 0.0

        for step, data in tqdm(enumerate(trainloader)):
            if step % 20 == 0 and not step == 0:
                print('BCELoss: {}'.format(total_loss/step))

            tensors = [t.to(torch.device('cuda:0')) for t in data if t is not None]
            input_ids, token_type_ids, attention_mask, start_target, end_target = tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]
            labels = (start_target, end_target)
            loss, logits = tag_model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                                    
            loss = loss[0] + loss[1]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tag_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tag_model.zero_grad()
            model.zero_grad()

        model.eval()
        tag_model.eval()

        with torch.no_grad():

            total_loss = 0.0

            for step, data in enumerate(tqdm(validloader)):

                tensors = [t.to(torch.device('cuda:0')) for t in data if t is not None]
                input_ids, token_type_ids, attention_mask, start_target, end_target = tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]
                labels = (start_target, end_target)
                loss, logits = tag_model(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        labels=labels
                                        )

                loss = loss[0] + loss[1]

                total_loss += loss.item()

            print('epoch: {}, Total BCE loss: {}'.format(epoch, total_loss/step))
        torch.save(tag_model.state_dict(), './model/bert-for-tagging-epoch-0707-%s.bin' % epoch)

def predict_for_competition(class_model, tag_model, tokenizer, device):
    with open('./data/test.json', 'r') as file:
        data = json.load(file)
    esun_uuid = data['esun_uuid']
    server_uuid = data['server_uuid']
    esun_timestamp = data['esun_timestamp']
    news = data['news']

    def preprocess(news, bi_cls):
        token_type_ids = []
        attention_mask = []
        input_ids = []
        news_contents = get_split(news)

        for news_content in news_contents:
            if bi_cls:
                news_content = '[CLS]金錢[SEP]犯罪[SEP]' + news_content + '[SEP]'
            else:
                news_content = '[CLS]金錢[SEP]犯人[SEP]' + news_content + '[SEP]'
            tokens = tokenizer.tokenize(news_content)
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            token_type_id = create_token_type(input_id)
            att_mask = create_mask(input_id)

            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            attention_mask.append(att_mask)

        f = lambda x: pad_sequences(x, maxlen=512, dtype='long', truncating='post', padding='post')
        
        input_ids = f(input_ids)
        token_type_ids = f(token_type_ids)
        attention_mask = f(attention_mask)

        return input_ids, token_type_ids, attention_mask

    input_ids, token_type_ids, attention_mask = preprocess(news, True)

    ml_prob = []
    logging.info('Predict Probability of being Money Laundary News..')
    for input_id, token_type_id, att_mask in zip(input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            f = torch.tensor
            input_id = f(input_id).to(device)
            token_type_id = f(token_type_id).to(device)
            att_mask = f(att_mask).to(device)
            logits = class_model(input_ids=input_id.view(1, -1),
                                token_type_ids=token_type_id.view(1, -1),
                                attention_mask=att_mask.view(1, -1),
                                labels=None
                                )
            ml_prob.append(logits.sigmoid()[0].item())

    logging.info('predict probability: {}'.format(ml_prob))

    # money laundary threshold 
    if max(ml_prob) < 0.6:
        return []

    input_ids, token_type_ids, attention_mask = preprocess(news, False)
    names = []
    for input_id, token_type_id, att_mask in zip(input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            f = torch.tensor
            input_id = f(input_id).to(device)
            token_type_id = f(token_type_id).to(device)
            att_mask = f(att_mask).to(device)
            logits = tag_model(input_ids=input_id.view(1, -1),
                                token_type_ids=token_type_id.view(1, -1),
                                attention_mask=att_mask.view(1, -1),
                                labels=None
                                )
            logging.info('post-processing')
            start_logits = logits[0][:, 0]
            end_logits = logits[0][:, 1]
            sort = start_logits.sort(descending=True)
            prob = sort[0].sigmoid()
            index = sort[1]

            # Name start candidate
            start_list = (prob > 0.95).float()
            start_candidates = index[:start_list.tolist().count(1)].cpu().numpy()

            logging.info('start position at:'.format(start_candidates))

            for candidate in start_candidates:
                end_logit = end_logits[candidate:candidate+4]
                end_index = end_logit.sigmoid().cpu().numpy().argmax()
                if end_logit[end_index].sigmoid() < 0.95:
                    continue
                name = input_id[candidate:candidate+end_index+1]
                name = ''.join(tokenizer.decode(name).split())
                #logging.info('predict name: ', name)
                names.append(name)
    
    # Delete duplicated names
    names = list(dict.fromkeys(names))
    return names

if __name__ == '__main__':
    args = parse()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.bert:
        model_version = 'bert-base-chinese'

    elif args.ernie:
        model_version = './ERNIE_pretrained/'

    tokenizer = BertTokenizer.from_pretrained(model_version)
    model = BertModel.from_pretrained(model_version)
    model.to(device)

    if args.train_binary_class:
        dataset = AMLDataset_For_BCL(r'./data/collect_result.csv', tokenizer)
        class_model = modified_bert_for_class(model)
        class_model.to(device)
        train_binary_classification(dataset, model, class_model)

    if args.predict_binary_class:
        dataset = AMLDataset_For_BCL(r'./data/not_ml_dataset.csv', tokenizer)
        class_model = modified_bert_for_class(model)
        class_model.to(device)
        predict_binary_classification(dataset, model, class_model, tokenizer)

    if args.train_tag_name:
        dataset = AMLDataset_For_Tag_Name(r'./data/ml_dataset.csv', tokenizer)
        # zero = 0
        # one = 0
        # for i in range(len(dataset)):
        #     zero += dataset[i][3].count(0)
        #     one += dataset[i][4].count(1)
        tag_model = modified_bert_for_tag(model)
        tag_model.to(device)
        train_tag_criminal_name(dataset, model, tag_model)
        

    if args.predict_competition:
        class_model = modified_bert_for_class(model)
        class_model.to(device)
        tag_model = modified_bert_for_tag(model)
        tag_model.to(device)
        class_model.load_state_dict(torch.load('./model/bert-for-classification-epoch-0702-24.bin', map_location=lambda storage, loc: storage))
        tag_model.load_state_dict(torch.load('./model/bert-for-tagging-epoch-0707-5.bin', map_location=lambda storage, loc: storage))
        prediction = predict_for_competition(class_model, tag_model, tokenizer, device)
        logging.info('Final Predict Result: {}'.format(prediction))