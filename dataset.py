from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import pdb
import pickle
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import re
from ast import literal_eval

class AMLDataset_For_Tag_Name(Dataset):
	def __init__(self, f_path, tokenizer):
		df = pd.read_csv(f_path, lineterminator='\n')
		news_content = []
		name_label = []
		texts = df.news_content.tolist()
		names = df.name.tolist()

		for text, name in zip(texts, names):
			name_list = literal_eval(name)
			split = get_split(text)
			
			for i in name_list:
				news_content += split
				name_label += [i] * len(split)

		# Add special token
		for i in range(len(news_content)):
			news_content[i] = '[CLS]金錢[SEP]犯人[SEP]' + news_content[i] + '[SEP]'
		
		self.news_contents = news_content
		self.name_labels = name_label
		self.tokenizer = tokenizer

		###
		# sta_labels = []
		# end_labels = []
		# for i in range(len(news_content)):
		# 	sta_label = [0] * len(news_content[i])
		# 	end_label = [0] * len(news_content[i])
		# 	text_tokens = self.tokenizer.tokenize(news_content[i])
		# 	input_id = self.tokenizer.convert_tokens_to_ids(text_tokens)
		# 	name_tokens = self.tokenizer.tokenize(name_label[i])
		# 	name_ids = self.tokenizer.convert_tokens_to_ids(name_tokens)
			
		# 	span_list = find_sub_list(name_ids, input_id)
		# 	for span in span_list:
		# 		sta_label[span[0]] = 1
		# 		end_label[span[1]] = 1
		# 	sta_labels.append(sta_label)
		# 	end_labels.append(end_label)

		# pdb.set_trace()

		###

	def __getitem__(self, idx):
		news_content, name_label = self.news_contents[idx], self.name_labels[idx]
		
		text_tokens = self.tokenizer.tokenize(news_content)
		input_id = self.tokenizer.convert_tokens_to_ids(text_tokens)
		input_ids = input_id
		token_type_ids = create_token_type(input_id)
		attention_mask = create_mask(input_id)
		name_tokens = self.tokenizer.tokenize(name_label)
		name_ids = self.tokenizer.convert_tokens_to_ids(name_tokens)
		span_list = find_sub_list(name_ids, input_ids)

		start_target = [0] * len(input_ids)
		end_target = [0] * len(input_ids)

		for span in span_list:
			start_target[span[0]] = 1
			end_target[span[1]] = 1

		return input_ids, token_type_ids, attention_mask, start_target, end_target

	def __len__(self):
		return len(self.news_contents)


class AMLDataset_For_BCL(Dataset):
	def __init__(self, f_path, tokenizer):
		df = pd.read_csv(f_path, lineterminator='\n')
		news_content = []
		label = []

		df = df.dropna()
		texts = df.news_content.tolist()
		categories = df.label.tolist()
		for text, category in zip(texts, categories):
			split = get_split(text)
			news_content += split 
			label += [category]*len(split)

		# Add special Token
		for i in range(len(news_content)):
			news_content[i] = '[CLS]金錢[SEP]犯罪[SEP]' + news_content[i] + '[SEP]'

		self.news_contents = news_content
		self.labels = label
		self.tokenizer = tokenizer

	def get_df(self):
		return self.df
		
	def __getitem__(self, idx):
		news_contents, labels = self.news_contents[idx], self.labels[idx]

		tokens = self.tokenizer.tokenize(news_contents)
		input_id = self.tokenizer.convert_tokens_to_ids(tokens)
		input_ids = input_id
		token_type_ids = create_token_type(input_id)
		attention_mask = create_mask(input_id)

		return input_ids, token_type_ids, attention_mask, labels

	def __len__(self):
		return len(self.labels)

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if np.array_equal(l[ind:ind+sll], sl): # check array is equal
            results.append((ind,ind+sll-1))

    return results

def create_mask(input_id):
    return [1 if idx != 0 else 0 for idx in input_id]

def create_token_type(input_id):
	token_type = [0] *  7 + [1] * (len(input_id)-7)
	return token_type

def get_split(text):
	split = []
	length = 450
	window_size = 300
	text = text.replace('\n', '')
	text = re.sub(' ', '', text)
	if len(text) // window_size > 0:
		n = len(text)//window_size
	else:
		n = 1

	for w in range(n):
		split_news = text[w*window_size:w*window_size + length]
		split.append(split_news)

	return split
	

def pad_for_BCL(batch):
	f = lambda x: [sample[x] for sample in batch]
	input_ids = f(0)
	token_type_ids = f(1)
	attention_mask = f(2)
	labels = f(3)

	f = lambda x: pad_sequences(x, maxlen=512, dtype='long', truncating='post', padding='post')

	input_ids = f(input_ids)
	token_type_ids = f(token_type_ids)
	attention_mask = f(attention_mask)

	f = torch.tensor
	return f(input_ids), f(token_type_ids), f(attention_mask), f(labels)

def pad_for_Tag_Name(batch):
	f = lambda x: [sample[x] for sample in batch]
	
	input_ids = f(0)
	token_type_ids = f(1)
	attention_mask = f(2)
	start_target = f(3)
	end_target = f(4)

	f = lambda x: pad_sequences(x, maxlen=512, dtype='long', truncating='post', padding='post')

	input_ids = f(input_ids)
	token_type_ids = f(token_type_ids)
	attention_mask = f(attention_mask)
	start_target = f(start_target)
	end_target = f(end_target)

	f = torch.tensor

	return f(input_ids), f(token_type_ids), f(attention_mask), f(start_target), f(end_target)


