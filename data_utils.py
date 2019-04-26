import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import platform
import pickle
from collections import Counter
import pdb

#pdb.set_trace()

# --------input: open file object     output: data in shape[paragrah,row,2], 2 means [character,label]------------
def _parse_data(f):
	if platform.system() == 'Windows':
		split_text = '\r\n'
	else:
		split_text = '\n'
	string = f.read().decode('utf-8')
	data = [[row.split() for row in sample.split(split_text)] for sample in string.strip().split(split_text+split_text)]

	f.close()
	return data

def load_data(embed_path=None):
	train = _parse_data(open('data/example/example.train','rb'))
	test = _parse_data(open('data/example/example.test','rb'))
	word_counts = Counter(row[0].lower() for sample in train for row in sample)
	vocab = [w for w,f in iter(word_counts.items()) if f>=2]

	chunk_tags = ['O','B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG']

	# save config data
	with open('model/config_file.pkl','wb') as output:
		pickle.dump((vocab,chunk_tags),output)

	train = _process_data(train,vocab,chunk_tags)
	test = _process_data(test,vocab,chunk_tags)

	if embed_path == None:
		return train,test,(vocab,chunk_tags)
	else:
		return train,test,(vocab,chunk_tags),np.array(_pre_embed(vocab,embed_path))

# ------- input is data[paragraph,row,2], vocab[dict_size], chunk_tags=[7]      output is x[paragraph,max_len], y_chunk---------------------
def _process_data(data,vocab,chunk_tags,max_len=None,onehot=False):
	if max_len is None:
		max_len = max(len(s) for s in data)
	word2idx = dict((w,i) for i,w in enumerate(vocab))

	# x_chunk: shape[paragraph,id]
	x = [[word2idx.get(w[0].lower(),1) for w in s]for s in data]  # dict.get() make the word not in dictionary to be a <unknown>(index 1)

	# y_chunk: shape[paragraph,id]
	y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

	x = pad_sequences(x,max_len)
	y_chunk = pad_sequences(y_chunk,max_len)

	if onehot:
		y_chunk = np.eye(len(chunk_tags),dtype='float32')[y_chunk]
	else:
		y_chunk = np.expand_dims(y_chunk,2)
	print(x.shape,y_chunk.shape)	
	return x,y_chunk

# -------- input is vocab and pre_embedding path, output is array [vocab_size,embed_size] --------------------
def _pre_embed(vocab,embed_path):
	seq = []
	value = []
	f_embed = open(embed_path,'r')
	lines_embed = f_embed.readlines()
	for line in lines_embed:
		line = line.replace('\n','')
		seq.append(line.split(' ')[0])
		value.append(line.split(' ')[1:])

	dict_embed = {k:v for k,v in zip(seq,value)}
	f_embed.close()

	# build the pre_trained embedding
	count = 0
	pre_embedding = []
	zero = [0]*len(dict_embed[','])
	for i in range(len(vocab)):
		if vocab[i] not in list(dict_embed.keys()):
			pre_embedding.append(zero)
		else:
			pre_embedding.append(dict_embed[vocab[i]])
			count = count+1
	print('words in pre_trained embeddings are ',count/len(vocab))
	return pre_embedding

# ------ input is data[sequence]    output is x[max_len] ----------
def process_data(data,vocab,max_len=100):
	word2idx = dict((w,i) for i,w in enumerate(vocab))
	x = [word2idx.get(w[0].lower(),1) for w in data]
	length = len(x)
	x = pad_sequences([x],max_len)
	return x,length

# ------ input is sequence a[x,y,1],b[x,y,1], calculate f1--------
def f1(a,b):
	tag_num,predict_num,match_num=0,0,0

	tag_index_l = []
	tag_index_r = []
	for i in range(a.shape[0]):
		index_l = []
		index_r = []
		for j in range(a.shape[1]):
			if a[i][j][0] in (1,3,5):
				tag_num = tag_num+1   #calculate the tag_num
				index_l.append(j)
				j_tmp = j+1
				while(j_tmp<a.shape[1] and a[i][j_tmp][0] == a[i][j_tmp-1][0]+1):
					j_tmp = j_tmp+1
				index_r.append(j_tmp)
			if b[i][j][0] in (1,3,5):
				predict_num = predict_num+1   #calculate the predict_num
		tag_index_l.append(index_l)
		tag_index_r.append(index_r)

	for i in range(a.shape[0]):
		for k in range(len(tag_index_l[i])):
			for p in range(tag_index_l[i][k],tag_index_r[i][k]):
				if (a[i][p][0]!=b[i][p][0]):
					match_num = match_num+1
					break
	match_num = tag_num - match_num
	print(tag_num,predict_num,match_num)
	recall = match_num/tag_num
	if predict_num == 0:
		precision = 0
		f1 = 0
	else:
		precision = match_num/predict_num
		f1 = 2*(precision*recall)/(precision+recall)
	return precision,recall,f1
		
					
				
				
			
