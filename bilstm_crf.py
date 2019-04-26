from keras.models import Model
from keras.layers import Embedding,Bidirectional,LSTM,Input
from keras_contrib.layers import CRF
from data_utils import *
import pickle
import numpy as np

Embed_dim = 100
BiRNN_unit = 200

def create_model(train=True):
	if train:
		(trainx,trainy),(testx,testy),(vocab,chunk_tags),embedding_matrix = load_data(embed_path='data/pretrain_embedding/wiki_100.utf8')
	else:
		(trainx,trainy),(testx,testy),(vocab,chunk_tags),embedding_matrix = load_data(embed_path='data/pretrain_embedding/wiki_100.utf8')
		with open('model/config_file.pkl','rb') as inputs:
			(vocab,chunk_tags) = pickle.load(inputs)
			print(len(vocab),chunk_tags)
#	if train:
#		seq_len = len(trainx[0])
#		inputs = Input(shape=(seq_len,))
#	else:
#		inputs = Input(shape=(seq_len,))
	inputs = Input(shape=(None,))
	embedding = Embedding(len(vocab),Embed_dim,mask_zero=True,weights=[embedding_matrix])(inputs)
	bilstm = Bidirectional(LSTM(BiRNN_unit // 2,return_sequences=True))(embedding)
	crf = CRF(len(chunk_tags),sparse_target=True)
	crf_layer = crf(bilstm)
	model = Model(inputs=inputs,outputs=crf_layer)
	model.summary()
	model.compile('adam',loss=crf.loss_function)
	if train:
		return model,(trainx,trainy),(testx,testy)
	else:
		return model,(vocab,chunk_tags)
