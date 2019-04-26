import argparse
import bilstm_crf
import numpy as np
from keras.callbacks import Callback,EarlyStopping
from data_utils import f1

class Metrics(Callback):
	def on_train_begin(self,logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	def on_epoch_end(self, epoch,logs={}):
		val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
#		print(val_predict.shape)
		val_tag = self.validation_data[1]
#		print(val_tag.shape)
		val_result = np.array([[[np.argmax(row)] for row in col] for col in val_predict])
#		print(val_result.shape)
		_val_precision,_val_recall,_val_f1 = f1(val_tag,val_result)
		self.val_precisions.append(_val_precision)
		self.val_recalls.append(_val_recall)
		self.val_f1s.append(_val_f1)
		print('- val_precision: %f - val_recall: %f - val_f1: %f'%(_val_precision,_val_recall,_val_f1))
		return
						
				
if __name__=="__main__":
	a = argparse.ArgumentParser()
	a.add_argument('--epochs')
	a.add_argument('--batch_size')

	args = a.parse_args()

	model,(trainx,trainy),(testx,testy) = bilstm_crf.create_model()
	
	
	batch_size = int(args.batch_size)
	epochs = int(args.epochs)
	earlystopping = EarlyStopping(monitor='val_f1s',patience=5,verbose=0,mode='auto')
	metrics = Metrics()
	#train
	history = model.fit(trainx,trainy,batch_size=batch_size,epochs=epochs,callbacks=[metrics,earlystopping],validation_data=(testx,testy))
	model.save('model/lstm_crf.h5')
