
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import random
import argparse
import sys

import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
import re
import sklearn.preprocessing as p
import sys
import random
import math
from sklearn.metrics import accuracy_score
import copy
import heapq
from tensorflow.contrib import layers
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
input_size = 46
hidden_size = 46
time_default = 1
time_step = 10
global_step=tf.Variable(0)
batch_size = tf.placeholder(tf.int32, [], name='batch_size')
layer_num = 4
class_num = 1
lr = tf.train.exponential_decay(0.0000001,global_step,10,0.9)
regularizer = layers.l1_regularizer(0.000001)
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32,[None,10,46])
cells = []
with tf.variable_scope('LSTM',initializer=tf.orthogonal_initializer()):
	for cell_num in range(layer_num):
		lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=0.1,state_is_tuple=True)
		lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=0.8,output_keep_prob=keep_prob)
		cells.append(lstm_cell)    
		mlstm_cell = rnn.MultiRNNCell(cells, 
                                state_is_tuple=True)
	init_state = mlstm_cell.zero_state(batch_size,dtype=tf.float32)
outputs = list()
state = init_state
with tf.variable_scope('RNN',initializer=tf.orthogonal_initializer()):
	for timestep in range(time_step):
		if timestep>0:
			tf.get_variable_scope().reuse_variables()
		(cell_output,state)=mlstm_cell(tf.reshape(X[:,timestep,:],[batch_size,input_size]),state)
		outputs.append(cell_output)
h_state = outputs[-1]
dense1 = tf.layers.dense(inputs=h_state,units = 16,activation=tf.nn.relu)
dropout1 = tf.layers.dropout(inputs=dense1,rate=0.5)
dense2 = tf.layers.dense(inputs=dropout1,units = 8,activation=tf.nn.relu)
dropout2 = tf.layers.dropout(inputs=dense2,rate=0.5)

W = tf.Variable(tf.truncated_normal([8, 1], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[1]), dtype=tf.float32)
y_pre = (tf.matmul(dropout2,W)+bias)

#W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
#bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
#y_pre = (tf.matmul(h_state, W) + bias)
regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
loss_norm = tf.reduce_mean(tf.square(y-y_pre))+regularization_loss#+0.01*tf.reduce_mean(tf.square(W))+0.01*tf.reduce_mean(tf.square(bias))
optimizer=tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss_norm)
#accuracy = tf.abs(y_pre-y)
loss = tf.reduce_mean(tf.square(y-y_pre))


#########################################
def load_data(paths):    
     queryOrder = list()
     queryDocCount = defaultdict(lambda: list())
     m = 0  # feature dimension
     for path in paths:
        
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                try:
                    qid = int(re.search(r"qid:([0-9]+).", line).group(1))
                    docid = line.strip().split("#docid = ")[1]
                    doc2 = docid.strip().split(" ")[0]
                    docid = doc2
                    if qid not in queryOrder:
                        queryOrder += [qid]
                    queryDocCount[qid] += [docid]
                    features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]
                    m = np.max([m, len(features)])

                except:
                    print("Unexpected error:", sys.exc_info()[0])
            N = len(queryOrder)
            n = np.max([len(value) for value in queryDocCount.values()])

            input = np.zeros([N, 1, n, m])
            output = -1 * np.ones([N, n])
            for line in lines:
                score = int(line.split()[0])
                qid = int(re.search(r"qid:([0-9]+).", line).group(1))
                docid = line.strip().split("#docid = ")[1]
                doc2 = docid.strip().split(" ")[0]
                docid = doc2
                features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]
                features = p.scale(features)
                input[queryOrder.index(qid)][0][queryDocCount[qid].index(docid)] = np.asarray(features)
                output[queryOrder.index(qid)][queryDocCount[qid].index(docid)] = score

       

        return input,output,queryOrder,queryDocCount,len(features)


    
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    #for i in range(num_labels):
    #    labels_one_hot[i,labels_dense[i]]=1
    
    return labels_one_hot




# In[155]:
def ndcg(qlist,dcg,n):
    #print(qlist)
    mm =copy.deepcopy(qlist)
    mm[::-1].sort()
   # mm = heapq.nlargest(qlist,10)
    indcg = 0
    nndcg = 0
    #print(mm)
    for i in range(n):
          nndcg = nndcg + reward(i,mm[i])
    
    return np.sum(dcg[0:n])/(nndcg+0.000001)

def initialize_w(m,n):
    return np.random.rand(m,n),np.zeros([m,n])

# test:a,b=initialize_w(3,2)
def get_G(gamma,r,t,M_t):
    g = 0
    for k in range(M_t):
        g = g +(gamma**(k-1))*r[t+k-1]
    return g
##
def choose_one(p):
    t = len(p)
    s = random.random()
    p = p/np.sum(p)
    begin = 0
    for i in range(t):
        begin = begin + p[i]
        if s < begin:
            return i
def choose_one_exact(p):
    #print(p.shape)
    return p.index(max(p))
def reward(posi,real_posi):
#sess.run(tf.global_variables_initializer())
    if posi==0:
        return 2**real_posi-1
    return (2**real_posi-1)/math.log(posi+1,2)


# In[156]:
# In[201]:


#Algorithm 2

# In[202]:
#paths = ['/home/jianghong/MQ2007/Fold1/train.txt','/home/jianghong/MQ2007/Fold2/train.txt','/home/jianghong/MQ2007/Fold3/train.txt','/home/jianghong/MQ2007/Fold4/train.txt','/home/jianghong/MQ2007/Fold5/train.txt']
path_exclu = '/aut/proj/ir/jianghong/mq2008/Fold5/'
#paths.remove(path_exclu+'train.txt')
feature,rank,qid,docid,leng=load_data([path_exclu+'train.txt'])
feature_vali,rank_vali,qid_vali,docid_vali,leng_vali = load_data([path_exclu+'vali.txt'])
gamma = 1
theta = 0.001
#print(n_classes)
#Algorithm 1
ndcg_best = 0
coutp = 0
#W,W_grad = initialize_w(1,leng)
best_result = 100
for ss in range(1):
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    feed_dict=dict()
    batch_counter=0
    batch_x = np.zeros([50,10,46])
    batch_y = np.zeros([50,1])
    for epoach in range(10000):
        #W_grad = np.zeros([1,leng])a
	err1 = 0
	cout1 = 0
	counter = 0
	counter2 = 0
	zp1 = np.zeros([3,2])
        for query in qid:
	    counter = counter +1
	    if counter>200:
		counter = 0
		#break
            M = len(docid[query])
            qnum = qid.index(query)
            doc0 = copy.deepcopy(docid[query])#This query's documents 
            doc = []
            rank0 = []
            x00 = copy.deepcopy(rank[qid.index(query)])
            for docc in doc0:
                if x00[doc0.index(docc)]!=-1:              #Only use relevant dataset
                    doc.append(docc)           
                    rank0.append(x00[doc0.index(docc)])    #Relevance
            #s,a,r,xs,ys = sampleAnEpisode(W,qnum,doc,rank0,feature,qid,weights, biases, keep_prob)
            X_ = doc          #Document
            Y = rank0        #Relevance
            s = []           #State
            s.append(X_)
            M = len(X_)
              #print((feature[qnum][0][X[0]]).size())
            data_feed = np.zeros([M,46],dtype=float)
            y_chosen = np.zeros(M,dtype=float)       
              #print(M)
              #print(M)
            a =np.zeros(M,dtype=int)
            a_exact =np.zeros(M,dtype=int)
            a_chosen =np.zeros(M,dtype=int)
            r =np.zeros(M,dtype=float)
            r_exact = np.zeros(M,dtype=float)
            r_chosen =np.zeros(M,dtype=float)
        
	
	    for i in range(M):

                add_0=copy.deepcopy(feature[qnum][0][doc0.index(X_[i])].reshape([1,46]))   
		time = np.zeros([10,46])
		time[9,:]=add_0.reshape([1,46])
		dcg_value = 2**Y[i]-1
		batch_x[batch_counter,:,:]=time
		batch_y[batch_counter,:]=dcg_value
		batch_counter = batch_counter+1
		if(batch_counter==49):
                	sess.run(train_op, feed_dict={X:batch_x,y:batch_y, keep_prob: 0.1, batch_size: 50})
                        batch_counter = 0
			train_accuracy = sess.run(loss, feed_dict={
            X:batch_x, y:batch_y, keep_prob: 1.0, batch_size: 50})
		#zp = (sess.run(y_pre, feed_dict={
            #X:time, keep_prob: 1.0, batch_size: 1}))
	#	zp1[int(Y[i]),1] = zp1[int(Y[i]),1]+zp
	#	zp1[int(Y[i]),0] = zp1[int(Y[i]),0]+1
			cout1=cout1+1
			err1 = err1+train_accuracy
                #err1 = err1 +train_accuracy
		 #       if cout1%500==0 and cout1!=0:
		#		print('epoach:')
		#	 	coutp = coutp+1
		 #        	print(coutp)
        	#	 	print('accuracy:')
        	#	 	print(err1/cout1)
        	#	 	print("Best test result: ")
		 	 #print(print(zp1))
			 #zp1 = np.zeros([3,2])
                 #        	print(best_result)
	if(epoach%1==0):
	    cout = 0
	    err = 0
	    bc2 = 0
	    for query in qid_vali:
		#p = p+1
		counter2 = counter2+1
		if counter2 >50:
			counter2 = 0
			#break
            	M = len(docid_vali[query])
            	qnum = qid_vali.index(query)
            	doc0 = copy.deepcopy(docid_vali[query])#This query's documents
            	doc = []
            	rank0 = []
            	x00 = copy.deepcopy(rank_vali[qid_vali.index(query)])
            	for docc in doc0:
                	if x00[doc0.index(docc)]!=-1:              #Only use relevant dataset
                    	    doc.append(docc)
                    	    rank0.append(x00[doc0.index(docc)])    #Relevance
            #s,a,r,xs,ys = sampleAnEpisode(W,qnum,doc,rank0,feature,qid,weights, biases, keep_prob)
            	X_ = doc          #Document`:wq
            	Y = rank0        #Relevance
            	s = []           #State
            	s.append(X_)
            	M = len(X_)
              #print((feature[qnum][0][X[0]]).size())
            	data_feed = np.zeros([M,46],dtype=float)
            	y_chosen = np.zeros(M,dtype=float)
              #print(M)
              #print(M)
            	a =np.zeros(M,dtype=int)
            	a_exact =np.zeros(M,dtype=int)
            	a_chosen =np.zeros(M,dtype=int)
            	r =np.zeros(M,dtype=float)
            	r_exact = np.zeros(M,dtype=float)
            	r_chosen =np.zeros(M,dtype=float)
         #   num_unrollings = M
	        #cout = 0
		#err = 0
            	for i in range(M):
		   # cout = cout +1
                #print(M)
                #print(X[i])
        #       print(feature[qnum][0][doc0.index(X[i])])
 		    etn = copy.deepcopy(feature_vali[qnum][0][doc0.index(X_[i])].reshape([1,46]))
		    #etn_add000 = np.zeros(50)
                    #etn_add000[0:46] = etn
		    #etn_add0 = etn_add000.reshape([1,50])
		#    time = []
		    #for ppz in range(9):
		#	time.append(np.zeros(50).reshape(1,50))
		   # time = np.zeros([10,50])
		 #   time.append(etn_add0)
		    time = np.zeros([10,46])
                    time[9,:]=etn.reshape([1,46])
                    m = np.zeros(3)
                #print(Y[i])
                    m[int(Y[i])] = 1
		    dcg_value = 2**Y[i]-1
		    batch_x[batch_counter,:,:]=time
                    batch_y[batch_counter,:]=dcg_value
                    bc2 = bc2+1
                    if(bc2==49):
                       # sess.run(train_op, feed_dict={X:batch_x,y:batch_y, keep_prob: 0.1, batch_size: 100})
                        bc2 = 0
                        train_accuracy = sess.run(loss, feed_dict={
            X:batch_x, y:batch_y, keep_prob: 1.0, batch_size: 50})
                #feed_dict[train_data[0]]=etn_add000   #[qid][0][doc0.index(doc[i])]
                #feed_dict[train_label[0]]=Y[i].reshape([1,1])a
		    #train_accuracy = sess.run(correct_prediction, feed_dict={
            #X:time, y:dcg_value.reshape([1,1]), keep_prob: 1.0, batch_size: 1})

                    cout = cout+1		    
#predic=sample_prediction.eval({sample_input:etn_add0})
		    err = err + train_accuracy
		    #p = p + Y[i]
                #_,l,predictions,lr=sess.run([optimizer,loss,train_prediction,learning_rate],feed_dict=feed_dict)a
	    print(epoach)
	    print("Validation Error is:")
            print(err/cout)
	    if err/cout<best_result:
	     	best_result = err/cout
	     	saver.save(sess, '/aut/proj/ir/jianghong/pre_mq2008_k5/MyModel'+str(epoach)+str(best_result))
		saver.save(sess, '/aut/proj/ir/jianghong/pre_mq2008_k5/MyModel_best')

    	    print("Best_Result")
	    print(best_result)
        #    num_unrollings = M
        #    _,l,predictions,lr=sess.run([optimizer,loss,train_prediction,learning_rate],feed_dict=feed_dict)
            
'''              #print(M)
            for i in range(M):
                pro = []
                ck = []
                cki = 0
                for j in (s[i]):
                    f = feature[qnum][0][X.index(j)]
                    fT = np.array(f).reshape(1,46)
                    predic=sample_prediction.eval({sample_input:fT})
                #    print(max(max(predic)))
                    pro.append(max(max(predic)))
                    ck.append(j)
                    cki = cki+1
                a_exact[i] = X.index(ck[int(choose_one_exact(pro))])
                a[i] = X.index(ck[int(choose_one(pro))])
                r[i] = reward(i,Y[a[i]])
                r_exact[i] = reward(i,Y[a_exact[i]])
                if r[i]>r_exact[i]:
                    a_chosen[i] = copy.deepcopy(a[i])
                    r_chosen[i] = copy.deepcopy(r[i])
                    y_chosen[i] = 0
                else:
                    a_chosen[i] = copy.deepcopy(a_exact[i])
                    r_chosen[i] = copy.deepcopy(r_exact[i])
                    y_chosen[i] = 1*gamma**i
                data_feed[i] = copy.deepcopy(feature[qnum][0][a_chosen[i]])
                x_ = copy.deepcopy(s[i])
                x_.remove(X[a_chosen[i]])
                s.append(x_)
                feed_dict = dict() 
                #y_all = np.ones([46,1])*y_chosen[i]
                num_unrollings = i
                for mmx in range(i):
                    feed_dict[train_data[i]]=f.reshape([1,window_size-1])
                    feed_dict[train_label[i]]=y_chosen[i].reshape([1,1])
                _,l,predictions,lr=sess.run([optimizer,loss,train_prediction,learning_rate],feed_dict=feed_dict)
'''
''' 

        if(epoach%10==0):
            np.save(path_exclu+'W'+str(epoach)+'.npy',W)
            score_ndcg1 = 0
            score_ndcg3 = 0
            score_ndcg5 = 0
            score_ndcg10 = 0
            count_vali = 0
            for query_vali in qid_vali:
                count_vali = count_vali +1
                Mvali = len(docid_vali[query_vali])
                qnum_vali = qid_vali.index(query_vali)
                doc0_vali = copy.deepcopy(docid_vali[query_vali])
                doc_vali = []
                rank0_vali = []
                x00_vali = copy.deepcopy(rank_vali[qid_vali.index(query_vali)])
                for docc_vali in doc0_vali:
                    if x00_vali[doc0_vali.index(docc_vali)]!=-1:
                        doc_vali.append(docc_vali)
                        rank0_vali.append(x00_vali[doc0_vali.index(docc_vali)])
                #s_vali,a_vali,r_vali = sampleAnEpisode(W,qnum_vali,doc_vali,rank0_vali,feature_vali,qid_vali)
                #xxm = ndcg(rank0_vali,r_vali)
                #print(rank0_vali)
                #print(r_vali)
                X = doc
                Y = rank0
                s = []
                s.append(X)
                M = len(X)
                  #print((feature[qnum][0][X[0]]).size())
                data_feed = np.zeros([M,46],dtype=float)
                y_chosen = np.zeros(M,dtype=float)
                  #print(M)
                  #print(M)
                a =np.zeros(M,dtype=int)
                a_exact =np.zeros(M,dtype=int)
                a_chosen =np.zeros(M,dtype=int)
                r =np.zeros(M,dtype=float)
                r_exact = np.zeros(M,dtype=float)
                r_chosen =np.zeros(M,dtype=float)
                  #print(M)
                for i in range(M):
                    pro = []
                    ck = []
                    cki = 0
                    for j in (s[i]):
                        f = feature[qnum][0][X.index(j)]
                        fT = np.array(f).reshape(1,46)
                        predic=sample_prediction.eval({sample_input:fT})
                 #       print(max(max(predic)))
                        pro.append(max(max(predic)))
                        ck.append(j)
                        cki = cki+1
                    a_exact[i] = X.index(ck[int(choose_one_exact(pro))])
                    a[i] = X.index(ck[int(choose_one(pro))])
                    r[i] = reward(i,Y[a[i]])
                    r_exact[i] = reward(i,Y[a_exact[i]])
                    if r[i]>r_exact[i]:
                        a_chosen[i] = copy.deepcopy(a[i])
                        r_chosen[i] = copy.deepcopy(r[i])
                        y_chosen[i] = 0
                    else:
                        a_chosen[i] = copy.deepcopy(a_exact[i])
                        r_chosen[i] = copy.deepcopy(r_exact[i])
                        y_chosen[i] = 1*gamma**i
                    data_feed[i] = copy.deepcopy(feature[qnum][0][a_chosen[i]])
                    x_ = copy.deepcopy(s[i])
                    x_.remove(X[a_chosen[i]])
                    s.append(x_)
                    feed_dict = dict() 
                    #y_all = np.ones([46,1])*y_chosen[i]
                    #feed_dict[train_data[0]]=f.reshape([1,window_size-1])
                    #feed_dict[train_label[0]]=y_all.reshape([1,window_size-1])
                    #_,l,predictions,lr=sess.run([optimizer,loss,train_prediction,learning_rate],feed_dict=feed_dict)
                rank0_vali = copy.deepcopy(Y)
                r_vali = copy.deepcopy(r_chosen)
                score_ndcg1 = score_ndcg1+ndcg(np.array(rank0_vali),r_vali,1)
                score_ndcg3 = score_ndcg1+ndcg(np.array(rank0_vali),r_vali,3)
                score_ndcg5 = score_ndcg1+ndcg(np.array(rank0_vali),r_vali,5)
                score_ndcg10 = score_ndcg1+ndcg(np.array(rank0_vali),r_vali,10)
            #print(W)
            print('epoach: '+str(epoach))
            print('NDCG1:')
            print(score_ndcg1/count_vali)
            print('NDCG3:')
            print(score_ndcg3/count_vali)
            print('NDCG5:')
            print(score_ndcg5/count_vali)
            print('NDCG10:')
            print(score_ndcg10/count_vali)
            if score_ndcg10/count_vali>ndcg_best:
                ndcg_best = score_ndcg10/count_vali
                np.save(path_exclu+'Wbest'+'.npy',W)
# In[66]:
'''
