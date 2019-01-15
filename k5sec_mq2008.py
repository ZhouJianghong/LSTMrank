from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

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
import os
from tensorflow.contrib import layers

#os.system('./pre_f1.py')
# LSTM
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
input_size = 46
hidden_size = 46
time_default = 1
time_step = 10#tf.placeholder_with_default(time_default,shape=None)
batch_size = tf.placeholder(tf.int32, [], name='batch_size')
layer_num = 4
class_num = 1
#lr = 0.001
global_step=tf.Variable(0)
#batch_size = tf.placeholder(tf.int32, [], name='batch_size')
#layer_num = 3
#class_num = 1
lr = tf.train.exponential_decay(0.001,global_step,1000,0.9)
regularizer = layers.l1_regularizer(0.00001)
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)
#ll
X = tf.placeholder(tf.float32,[None,10,46])
cells = []
for cell_num in range(layer_num):
        lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=0.1,state_is_tuple=True)
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=0.8,output_keep_prob=keep_prob)
        cells.append(lstm_cell)
#Input Data
mlstm_cell = rnn.MultiRNNCell(cells,
                                state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size,dtype=tf.float32)
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
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

#y_p = tf.argmax(y_pre[0])
#cross_entropy = tf.square(y-y_pre)+0.01*tf.reduce_mean(tf.square(W))+0.01*tf.reduce_mean(tf.square(bias))
regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
loss_norm = tf.reduce_mean(tf.square(y-y_pre))+regularization_loss#+0.01*tf.reduce_mean(tf.square(W))+0.01*tf.reduce_mean(tf.square(bias))

#train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#sess = tf.Session()a
optimizer=tf.train.AdamOptimizer(lr)

#train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_norm)

train_op = optimizer.minimize(loss_norm)
#optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

#ptimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

#print(sess.run(y_pre))
#print(sess.run(y))
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Model-setting
def lmax(a,b):
	if a>=b:
		return a
	return b

def load_similarity(paths):
	with open(paths,"r") as f:
         	lines = f.readlines()
		Len_query = 0
		Len_doc = 0 
		Len_com = 0
		
        	for line in lines:
			get1 = []
			get2 = []
                	tup = re.split(' ',line)
                	for q in tup:
                        	mm = (re.split(':',q))
                        	get1.append(mm[0].strip())
                        	get2.append(mm[1].strip())
			Len_query = lmax(Len_query,int(get2[0]))
			Len_doc = lmax(Len_query,int(get2[1]))
			for xx in get1[2:]:
				Len_com = lmax(int(xx),Len_com)
		#	for i in range(length(get1)-2): 
	     	#		similarity[int(get2[0])][int(get2[1])][int(get1[i+2])]=get2[i+2]
                similarity = np.zeros([Len_query,Len_doc,Len_com])
		for line in lines:
                        get1 = []
                        get2 = []
                        tup = re.split(' ',line)
                        for q in tup:
                                mm = (re.split(':',q))
                                get1.append(mm[0].strip())
                                get2.append(mm[1].strip())
                        #        Len_query = lmax(Len_query,int(get2[0]))
                        #        Len_doc = lmax(Len_query,int(get))
                        #        Len_com = lmax(max(int(get2[2:])),Len_com)
                        for i in range(len(get1)-2):
                                similarity[int(get2[0])-1][int(get2[1])-1][int(get1[i+2])-1]=float(get2[i+2])
	return similarity
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
    ideal_ndcg = 0
    real_ndcg = 0
  #  print(len(mm))
 #   print(n)
    for i in range(min(n,len(mm))):
#	  print(mm[i])
#	  print(dcg[i])
          ideal_ndcg = ideal_ndcg + (2**mm[i]-1)/math.log(i+2,2)
          real_ndcg = real_ndcg + (2**dcg[i]-1)/math.log(i+2,2)
    if ideal_ndcg ==0:
	return 1
    return real_ndcg/(ideal_ndcg)

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
    same = True
    for i in range(len(p)):
	if p[0]!=p[i]:
		same = False
    if same:
	return int(random.random()*len(p))
    s = random.random()
    p = p/np.sum(p)
    begin = 0
    for i in range(t):
        begin = begin + p[i]
        if s < begin:
            return i
def choose_one_exact(p):
    #print(p)
    same = True
 #   print(len(p))
    for i in range(len(p)):
#	print(p[i])
        if p[0]!=p[i]:
	
                same = False
    if same:
        return int(random.random()*len(p))

    return p.index(max(p))
def reward(posi,real_posi):
    #if posi==0:
    #    return 2**real_posi-1
    return (2**real_posi-1)/math.log(posi+2,2)

# Main function
#path_similarity = './Large_simi.txt'
#sim = load_similarity(path_similarity)
#paths = ['/home/jianghong/MQ2007/Fold1/train.txt','/home/jianghong/MQ2007/Fold2/train.txt','/home/jianghong/MQ2007/Fold3/train.txt','/home/jianghong/MQ2007/Fold4/train.txt','/home/jianghong/MQ2007/Fold5/train.txt']
path_exclu = '/aut/proj/ir/jianghong/mq2008/Fold5/'
#paths.remove(path_exclu+'train.txt')
feature,rank,qid,docid,leng=load_data([path_exclu+'train.txt'])
feature_vali,rank_vali,qid_vali,docid_vali,leng_vali = load_data([path_exclu+'vali.txt'])  #in here we load train and vali 
gamma = 0.9
theta = 0.001
#print(n_classes)
#Algorithm 1
ndcg_best = 0
cget1 = 0
cget2 = 0
batch_counter = 0
W,W_grad = initialize_w(1,leng)
for ss in range(1):
    saver = tf.train.Saver()
    #tf.initialize_all_variables().run(a
    #saver.restore(sess, '/aut/proj/ir/jianghong/testpre/MyModel_smalltestkeepf8k8r00000187220.2931517784547851')
    saver.restore(sess,'/aut/proj/ir/jianghong/pre_mq2008_k5/MyModel_best')
    feed_dict=dict()
    for epoach in range(25000):
        W_grad = np.zeros([1,leng])
        for query in qid:
	    cget1= cget1 +1
#	    print(cget1)
	    #if cget1>10:
		#cget1 = 0
	    	#break;
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
	    count_chosen = 0
            a =np.zeros(M,dtype=int)
            a_exact =np.zeros(M,dtype=int)
            a_chosen =np.zeros(M,dtype=int)
            r =np.zeros(M,dtype=float)
            r_exact = np.zeros(M,dtype=float)
            r_chosen =np.zeros(M,dtype=float)
         #   num_unrollings = M
#	    for i in range(M):
                #print(M)
		#print(X[i])
	#	print(feature[qnum][0][doc0.index(X[i])])
#                feed_dict[train_data[0]]=feature[qnum][0][doc0.index(X[i])].reshape([1,window_size-1])   #[qid][0][doc0.index(doc[i])]
#                feed_dict[train_label[0]]=Y[i].reshape([1,1])
# 		_,l,predictions,lr=sess.run([optimizer,loss,train_prediction,learning_rate],feed_dict=feed_dict)
#        if(epoach%1000==0):
#	     saver.save(sess, './pre/MyModel'+str(epoach))
    
        #    num_unrollings = M
        #    _,l,predictions,lr=sess.run([optimizer,loss,train_prediction,learning_rate],feed_dict=feed_dict)
            
              #print(M)
	    number_doc = 0
	#    cget = 0
	    fea_archive = []
	    batch_x = np.zeros([50,10,46])
	    batch_y = np.zeros([50,1])

            for i in range(min(M,10)):# Range all documents
		number_doc = number_doc+1
                pro = []
                ck = []
		fea_pro_archive = []
                cki = 0
		ndcg_archive = []
		last = np.zeros(input_size)
		last2 = np.zeros(input_size)
		if random.random()>=0:
                	for j in (s[i]):
                    #f = feature[qnum][0][doc0.index(j)]# We must be the feature of a chosen doc
                    #fT = np.array(f).reshape(1,46)
                    		add_0=copy.deepcopy(feature[qnum][0][doc0.index(j)].reshape([1,46]))   #[qid][
                    		add_000 = np.zeros(input_size)
                    		add_000[0:46] = add_0
		    		time = np.zeros([10,input_size])
                    		time[9,:]=add_000.reshape([1,input_size])
				time[8,:]=last
				time[7,:]=last2
                    		tx = min(count_chosen,9)
		    #y_used = copy.deepcopy(y_chosen)
		    #y_used[i]=Y[X_.index(j)]
		    #y_this = reward(i,Y[X_.index(j)])
		    #y_this = ndcg(rank0,y_used,i)
		    
	#	    tmx = len(a_choesn)
		    	#	for iz in range(tx):
			#pz = X_[a_chosen[count_chosen-iz-1]]
			#		time[8-iz,:] = fea_archive[-iz]#feature[qnum][0][doc0.index(pz)].reshape([1,46])
		    		fea_pro_archive.append(time[9,:]) 
		    		batch_x_test = np.zeros([1,10,46])
		    		batch_y_test = np.zeros([1,1])
		    		batch_x_test[0,:,:]=time
		    #batch_y_test[0,:]=
                #print(add_000)
                    #feed_dict[train_data[0]]= add_000.reshape([1,window_size-1])
                    #feed_dict[train_label[0]]=Y[i].reshape([1,1])
                    #_,l,predictions,lr=sess.run([train_step,loss,train_prediction,learning_rate],feed_dict=feed_dict)
                    		predict = sess.run(y_pre, feed_dict={ X:batch_x_test, keep_prob: 1.0, batch_size: 1})
   	          #  print(predict)
			
                    		predic =predict+1#abs(predict-y_this)#predict#+0.5*abs(predict-y_this)#0*predict[0][1]+1**predict[0][1]+2**predict[0][2]
                    #predic=sample_prediction.eval({sample_input:add_000.reshape([1,window_size-1])})
                #    print(max(max(predic)))
                    		pro.append(predic)
                    		ck.append(j)
                    		cki = cki+1
		 #   ndcg_archive.append(y_this)
		#print(pro)
	#	print(pro)a
		#print(pro)
			cc = int(choose_one_exact(pro))
			time = np.zeros([10,input_size])

			time[9,:]=fea_pro_archive[cc]
	                fea_archive.append(fea_pro_archive[cc])
			a_exact[i] = X_.index(ck[cc])
		
		else:
			cc = random.randint(0,len(s[i])-1)
			ck = copy.deepcopy(s[i])
			add_0=copy.deepcopy(feature[qnum][0][doc0.index(s[i][cc])].reshape([1,46])) 
			#fea_pro_archive.append(add_0)
			fea_archive.append(add_0)
			time = np.zeros([10,input_size])

                        time[9,:]=add_0
			a_exact[i] = X_.index(s[i][cc])

	#	print(pro)
	#	print(cc)
		#print(pro)
                #a_exact[i] = X_.index(ck[cc])
		#cc2 = int(choose_one(pro))
                #a[i] = X_.index(ck[cc2])
                #r[i] = reward(i,Y[a[i]])
		
                r_exact[i] =  reward(1,Y[a_exact[i]])#ndcg_archive[cc]# reward(i,Y[a_exact[i]])
                #if r[i]>r_exact[i]:# or random.random()>0.5**(1/(epoach+0.001)):
                #    a_chosen[i] = copy.deepcopy(a[i])
                 #   r_chosen[i] = copy.deepcopy(r[i])
                  #  y_chosen[i] = Y[a[i]]# pro[cc]

		 #   count_chosen = count_chosen+1
                #else:
                a_chosen[i] = copy.deepcopy(a_exact[i])
                r_chosen[i] = copy.deepcopy(r_exact[i])
                y_chosen[i] = Y[a_exact[i]]#pro[cc2]
		count_chosen = count_chosen+1
		# cc = cc2
		#In_c =doc0.index(X[a_chosen[i]])
                #data_feed[i] = copy.deepcopy(feature[qnum][0][In_c])
                x_ = copy.deepcopy(s[i])
                x_.remove(X_[a_chosen[i]])
                s.append(x_)
                #feed_dict = dict() 
                #y_all = np.ones([46,1])*y_chosen[i]
                #feed_dict[num_unrollings] = i
#		time = np.zeros([10,input_size])
 #               time[9,:]=fea_pro_archive[cc]
#		fea_archive.append(fea_pro_archive[cc])
                tx = min(count_chosen,9)
		
		time[8,:]=last
		time[7,:]=last2
		last2 = last
		last = a_chosen[i]
                #m = np.zeros(3)
		#m[int(y_chosen[i])]=1
		#cget = cget+1
		#if i==0:
		#	print(pro[cc2])
		#	print(r_chosen[0])
	#	if cget%100==0:
	#		print(r_chosen)
	#		print(y_chosen)
		#for iz in range(tx):
                        #pz = X_[a_chosen[count_chosen-iz-1]]
                 #       time[9-iz,:] = fea_archive[-iz]#feature[qnum][0][doc0.index(pz)].reshape([1,46])
	#	print(time)
	#	print(m
		#print("yes")
		batch_x[batch_counter,:,:]=time
                batch_y[batch_counter,:]= r_chosen[i]#ndcg(rank0,y_chosen,i)
                batch_counter = batch_counter+1
                if(batch_counter==49):
                        sess.run(train_op, feed_dict={X:batch_x,y:batch_y, keep_prob: 0.5, batch_size: 50})
                        batch_counter = 0

                #sess.run(train_op,feed_dict={X:time,y:r_chosen[i].reshape([1,1]), keep_prob: 0.1, batch_size: 1})


                #for mmx in range(i):
                 #   feed_dict[train_data[i]]=f.reshape([1,window_size-1])
                  #  feed_dict[train_label[i]]=y_chosen[i].reshape([1,1])
                #_,l,predictions,lr=sess.run([train_step,loss,train_prediction,learning_rate],feed_dict=feed_dict)
        if(epoach%1==0):  #After here, we only have vali
   
#         np.save(path_exclu+'W'+str(epoach)+'.npy',W)
            score_ndcg1 = 0
            score_ndcg3 = 0
            score_ndcg5 = 0
            score_ndcg10 = 0
            count_vali = 0
            for query in qid_vali:
		cget2 = cget2 +1
		#if cget2>50:
		#	cget2 = 0
		#	break
                #	continue
	        #print(cget2)
		count_vali = count_vali+1
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
            	count_chosen = 0
            	a =np.zeros(M,dtype=int)
            	a_exact =np.zeros(M,dtype=int)
            	a_chosen =np.zeros(M,dtype=int)
            	r =np.zeros(M,dtype=float)
            	r_exact = np.zeros(M,dtype=float)
            	r_chosen =np.zeros(M,dtype=float)
            	number_doc = 0
		fea_archive = []

#		print(X_)
#		print(Y)
            	for i in range(min(M,10)):# Range all documents
                	number_doc = number_doc+1
                	pro = []
               		ck = []
                	cki = 0
			fea_pro_archive = []
			last = np.zeros(input_size)
			last2 =  np.zeros(input_size)
                	for j in (s[i]):
                    #f = feature[qnum][0][doc0.index(j)]# We must be the feature of a chosen doc
                    #fT = np.array(f).reshape(1,46)
                    		add_0=copy.deepcopy(feature_vali[qnum][0][doc0.index(j)].reshape([1,46]))   #[qid][
                    		add_000 = np.zeros(input_size)
                    		add_000[0:46] = add_0
                    		time = np.zeros([10,input_size])
                    		time[9,:]=add_000.reshape([1,input_size])
				time[8,:]=last
				time[7,:]=last2
                    		tx = min(count_chosen,9)
                    #tmx = len(a_choesn)
				y_this = Y[X_.index(j)]
                    	#	if count_chosen-1>=0:
                        #		time[9,49]=math.fabs(y_this-y_chosen[count_chosen-1])
                    	#	if count_chosen-2>=0:
                        #		time[9,48]=math.fabs(y_this-y_chosen[count_chosen-2])
                    	#	if count_chosen-3>=0:
                        #		time[9,47]=math.fabs(y_this-y_chosen[count_chosen-3])

                    #tmx = len(a_choesn)
                  #  		for iz in range(tx):
                        #pz = X_[a_chosen[count_chosen-iz-1]]
                   #     		time[8-iz,:] = fea_archive[-iz]#feature[qnum][0][doc0.index(pz)].reshape([1,46])
                    		fea_pro_archive.append(time[9,:])
                                #predict = sess.run(y_pre, feed_dict={X:time, keep_prob: 1.0, batch_size: 1})

                    		#for iz in range(tx):
                        	#	pz = X_[a_chosen[count_chosen-iz-1]]
                        	#	time[8-iz,:] = feature[qnum][0][doc0.index(pz)].reshape([1,46])
                        	batch = np.zeros([1,10,46])
				batch[0,:,:]=time
				predict = sess.run(y_pre, feed_dict={X:batch, keep_prob: 1.0, batch_size: 1})
                    		predic = predict#0*predict[0][0]+1**predict[0][1]+2**predict[0][2]

                    #predic=sample_prediction.eval({sample_input:add_000.reshape([1,window_size-1])})
                #    print(max(max(predic)))
                    		pro.append(predic)
                    		ck.append(j)
                    		cki = cki+1
			cc = int(choose_one_exact(pro))
                	a_exact[i] = X_.index(ck[cc])
                	a[i] = X_.index(ck[int(choose_one(pro))])
                	r[i] = reward(i,Y[a[i]])
                	r_exact[i] = reward(i,Y[a_exact[i]])
                	#if r[i]>r_exact[i]:
                    	#	a_chosen[i] = copy.deepcopy(a[i])
                    	#	r_chosen[i] = copy.deepcopy(Y[a[i]])
                    	#	y_chosen[i] = 0
                    	#	count_chosen = count_chosen+1
                	#else:
                    	a_chosen[i] = copy.deepcopy(a_exact[i])
#			print(Y[a[i]])
                    	#r_chosen[i] = copy.deepcopy(Y[a[i]])
                    	y_chosen[i] = copy.deepcopy(Y[a[i]])
                    	count_chosen = count_chosen+1
                #	In_c =doc0.index(X[a_chosen[i]])
                #	data_feed[i] = copy.deepcopy(feature[qnum][0][In_c])
                	x_ = copy.deepcopy(s[i])
                	x_.remove(X_[a_chosen[i]])
                	s.append(x_)
                	#feed_dict = dict()
                #y_all = np.ones([46,1])*y_chosen[i]
                #feed_dict[num_unrollings] = i
                	time = np.zeros([10,46])
                	#time[9,0:46]=feature[qnum][0][doc0.index(X_[a_chosen[i]])].reshape([1,46])
                	tx = min(count_chosen,9)
			time[7,:]=last2
			time[8,:]=last
			time[9,:]=fea_pro_archive[cc]
			last2 = time[8,:]
			last = time[9,:]
	                fea_archive.append(fea_pro_archive[cc])
			#print("yes")
                #	m = np.zeros(3)
                #	m[y_chosen[i]]=1
                #	for iz in range(tx):
                  #      	pz = X_[a_chosen[count_chosen-iz]]
                 #       	time[9-iz,0:46] = feature[qnum][0][doc0.index(pz)].reshape([1,46])
                #	sess.run(train_op,feed_dict={X:time,y:m.reshape([1,3]), keep_prob: 0.1, batch_size: 1})
#                print(rank0)
#		print(r_chosen)
#		print(ndcg(np.array(rank0),r_chosen,10))
	
                score_ndcg1 = score_ndcg1+ndcg(np.array(rank0),y_chosen,1)
                score_ndcg3 = score_ndcg3+ndcg(np.array(rank0),y_chosen,3)
                score_ndcg5 = score_ndcg5+ndcg(np.array(rank0),y_chosen,5)
                score_ndcg10 = score_ndcg10+ndcg(np.array(rank0),y_chosen,10)
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
	    avg = (score_ndcg1+score_ndcg3+score_ndcg5+score_ndcg10)/(4*count_vali)
            if avg>ndcg_best:
                ndcg_best = avg
                saver.save(sess, '/aut/proj/ir/jianghong/sec_l3_mq2008_k5/MyModel_1019_05absndcg'+str(epoach)+str(ndcg_best))
		with open("/aut/proj/ir/jianghong/sec_l3_mq2008_k5/result_05absndcg.txt","a") as f:
		       content = str(epoach)+" NDCG1: "+str(score_ndcg1/count_vali)+" ndcg3: "+str(score_ndcg3/count_vali)+" NDCG5: "+str(score_ndcg5/count_vali)+" NDCG10: "+str(score_ndcg10/count_vali)+"avg:"+str(avg)+"\n"
                       f.write(content)



