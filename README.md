# LSTMrank
A dynamic document ranking method
input train.txt data in k5sec_mq2008_pre.py, no validate data
input train.txt data in k5sec_mq2008.py, validate data is vali.txt
input train.txt+vali.txt in k5ndcg_mq2008.py, validate data is test.txt

validate data and train data's variables are strictly seperated by their name(validate data has postfix _vali)
after double check_, I did not find any information leakage.

I also did an experiment by seperating each file into two files, so that each one only contains train data or validate data, the result remains the same.

Thus, I assume there is no information leakage.
