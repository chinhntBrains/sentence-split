
# coding: utf-8

# In[1]:


import numpy as np
import keras
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # Preprocessing

# In[2]:


data_path = "data"
vocab = ["PAD"]
ids_data = []
label_data = []
sentence = []
sentence_label = []
labels = []
for file in os.listdir(data_path):
    with open(os.path.join(data_path, file)) as f:
        for line in f:
            if line.strip() == "":
                ids_data.append(sentence)
                label_data.append(sentence_label)
                sentence_label = []
                sentence = []
                continue
            word, label = line.strip().split()
            if word not in vocab:
                vocab.append(word)
            sentence.append(vocab.index(word))
            if label not in labels:
                labels.append(label)
            sentence_label.append(labels.index(label))
        ids_data.append(sentence)
        label_data.append(sentence_label)


# In[3]:


max_len = max([len(x) for x in ids_data])
X = np.zeros((len(ids_data), max_len))
y = np.zeros((len(ids_data), max_len, len(labels)))
sentence_length = np.zeros(len(ids_data))
for i in np.arange(len(ids_data)):
    X[i, 0:len(ids_data[i])] = ids_data[i]
    y[i, range(0,len(ids_data[i])), label_data[i]] = 1
    sentence_length[i] = len(ids_data[i])


# ## Data visualize

# In[4]:


print("Number of sentence: ", len(ids_data))
print("Number of word in vocab: ", len(vocab))
print("Number of label in data: ", len(labels))


# # Model

# In[7]:


from model import SentenceSplit
import tensorflow as tf


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, np.array(y),
                                                    test_size=0.2, random_state=42)


# In[9]:


def batch(x, y, x_length, batch_size):
    total_step = len(x) // batch_size
    for s in range(total_step):
        cur_i = s*batch_size
        yield x[cur_i:cur_i+batch_size], y[cur_i:cur_i+batch_size], x_length[cur_i:cur_i+batch_size]


# In[ ]:


sess = tf.Session()     
model = SentenceSplit(256, 128, len(labels), 300, len(vocab), max_len)
global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
best_loss = 9999
optimize = tf.train.AdamOptimizer(0.001).minimize(model.losses)
def train_step(sess, global_step, epoch, optimize, x_batch, y_batch, length_batch):
    feed_dict = {
    model.input_sents: x_batch,
    model.sent_lengths : length_batch,
    model.labels : y_batch
    }
    _, loss = sess.run([optimize, model.losses], feed_dict=feed_dict) 
    print("epoch: {}: step: {}, loss: {}".format(epoch,  global_step, loss))
    return loss


# In[ ]:


saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for e in range(10):
    all_loss = []
    for step, (x_batch, y_batch, length_batch) in enumerate(batch(X_train, y_train, sentence_length, 64)):
        tmp_loss = train_step(sess, step, e, optimize,x_batch, y_batch, length_batch )
        all_loss.append(tmp_loss)
    if np.mean(all_loss) >= best_loss:
        break
    best_loss = np.mean(all_loss)
    save_path = saver.save(sess, "./runs/")


# In[ ]:


X_train.shape

