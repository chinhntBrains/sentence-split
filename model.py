import tensorflow as tf


class SentenceSplit():
  def __init__(self, hidden_layer,projection_layer,
               num_class, vector_size,  num_vocab, max_sent_len):
    self.hidden_layer = hidden_layer
    self.project_layer = projection_layer
    self.num_class = num_class
    self.vector_size = vector_size
    self.num_vocab = num_vocab
    self.max_sent_len = max_sent_len
    self._add_placeholder()
    self._add_embedding()
    self. _add_graph()
    self._add_losses()

  def _add_placeholder(self):
    self.input_sents = tf.placeholder(tf.int32, [None, self.max_sent_len], name="input_word_ids")
    self.sent_lengths = tf.placeholder(tf.int32, [None], name="sent_length")
    self.labels = tf.placeholder(tf.float32, [None, self.max_sent_len, self.num_class],
                                 name="labels")

  def _add_embedding(self):
    with tf.device('/cpu:0'):
      embedding_matrix = tf.get_variable("embedding_word_matrix", [self.num_vocab, self.vector_size],
                                         initializer=tf.random_normal_initializer())
      self.input_sents_embdedding = tf.nn.embedding_lookup(embedding_matrix, self.input_sents)

  def _add_graph(self):
    batch_size, _ = tf.unstack(tf.shape(self.input_sents))
    fw_lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_layer)
    bw_lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_layer)
    (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
      fw_lstm_cell,
      bw_lstm_cell,
      self.input_sents_embdedding,
      self.sent_lengths,
      dtype=tf.float32
      )
    birnn_output = tf.concat((fw_out, bw_out), axis=2)
    W1 = tf.get_variable('W1', [2 * self.hidden_layer, self.project_layer], tf.float32)
    b1 = tf.get_variable('b1', [self.project_layer], tf.float32)
    W2 = tf.get_variable('W_fw', [self.project_layer, self.num_class], tf.float32)
    b2 = tf.get_variable('b_fw', [self.num_class], tf.float32)
    z1 = tf.nn.relu(tf.matmul(birnn_output, tf.tile(tf.expand_dims(W1, 0),
                                                     [batch_size,1,1])) + b1)
    self.logit = tf.matmul(z1, tf.tile(tf.expand_dims(W2, 0),
                                  [batch_size,1,1])) + b2
    self.predict = tf.argmax(tf.nn.softmax(self.logit),2)

  def _add_losses(self):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,logits=self.logit )
    self.losses = tf.reduce_mean(loss)
      
    
                                         
