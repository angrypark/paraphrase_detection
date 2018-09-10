import os
import tensorflow as tf
import numpy as np
from gensim.models import FastText

from utils.utils import JamoProcessor
from models.base import BaseModel

def get_embeddings(vocab_list_dir, 
                   pretrained_embed_dir, 
                   vocab_size, 
                   embed_dim):
    embedding = np.random.uniform(-1/16, 1/16, [vocab_size, embed_dim])
    if os.path.isfile(pretrained_embed_dir) & os.path.isfile(vocab_list_dir):
        with open(vocab_list_dir, "r") as f:
            vocab_list = [word.strip() for word in f if len(word)>0]
        processor = JamoProcessor()
        ft = FastText.load(pretrained_embed_dir)
        num_oov = 0
        for i, vocab in enumerate(vocab_list):
            try:
                embedding[i, :] = ft.wv[processor.word_to_jamo(vocab)]
            except:
                num_oov += 1
        print("Pre-trained embedding loaded. Number of OOV : {} / {}".format(num_oov, len(vocab_list)))
    else:
        print("No pre-trained embedding found, initialize with random distribution")
    return embedding

class DeepParaphrase(BaseModel):
    def __init__(self, data, config, mode="train"):
        super(DeepParaphrase, self).__init__(data, config)
        self.mode = mode
        self.build_model()
        self.init_saver()
        
    def build_model(self):
        with tf.variable_scope("inputs"):
            # Placeholders for input
            self.input_A = tf.placeholder(tf.int32, [None, self.config.max_length], name="input_A")
            self.input_B = tf.placeholder(tf.int32, [None, self.config.max_length], name="input_B")
            
            self.sentence_vector_diff = tf.placeholder(tf.float32, shape=[None, 1024], name="sentence_vector_diff")
            self.extra_features = tf.placeholder(tf.float32, shape=[None, self.config.extra_features_dim], name="extra_features")
            
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            
            self.labels = tf.placeholder(tf.float32, [None], name="labels")
            
            
        cur_batch_length = tf.shape(self.input_A)[0]

        # Define learning rate and optimizer
        learning_rate = tf.train.exponential_decay(self.config.learning_rate, 
                                                   self.global_step_tensor,
                                                   decay_steps=30000, 
                                                   decay_rate=0.96,
                                                   staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # Embedding layer
        with tf.variable_scope("embedding"):
            embeddings = tf.Variable(get_embeddings(self.config.vocab_list, 
                                                    self.config.pretrained_embed_dir, 
                                                    self.config.vocab_size, 
                                                    self.config.embed_dim),
                                     trainable=True, 
                                     name="embeddings")
            A_embedded = tf.nn.embedding_lookup(embeddings, self.input_A, name="A_embedded")
            B_embedded = tf.nn.embedding_lookup(embeddings, self.input_B, name="B_embedded")
            A_embedded, B_embedded = tf.cast(A_embedded, tf.float32), tf.cast(B_embedded, tf.float32)

        with tf.variable_scope("similarity_matrix"):
            transposed_B = tf.transpose(B_embedded, [0, 2, 1])
            similarity_matrix = tf.matmul(A_embedded, transposed_B)
            similartiy_matrix_expanded = tf.expand_dims(similarity_matrix, -1)
            
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            filters = [3, 5]
            num_filters = 20
            for i, filter_size in enumerate(filters):
                with tf.name_scope("conv_maxpool_{}".format(filter_size)):
                    # Convolution Layer
                    filter_shape = [filter_size, self.config.max_length, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        similartiy_matrix_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.config.max_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
                    
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, len(filters)*num_filters])
            self.similarity_features = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        
        self.feature_vector = tf.concat([self.sentence_vector_diff, self.similarity_features, self.extra_features], 1)
            
        # Predict a response
        with tf.variable_scope("prediction") as vs:
            dense_output = tf.layers.dense(self.feature_vector, 128)
            self.logits = tf.layers.dense(dense_output, 1)

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.expand_dims(self.labels, -1))
            self.loss = tf.reduce_mean(losses)
            self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        # Calculate accuracy
        with tf.name_scope("score"):
            # Apply sigmoid to convert logits to probabilities
            self.probs = tf.sigmoid(self.logits)
            self.predictions = tf.cast(self.probs > 0.5, dtype=tf.int32)
            correct_predictions = tf.equal(self.predictions, tf.to_int32(tf.expand_dims(self.labels, -1)))
            self.score = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def val(self, sess, feed_dict=None):
        loss = sess.run(self.loss, feed_dict=feed_dict)
        score = sess.run(self.score, feed_dict=feed_dict)
        return loss, score, None

    def infer(self, sess, feed_dict=None):
        return sess.run(self.predictions, feed_dict=feed_dict)