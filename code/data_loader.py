import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
import os
import editdistance
import sys 

from preprocessor import DynamicPreprocessor
from utils.dirs import create_dirs
from utils.logger import SummaryWriter
from utils.config import load_config, save_config
from models.base import get_model
from utils.utils import JamoProcessor
from text.tokenizers import SentencePieceTokenizer

class ParaphraseDataGenerator:
    def __init__(self, preprocessor, config):
        self.preprocessor = preprocessor
        
        # get size of train and validataion set
        self.train_size = 300000 + 19312
        self.val_size = 15134
        
        # data config
        self.train_dir = config.train_dir
        self.val_dir = config.val_dir
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.num_epochs = config.num_epochs
        
        # get length
        self.length_dict = {'sol.tokenized_1.txt': 25433531,
                            'sol.tokenized_10.txt': 22033600,
                            'sol.tokenized_11.txt': 24918468,
                            'sol.tokenized_12.txt': 24189168,
                            'sol.tokenized_2.txt': 22438994,
                            'sol.tokenized_3.txt': 28275733,
                            'sol.tokenized_4.txt': 30539388,
                            'sol.tokenized_5.txt': 23822359,
                            'sol.tokenized_6.txt': 22571455,
                            'sol.tokenized_7.txt': 21582304,
                            'sol.tokenized_8.txt': 23153158,
                            'sol.tokenized_9.txt': 29596797,
                            'sol.validation.txt': 219686, 
                            'sol.small.txt': 1000, 
                            "train.txt": 300000, 
                            "val.txt": 19312, 
                            "test.txt": 15134}
        
        self.pretrained_model, self.pretrained_sess = self.get_pretrained_model()
        self.feature_extractor = FeatureExtractor()
        
    def get_pretrained_model(self):
        NAME = "delstm_1024_nsrandom4_lr1e-3"
        TOKENIZER = "SentencePieceTokenizer"
        base_dir = "/media/scatter/scatterdisk/reply_matching_model/runs/{}/".format(NAME)
        config_dir = base_dir + "config.json"
        best_model_dir = base_dir + "best_loss/best_loss.ckpt"
        
        model_config = load_config(config_dir)
        model_config.add_echo = False
        graph = tf.Graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with graph.as_default():
            Model = get_model(model_config.model)
            data = DataGenerator(self.preprocessor, model_config)
            infer_model = Model(data, model_config)
            infer_sess = tf.Session(config=tf_config, graph=graph)
            infer_sess.run(tf.global_variables_initializer())
            infer_sess.run(tf.local_variables_initializer())

        infer_model.load(infer_sess, model_dir=best_model_dir)
        return infer_model, infer_sess
    
    def get_train_iterator(self, batch_size):
        fnames = [fname for fname in sorted(os.listdir(self.train_dir)) if "validation" not in fname]
        while True:
            for fname in fnames:
                with open(os.path.join(self.train_dir, fname), "r") as f:
                    lines = [line.strip() for line in f]
                    length = self.length_dict[fname]
                    num_batches_per_file = (length-1)//batch_size + 1
                    A, B, labels = split_data(lines)
                    extracted_features = self.feature_extractor.extract_features(A, B)
                    A, A_lengths = zip(*[self.preprocessor.preprocess(line) for line in A])
                    B, B_lengths = zip(*[self.preprocessor.preprocess(line) for line in B])

                    for batch_num in range(num_batches_per_file):
                        start = batch_num*batch_size
                        end = min((batch_num+1)*batch_size, length)
                        batch_A, batch_B = A[start:end], B[start:end]
                        batch_A_lengths, batch_B_lengths = A_lengths[start:end], B_lengths[start:end]
                        batch_sentence_diff = self.get_sentence_diff(self.pretrained_model, self.pretrained_sess, batch_A, batch_B, batch_A_lengths, batch_B_lengths)
                        batch_extra_features = extracted_features[start:end]

                        yield batch_A, batch_B, batch_sentence_diff, batch_extra_features, labels[start:end]
                    
    def get_val_iterator(self, batch_size):
        fname = os.path.basename(self.val_dir)
        with open(self.val_dir, "r") as f:
            lines = [line.strip() for line in f]
            length = self.length_dict[fname]
            num_batches_per_file = (length-1)//batch_size + 1
            A, B, labels = split_data(lines)
            extracted_features = self.feature_extractor.extract_features(A, B)
            A, A_lengths = zip(*[self.preprocessor.preprocess(line) for line in A])
            B, B_lengths = zip(*[self.preprocessor.preprocess(line) for line in B])

            for batch_num in range(num_batches_per_file):
                start = batch_num*batch_size
                end = min((batch_num+1)*batch_size, length)
                batch_A, batch_B = A[start:end], B[start:end]
                batch_A_lengths, batch_B_lengths = A_lengths[start:end], B_lengths[start:end]
                batch_sentence_diff = self.get_sentence_diff(self.pretrained_model, self.pretrained_sess, batch_A, batch_B, batch_A_lengths, batch_B_lengths)
                extra_features = extracted_features[start:end]

                yield batch_A, batch_B, batch_sentence_diff, extra_features, labels[start:end]
                
    def get_sentence_diff(self, model, sess, A, B, A_lengths, B_lengths):
        feed_dict = {model.input_queries: A,
                     model.input_replies: B,
                     model.queries_lengths: A_lengths,
                     model.replies_lengths: B_lengths, 
                     model.dropout_keep_prob: 1}
        A_sentence_vector = sess.run(model.encoding_queries, feed_dict=feed_dict)
        feed_dict = {model.input_queries: B,
                     model.input_replies: A,
                     model.queries_lengths: B_lengths,
                     model.replies_lengths: A_lengths, 
                     model.dropout_keep_prob: 1}
        B_sentence_vector = sess.run(model.encoding_queries, feed_dict=feed_dict)
        return A_sentence_vector - B_sentence_vector
    

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.jamo_processor = JamoProcessor()
    
    def tokens_diff(self, a, b):
        a_tokens = a.split(" ")
        b_tokens = b.split(" ")
        return len(set(a_tokens) & set(b_tokens)) / max(len(a_tokens), len(b_tokens))

    def edit_distance(self, a, b):
        a_jamos = self.jamo_processor.word_to_jamo(a).replace("_", "")
        b_jamos = self.jamo_processor.word_to_jamo(b).replace("_", "")
        return editdistance.eval(a_jamos, b_jamos)
    
    def extract_features(self, A, B):
        extracted_features = list()
        for a, b in zip(A, B):
            ls = [self.tokens_diff(a, b), self.edit_distance(a, b)]
            extracted_features.append(ls)
        return extracted_features
        

class DataGenerator:
    def __init__(self, preprocessor, config):
        # get size of train and validataion set
        self.train_size = 298554955
        with open(config.val_dir, "r") as f:
            self.val_size = sum([1 for line in f])
            
        # data config
        self.train_dir = config.train_dir
        self.val_dir = config.val_dir
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.num_epochs = config.num_epochs
            
    def get_train_iterator(self, index_table):
        train_files = [os.path.join(self.train_dir, fname) 
                       for fname in sorted(os.listdir(self.train_dir)) 
                       if "validation" not in fname]
        
        train_set = tf.data.TextLineDataset(train_files)
        train_set = train_set.map(lambda line: parse_single_line(line, index_table, self.max_length),
                                  num_parallel_calls=8)
        train_set = train_set.shuffle(buffer_size=10000)
        train_set = train_set.batch(self.batch_size)
        train_set = train_set.repeat(self.num_epochs)
        
        train_iterator = train_set.make_initializable_iterator()
        return train_iterator
        
    def get_val_iterator(self, index_table):
        val_set = tf.data.TextLineDataset(self.val_dir)
        val_set = val_set.map(lambda line: parse_single_line(line, index_table, self.max_length),
                              num_parallel_calls=2)
        val_set = val_set.shuffle(buffer_size=1000)
        val_set = val_set.batch(self.batch_size)

        val_iterator = val_set.make_initializable_iterator()
        return val_iterator
            
    def load_test_data(self):
        base_dir = "/home/angrypark/reply_matching_model/data/"
        with open(os.path.join(base_dir, "test_queries.txt"), "r") as f:
            test_queries = [line.strip() for line in f]
        with open(os.path.join(base_dir, "test_replies.txt"), "r") as f:
            replies_set = [line.strip().split("\t") for line in f]
        with open(os.path.join(base_dir, "test_labels.txt"), "r") as f:
            test_labels = [[int(y) for y in line.strip().split("\t")] for line in f]

        test_queries, test_queries_lengths = zip(*[self.preprocessor.preprocess(query)
                                                         for query in test_queries])
        test_replies = list()
        test_replies_lengths = list()
        for replies in replies_set:
            r, l = zip(*[self.preprocessor.preprocess(reply) for reply in replies])
            test_replies.append(r)
            test_replies_lengths.append(l)
        return test_queries, test_replies, test_queries_lengths, test_replies_lengths, test_labels
        
        
def parse_single_line(line, index_table, max_length):
    """get single line from train set, and returns after padding and indexing
    :param line: corpus id \t query \t reply
    """
    splited = tf.string_split([line], delimiter="\t")
    query = tf.concat([["<SOS>"], tf.string_split([splited.values[1]], delimiter=" ").values, ["<EOS>"]], axis=0)
    reply = tf.concat([["<SOS>"], tf.string_split([splited.values[2]], delimiter=" ").values, ["<EOS>"]], axis=0)
    
    paddings = tf.constant([[0, 0],[0, max_length]])
    padded_query = tf.slice(tf.pad([query], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
    padded_reply = tf.slice(tf.pad([reply], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
    
    indexed_query = tf.squeeze(index_table.lookup(padded_query))
    indexed_reply = tf.squeeze(index_table.lookup(padded_reply))
    
    return indexed_query, indexed_reply, tf.shape(query)[0], tf.shape(reply)[0]

def split_data(data):
    A, B, labels = zip(*[line.split('\t') for line in data])
    labels = [1 if l=="1" else 0 for l in labels]
    return A, B, labels