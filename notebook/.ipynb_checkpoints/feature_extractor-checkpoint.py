import tensorflow as tf
import numpy as np
import os
import sys
from collections import namedtuple
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import editdistance

sys.path.append("/home/angrypark/korean-text-matching-tf")

from data_loader import DataGenerator
from trainer import MatchingModelTrainer
from preprocessor import DynamicPreprocessor
from utils.dirs import create_dirs
from utils.logger import SummaryWriter
from utils.config import load_config, save_config
from models.base import get_model
from utils.utils import JamoProcessor
from text.tokenizers import SentencePieceTokenizer

Config = namedtuple("config", ["sent_piece_model"])
config = Config("/media/scatter/scatterdisk/tokenizer/sent_piece.100K.model")
processor = JamoProcessor()
tokenizer = SentencePieceTokenizer(config)

def my_word_tokenizer(raw, pos=["Noun", "Alpha", "Verb", "Number"], stopword=[]):
    return [word for word in tokenizer.tokenize(raw)]

def my_char_tokenizer(raw, pos=["Noun", "Alpha", "Verb", "Number"], stopword=[]):
    return [processor.word_to_jamo(word) for word in tokenizer.tokenize(raw)]

def proper_edit_distance(a_jamos, b_jamos):
    long_length = max([len(a_jamos), len(b_jamos)])
    edit_distance = editdistance.eval(a_jamos, b_jamos) / long_length
    return edit_distance

def substring(a_jamos, b_jamos):
    long_length = max([len(a_jamos), len(b_jamos)])
    match = SequenceMatcher(None, a_jamos, b_jamos).find_longest_match(0, len(a_jamos), 0, len(b_jamos))
    return match.size / long_length

class FeatureExtractor:
    def __init__(self):
        self.infer_model, self.infer_sess = self._load_pretrained_model()
        self.tfidf_char_vectorizer = pickle.load(open("../dump/tfidf_char_vectorizer.pkl", "rb"))
        self.tfidf_word_vectorizer = pickle.load(open("../dump/tfidf_word_vectorizer_big.pkl", "rb"))
        self.processor = JamoProcessor()
        self.tokenizer = SentencePieceTokenizer(config)
        
    def _load_pretrained_model(self):
        base_dir = "/media/scatter/scatterdisk/reply_matching_model/runs/delstm_1024_nsrandom4_lr1e-3/"
        config_dir = base_dir + "config.json"
        best_model_dir = base_dir + "best_loss/best_loss.ckpt"
        model_config = load_config(config_dir)
        model_config.add_echo = False
        preprocessor = DynamicPreprocessor(model_config)
        preprocessor.build_preprocessor()

        infer_config = load_config(config_dir)
        setattr(infer_config, "tokenizer", "SentencePieceTokenizer")
        setattr(infer_config, "soynlp_scores", "/media/scatter/scatterdisk/tokenizer/soynlp_scores.sol.100M.txt")
        infer_preprocessor = DynamicPreprocessor(infer_config)
        infer_preprocessor.build_preprocessor()
        graph = tf.Graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with graph.as_default():
            Model = get_model(model_config.model)
            data = DataGenerator(preprocessor, model_config)
            infer_model = Model(data, model_config)
            infer_sess = tf.Session(config=tf_config, graph=graph)
            infer_sess.run(tf.global_variables_initializer())
            infer_sess.run(tf.local_variables_initializer())

        infer_model.load(infer_sess, model_dir=best_model_dir)
        self.infer_preprocessor = infer_preprocessor
        return infer_model, infer_sess
        
    def _batch_infer(self, batch_A, batch_B):
        indexed_A, A_lengths = zip(*[self.infer_preprocessor.preprocess(a) for a in batch_A])
        indexed_B, B_lengths = zip(*[self.infer_preprocessor.preprocess(b) for b in batch_B])
        
        feed_dict = {self.infer_model.input_queries: indexed_A,
             self.infer_model.input_replies: indexed_B,
             self.infer_model.queries_lengths: A_lengths,
             self.infer_model.replies_lengths: B_lengths,
             self.infer_model.dropout_keep_prob: 1, 
             }
        A_sentence_vectors, AB_probs = self.infer_sess.run([self.infer_model.encoding_queries, 
                                                            self.infer_model.positive_probs], 
                                                            feed_dict=feed_dict)

        feed_dict = {self.infer_model.input_queries: indexed_B,
             self.infer_model.input_replies: indexed_A,
             self.infer_model.queries_lengths: B_lengths,
             self.infer_model.replies_lengths: A_lengths,
             self.infer_model.dropout_keep_prob: 1, 
             }
        B_sentence_vectors, BA_probs = self.infer_sess.run([self.infer_model.encoding_queries, 
                                                            self.infer_model.positive_probs], 
                                                            feed_dict=feed_dict)

        semantic_sim = [cosine_similarity([a_vector], [b_vector])[0][0] for a_vector, b_vector 
                        in zip(list(A_sentence_vectors), list(B_sentence_vectors))]
        return [p[0] for p in AB_probs], [p[0] for p in BA_probs], semantic_sim
    
    def extract_features(self, sentences_A, sentences_B):
        def get_semantic_sim(A, B, batch_size=512):
            length = len(A)
            num_batches = (length - 1) // batch_size + 1
    
            result = {"ab_probs": list(), "ba_probs": list(), "semantic_sim": list()}
            for batch_num in range(num_batches):
                start = batch_num * batch_size
                end = min([(batch_num+1) * batch_size, length])
                
                ab_probs, ba_probs, semantic_sim = self._batch_infer(A[start:end], B[start:end])
                result["ab_probs"] += list(ab_probs)
                result["ba_probs"] += list(ba_probs)
                result["semantic_sim"] += semantic_sim
            return result
        
        def get_word_tfidf_sim(A, B):
            word_sim = list()
            for a, b in zip(A, B):
                word_sim.append(cosine_similarity(self.tfidf_word_vectorizer.transform([a]), 
                                                  self.tfidf_word_vectorizer.transform([b]))[0][0])
            return {"tfidf_word_sim": word_sim}
                
        def get_char_tfidf_sim(A, B):
            char_sim = list()
            for a, b in zip(A, B):
                char_sim.append(cosine_similarity(self.tfidf_char_vectorizer.transform([a]), 
                                                  self.tfidf_char_vectorizer.transform([b]))[0][0])
            return {"tfidf_char_sim": char_sim}
            
        def get_edit_distance(A, B):
            edit_distance = list()
            substring_ratio = list()
            for a, b in zip(A, B):
                a_jamos = self.processor.word_to_jamo(a).replace("_", "")
                b_jamos = self.processor.word_to_jamo(b).replace("_", "")
                edit_distance.append(proper_edit_distance(a_jamos, b_jamos))
                substring_ratio.append(substring(a_jamos, b_jamos))
            return {"edit_distance": edit_distance, 
                    "substring_ratio": substring_ratio}
        
        extracted_features = dict()
        extracted_features.update(get_semantic_sim(sentences_A, sentences_B, batch_size=512))
        extracted_features.update(get_word_tfidf_sim(sentences_A, sentences_B))
        extracted_features.update(get_char_tfidf_sim(sentences_A, sentences_B))
        extracted_features.update(get_edit_distance(sentences_A, sentences_B))
        
        return extracted_features
