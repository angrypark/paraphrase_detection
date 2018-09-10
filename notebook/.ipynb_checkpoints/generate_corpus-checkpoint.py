import os
import sys
# load model
import tensorflow as tf
import numpy as np
import pandas as pd
import ahocorasick
import time

sys.path.append('/home/angrypark/similar_sentence_candidate')
sys.path.append("/home/angrypark/ml/")
from dataset import Dataset, Vectorizer
from model import model_mapper
import hparams_utils
import misc_utils as utils

process_id = 1
verbose = 20000
batch_size = 128

os.environ['CUDA_VISIBLE_DEVICES'] = str(process_id%2)
model_dir = '/media/scatter/scatterdisk/8000_rnn256_l21e-5_negdistortion0.75_lr1e-4/'
ckpt_path = os.path.join(model_dir, 'best_ROAUC', 'best_ROAUC.ckpt-820000')

base_pattern = "/media/scatter/scatterdisk/reply_matching_model/sol.raw_{}.txt"
target_pattern = "/media/scatter/scatterdisk/reply_matching_model/sol.scored_{}.txt"

problem_files_dict = {"sol.scored_1.txt" : 6147763,
"sol.scored_2.txt" : 22438994,
"sol.scored_3.txt" : 12904329,
"sol.scored_4.txt" : 11108660,
"sol.scored_5.txt" : 4032689,
"sol.scored_6.txt" : 22571455,
"sol.scored_7.txt" : 21582304,
"sol.scored_8.txt" : 653920,
"sol.scored_9.txt" : 2267014,
"sol.scored_10.txt" : 22033600,
"sol.scored_11.txt" : 692324,
"sol.scored_12.txt" : 5070643}

def infer_batch(self, send_sentences, recv_sentences):
    instances = []
    for send, recv in zip(send_sentences, recv_sentences):
        send_tokens = self.vectorizer.vectorize(send)
        recv_tokens = self.vectorizer.vectorize(recv)
        if self.add_eos:
            send_tokens += [self.vectorizer.EOS]
            recv_tokens += [self.vectorizer.EOS]
        instances.append({'send_tokens': send_tokens,
                          'recv_tokens': recv_tokens})

    return self._batchify(instances, num_negative=0)

def load_pipeline():
    import sys
    sys.path.append('/home/shuuki4/ml')
    from pingpong.utils import get_ingredient_factory
    factory = get_ingredient_factory()
    pipeline = factory.get_tokenized_pipeline()
    return pipeline

def main():
    # Load pretrained model
    hparams = hparams_utils.load_hparams(model_dir)

    dataset = Dataset(
        '/media/scatter/scatterdisk/8000_rnn256_l21e-5_negdistortion0.75_lr1e-4/8000_vocab.txt',
        add_eos=True,
        num_negative=4
    )
    graph = tf.Graph()
    with graph.as_default():
        inputs = dataset.build_placeholder()
        model = model_mapper(hparams.model_name)(inputs, hparams, tf.contrib.learn.ModeKeys.INFER)
        sess = utils.get_session(graph)
        model.saver.restore(sess, ckpt_path)
    Dataset.infer_batch = infer_batch
    print("[Process {}] Loaded Model".format(process_id))
    
    pipeline = load_pipeline()
    print("[Process {}] Loaded Pipeline".format(process_id))
    
    total_length = 40966006
        
    with open(base_pattern.format(process_id), "r") as f1, open(target_pattern.format(process_id), "a") as f2:
        restart_num = problem_files_dict["sol.scored_{}.txt".format(process_id)]
        start_time = time.time()
        queries_batch = list()
        recvs_batch = list()
        corpus_batch = list()
        i = restart_num
        
        for _ in range(restart_num):
            line = f1.readline()
            
        for line in f1:
            splits = line.split("\t")
            i += 1
            if len(splits)!=3:
                continue
            queries_batch.append(splits[1])
            recvs_batch.append(splits[2])
            corpus_batch.append(splits[0])

            if i % batch_size ==0:
                tokenized_queries_batch = [pipeline.run(q) for q in queries_batch]
                tokenized_recvs_batch = [pipeline.run(r.strip()) for r in recvs_batch]
                feed_dict = utils.feedify(inputs, dataset.infer_batch(tokenized_queries_batch, tokenized_recvs_batch))
                scores_batch = [float(s) for s in list(model.val(sess, feed_dict=feed_dict)['scores'].squeeze())]
                for corpus, query, recv, score in zip(corpus_batch, queries_batch, recvs_batch, scores_batch):
                    if score>2:
                        f2.write(corpus + "\t" + query + "\t" + recv)
                queries_batch = list()
                recvs_batch = list()
                corpus_batch = list()

            if i % verbose == 0:
                now_time = time.time()
                print("[Process {}] {:>5d} / {} ({:.2f}%, {:.2f} iter/sec)".format(process_id, 
                                                                      i,
                                                                      total_length,
                                                                      i/total_length*100,
                                                                      i/(now_time-start_time)))

if __name__ == "__main__":
    main()
    