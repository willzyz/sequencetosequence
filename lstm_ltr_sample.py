## Implementation of lstm learning-to-rank model with Tensorflow. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import functools

import numpy as np
import tensorflow as tf
import os
import random
import load_data
import time
import logging
from datetime import datetime
from util import clean_output, latest_checkpoint, print_model

learn = tf.contrib.learn
FLAGS = None

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 50
NEGATIVE_OFFER_NUMBER = 10
query_word_number = 0
offer_word_number = 0
offer_size = 0
batch_size = 128
query_vocab_processor = None
offer_vocab_processor = None

data_dir = None
model_dir = None

# Input filename
click_file = "sample_clicklog.tsv"
query_file = "QueryTokenized.tsv"
offer_file = "OfferTokenized.tsv"

eval_click_file = "testdata.tsv"
eval_query_file = "test_queries_tokenized.tsv"
eval_offer_file = "test_offers_tokenized.tsv"

# Output filename
query_vocab_processor_file = "query_vocab_processor.txt"
offer_vocab_processor_file = "offer_vocab_processor.txt"

logger = None
use_print = False


def print_message(msg):
    """Print log messages."""
    if use_print is False:
        logger.info(msg)
    else:
        print(msg)


def parse_flags():
    """Parse all input parameters"""
    global logger
    global use_print
    global data_dir, model_dir
    global click_file, query_file, offer_file
    global eval_click_file, eval_query_file, eval_offer_file

    use_print = FLAGS.use_print
    data_dir = FLAGS.data_dir
    model_dir = FLAGS.model_dir

    log_path = os.path.join(FLAGS.log_dir, "DeepIntent_ltr.log")
    logging.basicConfig(filename=log_path,  level=logging.DEBUG)
    # logging.config.fileConfig("DeepIntent_ltr.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    #formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    #ch.setFormatter(formatter)
    logger.addHandler(ch)

    print_message("Parameters:" +
                  " --log_dir=" + str(FLAGS.log_dir) +
                  " --use_print=" + str(use_print) +
                  " --data_dir=" + str(data_dir) +
                  " --model_dir=" + str(model_dir) +
                  " --log_dir=" + str(FLAGS.log_dir) +
                  " --large_dataset=" + str(FLAGS.large_dataset)
                  )
    if FLAGS.large_dataset is False:
        print_message("FLAGS.large_dataset is False")
        prefix = "small_"
        click_file = prefix + click_file
        query_file = prefix + query_file
        offer_file = prefix + offer_file

        eval_click_file = prefix + eval_click_file
        eval_query_file = prefix + eval_query_file
        eval_offer_file = prefix + eval_offer_file
    else:
        print_message("FLAGS.large_dataset is True")


def get_cell_fn(cell_type='gru'):
    """Choose a proper RNN cell function from cell_type."""
    cell_fn = tf.contrib.rnn.GRUCell
    if cell_type == 'rnn':
        cell_fn = tf.contrib.rnn.BasicRNNCell
    elif cell_type == 'gru':
        cell_fn = tf.contrib.rnn.GRUCell
    elif cell_type == 'lstm':
        cell_fn = functools.partial(
            tf.contrib.rnn.BasicLSTMCell, state_is_tuple=True)
    return cell_fn


def bidirectional_rnn(cell, inputs, dtype=tf.float32):
    """Bidirectional RNN models"""
    # Creates a bidirectional recurrent neural network
    #  Adopts tensorflow API: tf.contrib.rnn.static_bidirectional_rnn
    # refer the following url for details:
    # https://www.tensorflow.org/versions/r1.0/api_docs/python/contrib.rnn/recurrent_neural_networks
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        cell, cell, inputs, dtype=dtype)
    return outputs


def rnn_model_layer_query(features, n_words):
    """RNN model layer"""
    # Convert indexes of words into embeddings.
    #  This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    #  maps word indexes of the sequence into [batch_size, sequence_length, EMBEDDING_SIZE].

    # Maps a sequence of symbols to a sequence of embeddings,
    # refer the following url for details:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/encoders.py
    with tf.variable_scope("query_words"):
        print_message("Features size: " + str(features.get_shape()))  # [batch_size, MAX_DOCUMENT_LENGTH], [100, 10]
        word_vectors = tf.contrib.layers.embed_sequence(
            features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
        print_message("Word_vectors size: " + str(word_vectors.get_shape()))
        # [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE], [100, 10, 50]

        # For debug, output word vectors
        if FLAGS.debug:
            word_vectors = tf.Print(word_vectors, [word_vectors], first_n=1,
                                    summarize=n_words*EMBEDDING_SIZE, message="Word_vectors: ")

        # Split into list of embedding per word, while removing doc length dim.
        # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
        word_list = tf.unstack(word_vectors, axis=1)
        print_message("Word_list size: %d" % len(word_list))  # Size: MAX_DOCUMENT_LENGTH, 10

        if FLAGS.debug:
            for i in range(len(word_list)):
                print_message("Id: " + str(i) + " word_list size: " + str(word_list[i].get_shape()))

        # Create a RNN cell with hidden size of EMBEDDING_SIZE.
        # cell_fn_name can be one of {'rnn', 'gru', 'lstm'}
        cell_fn_name = 'lstm'
        cell_fn = get_cell_fn(cell_fn_name)
        cell = cell_fn(EMBEDDING_SIZE)

        # if FLAGS.attention:
        #  attn_length = MAX_DOCUMENT_LENGTH
        #  attn_size = None
        #  attn_vec_size = None
        #  cell = tf.contrib.rnn.AttentionCellWrapper(
        #    cell, attn_length=attn_length, attn_size=attn_size,
        #    attn_vec_size=attn_vec_size, state_is_tuple=False)

        # Create an unrolled Recurrent Neural Networks to length of
        # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
        # if FLAGS.bidirectionalRNN:
        #  outputs = bidirectionalRNN(cell, word_list, dtype=tf.float32)
        # else:
        #  # Refer the following url for details:
        #  # https://www.tensorflow.org/versions/master/api_docs/python/contrib.rnn/recurrent_neural_networks#static_rnn
        outputs, _ = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)
        #  #_, encoding = tf.nn.rnn_cell.static_rnn(cell, word_list, dtype=tf.float32)

        # mean pooling
        # encoding = tf.reduce_mean(outputs, 0)
        # last pooling
        encoding = outputs[-1]
        print_message("query encoding size: " + str(encoding.get_shape()))  # [batch_size, EMBEDDING_SIZE] [100, 50]
        return encoding


def rnn_model_layer_offer(features, n_words):
    """RNN model layer"""
    with tf.variable_scope("offer_words"):
        word_vectors = tf.contrib.layers.embed_sequence(
            features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='offer_words')

        # For debug, output word vectors
        if FLAGS.debug:
            word_vectors = tf.Print(word_vectors, [word_vectors], first_n=1,
                                    summarize=n_words*EMBEDDING_SIZE, message="Word_vectors: ")

        word_list = tf.unstack(word_vectors, axis=1)
        cell_fn_name = 'lstm'
        cell_fn = get_cell_fn(cell_fn_name)
        cell = cell_fn(EMBEDDING_SIZE)
        outputs, _ = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)
        encoding = tf.reduce_mean(outputs, 0)
        print_message("offer encoding size: " + str(encoding.get_shape()))
        # [batch_size*(NEGATIVE_OFFER_NUMBER+1), EMBEDDING_SIZE] [100*11, 50]
        return encoding


def normalize_tensor(data_tensor):
    # Normalize the data_tensor to [0,1]
    # i.e., normalize_value = (value − min_value) / (max_value − min_value)
    data_tensor = tf.div(
        tf.subtract(
            data_tensor,
            tf.reduce_min(data_tensor)
        ),
        tf.subtract(
            tf.reduce_max(data_tensor),
            tf.reduce_min(data_tensor)
        )
    )
    data_tensor = tf.clip_by_value(data_tensor, 1e-10, 1.0-1e-10)
    return data_tensor


def calculate_cosine_sim(query_encodings, offer_encodings, name_scope="Cosine_Similarity"):
    # Calculate the loss for evaluation
    with tf.name_scope(name_scope):
        print_message("Query_encoding size: " + str(query_encodings.get_shape()))  # [None, 50]
        print_message("Offer_encodings size: " + str(offer_encodings.get_shape()))  # [None, 50]
        normed_query_encodings = tf.nn.l2_normalize(query_encodings, dim=1)
        normed_offer_encodings = tf.nn.l2_normalize(offer_encodings, dim=1)

        cosine_similarity = tf.multiply(normed_query_encodings, normed_offer_encodings)
        print_message("Cosine_similarity size: " + str(cosine_similarity.get_shape()))  # [1100, 50]
        cosine_similarity = tf.reduce_sum(cosine_similarity, axis=1, keep_dims=False)
        print_message("Reduced cosine_similarity size: " + str(cosine_similarity.get_shape()))  # [1, 11]
        cosine_similarity = normalize_tensor(cosine_similarity)
        return cosine_similarity


def train_cosine_sim(query_encodings, offer_encodings):
    # Calculate the cosine similarity between queries and offers
    # One query corresponds to NEGATIVE_OFFER_NUMBER+1 offers,
    # i.e., 1 positive offer, NEGATIVE_OFFER_NUMBER negative offers
    # Steps:
    # 1. expands the dimension of query_encodings to the same dimension of offer_encodings
    # 2. generate cosine similarity between queries and offers

    with tf.name_scope('Cosine_Similarity'):
        print_message("query_encoding size: " + str(query_encodings.get_shape()))  # [100, 50]
        print_message("offer_encodings size: " + str(offer_encodings.get_shape()))  # [100*11, 50]
        expanded_query_encodings = tf.tile(query_encodings, [1, NEGATIVE_OFFER_NUMBER+1])
        expanded_query_encodings = tf.reshape(expanded_query_encodings, [-1, EMBEDDING_SIZE])
        cosine_similarity = calculate_cosine_sim(expanded_query_encodings,
                                                 offer_encodings, name_scope="Training_Cosine_Similarity")
        # 1 dimension to 2 dimensions,
        # i.e., [batch_size*(NEGATIVE_OFFER_NUMBER+1)] to [batch_size, NEGATIVE_OFFER_NUMBER+1]
        cosine_similarity = tf.reshape(cosine_similarity, [-1, NEGATIVE_OFFER_NUMBER+1])
        return cosine_similarity


def read_data(filename):
    # Read data from data_dir
    data_path = os.path.join(data_dir, filename)
    data_sentences = load_data.tf_read_text(data_path)

    avg_len = load_data.tf_calculate_average_length(data_sentences)

    rnn_length = avg_len * 2
    return data_sentences, rnn_length


def transform_to_word_id_vector(data_sentences, is_train=True, is_query=True, rnn_length=None, is_chief=False):
    global query_vocab_processor
    global offer_vocab_processor
    # Transform words vector to word-ids vector
    # Different between fit_transform() and transform():
    #   fit_transform() needs an extra operation to fit into the length of RNN_length
    #   They both transform words vector to word-ids vector
    if is_train is True:
        # Process vocabulary
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/preprocessing/text.py
        if is_query is True:
            query_vocab_processor = learn.preprocessing.VocabularyProcessor(rnn_length)
            data_sentences = np.array(list(query_vocab_processor.fit_transform(data_sentences)))
            if is_chief:
                query_vocab_processor.save(os.path.join(FLAGS.model_dir, query_vocab_processor_file))
        else:
            offer_vocab_processor = learn.preprocessing.VocabularyProcessor(rnn_length)
            data_sentences = np.array(list(offer_vocab_processor.fit_transform(data_sentences)))
            if is_chief:
                offer_vocab_processor.save(os.path.join(FLAGS.model_dir, offer_vocab_processor_file))
    else:
        if is_query is True:
            query_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
                os.path.join(FLAGS.model_dir, query_vocab_processor_file)
            )
            data_sentences = np.array(list(query_vocab_processor.transform(data_sentences)))
        else:
            offer_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
                os.path.join(FLAGS.model_dir, offer_vocab_processor_file)
            )
            data_sentences = np.array(list(offer_vocab_processor.transform(data_sentences)))

    vocab_processor = None
    if is_query is True:
        vocab_processor = query_vocab_processor
    else:
        vocab_processor = offer_vocab_processor
    # x_test = np.array(list(vocab_processor.transform(x_test)))
    n_words = len(vocab_processor.vocabulary_)
    print_message('Total words: %d' % n_words)
    if FLAGS.debug:
        # Given word id, output word
        for i in range(30):
            print_message(vocab_processor.vocabulary_.reverse(i))
    return data_sentences, n_words


def get_click_data(filename):
    # get click data
    data_path = os.path.join(data_dir, filename)
    click_pairs = np.loadtxt(data_path)
    clicks_q = click_pairs[:, 0].astype('int32') - 1
    clicks_o = click_pairs[:, 1].astype('int32') - 1
    clicks_count = click_pairs[:, 2].astype('float')
    # clicks = load_data.read_clicks(n_queries, n_offers, datapath)
    return clicks_q, clicks_o, clicks_count


def generate_negative_offer_ids(negative_offer_number, positive_offer_id, offer_size):
    # For each positive offer, generate [negative_offer_number] negative offers
    negative_offer_ids = []
    for i in range(negative_offer_number):
        i = random.randint(0, offer_size-2)
        if i >= positive_offer_id:
            negative_offer_ids.append(i+1)
        else:
            negative_offer_ids.append(i)
    return negative_offer_ids


def pull_batch(batch_idx, query_sentences, offer_sentences, clicks_q, clicks_o):
    # generate next batch for training
    batch_query_tensors = np.array([query_sentences[x] for x in
                                    clicks_q[batch_idx*batch_size: (batch_idx+1)*batch_size]])

    batch_offer_tensors = list()
    for i in range(batch_size):
        positive_offer_tensor = offer_sentences[clicks_o[batch_idx*batch_size + i]]
        # negative_offer_ids = generate_negative_offer_ids(
        #  NEGATIVE_OFFER_NUMBER, clicks_o[batch_idx], offer_size)
        batch_offer_tensors.append(positive_offer_tensor)

        negative_offer_ids = random.sample(range(offer_size), NEGATIVE_OFFER_NUMBER)
        for x in negative_offer_ids:
            batch_offer_tensors.append(offer_sentences[x])

        # negative_offer_tensors = [offer_sentences[x] for x in negative_offer_ids]
        # offer_tensors = np.array([positive_offer_tensor] + negative_offer_tensors)
        # batch_offer_tensors.append(offer_tensors)

    batch_offer_tensors = np.array(batch_offer_tensors)
    return batch_query_tensors, batch_offer_tensors


def load_training_dataset(is_chief=False):
    # Dataset for training and validation
    global offer_size, query_word_number, offer_word_number
    start_time = time.time()
    clicks_q, clicks_o, _ = get_click_data(click_file)
    print_message(str(datetime.now()) + " " + "Read click data finished.")
    query_origin_sentences, query_rnn_length = read_data(query_file)
    query_sentences, query_word_number = transform_to_word_id_vector(
        query_origin_sentences, is_train=True, is_query=True, rnn_length=query_rnn_length, is_chief=is_chief)
    print_message(str(datetime.now()) + " " + "Read query data finished.")
    offer_origin_sentences, offer_rnn_length = read_data(offer_file)
    offer_sentences, offer_word_number = transform_to_word_id_vector(
        offer_origin_sentences, is_train=True, is_query=False, rnn_length=offer_rnn_length, is_chief=is_chief)
    print_message(str(datetime.now()) + " " + "Read offer data finished.")

    end_time = time.time()
    print_message(str(datetime.now()) + " " + "Load training data from Disk to memory: " + str(end_time-start_time))

    offer_size = len(offer_sentences)
    query_offer_size = len(clicks_q)
    # 95% for training
    train_query_offer_size = int(0.95 * query_offer_size)
    valid_query_offer_size = query_offer_size - train_query_offer_size

    train_clicks_q, train_clicks_o = clicks_q[:train_query_offer_size], clicks_o[:train_query_offer_size]
    valid_clicks_q, valid_clicks_o = \
        clicks_q[train_query_offer_size: query_offer_size], clicks_o[train_query_offer_size: query_offer_size]

    # 5% data for validation
    if FLAGS.use_validation:
        valid_query_tensors = [query_sentences[x]
                               for pair in zip(valid_clicks_q, valid_clicks_q)
                               for x in pair]
        valid_clicks_neg_o = [random.randint(0, offer_size-1) for i in range(valid_query_offer_size)]
        valid_offer_tensors = [offer_sentences[x]
                               for pair in zip(valid_clicks_o, valid_clicks_neg_o)
                               for x in pair]
        valid_labels = np.array([1, 0] * valid_query_offer_size)
    else:
        valid_query_tensors = None
        valid_offer_tensors = None
        valid_labels = None

    return query_sentences, offer_sentences, train_clicks_q, train_clicks_o, train_query_offer_size, \
           valid_query_tensors, valid_offer_tensors, valid_labels, \
           query_rnn_length, offer_rnn_length


def load_evaluation_dataset():
    # Test dataset for evaluation
    start_time = time.time()
    eval_clicks_q, eval_clicks_o, eval_clicks_count = get_click_data(eval_click_file)
    print_message(str(datetime.now()) + " " + "Read click data finished.")
    eval_query_sentences, _ = read_data(eval_query_file)
    eval_query_sentences, eval_query_word_number = transform_to_word_id_vector(
        eval_query_sentences, is_train=False, is_query=True, rnn_length=None)
    print_message(str(datetime.now()) + " " + "Read query data finished.")
    eval_offer_sentences, _ = read_data(eval_offer_file)
    eval_offer_sentences, eval_offer_word_number = transform_to_word_id_vector(
        eval_offer_sentences, is_train=False, is_query=False, rnn_length=None)
    print_message(str(datetime.now()) + " " + "Read offer data finished.")
    end_time = time.time()

    print_message(str(datetime.now()) + " " + "Load evaluation data from Disk to memory: " + str((end_time-start_time)))

    eval_query_tensors = [eval_query_sentences[x] for x in eval_clicks_q]
    eval_offer_tensors = [eval_offer_sentences[x] for x in eval_clicks_o]

    eval_labels = np.array([1 if x > 0 else 0 for x in eval_clicks_count])

    return eval_query_tensors, eval_offer_tensors, eval_labels


def train_dist(ps_hosts, worker_hosts, process_index=0):
    # Distributed training
    # Identify role
    task_index = FLAGS.task_index * FLAGS.worker_per_host + process_index
    print_message(str(datetime.now()) + "Identify role, host worker:%d started, task_index:%d, Pid:%d"
                  % (process_index, task_index, os.getpid()))
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": {task_index: worker_hosts[task_index]}})
    is_chief = task_index == 0

    ps_count = len(cluster.job_tasks("ps"))
    workers_count = len(worker_hosts)

    query_sentences, offer_sentences, train_clicks_q, train_clicks_o, train_query_offer_size, \
    valid_query_tensors, valid_offer_tensors, valid_labels, \
    query_rnn_length, offer_rnn_length = load_training_dataset(is_chief=is_chief)

    # if is_chief:
    #     eval_query_tensors, eval_offer_tensors, eval_labels = load_evaluation_dataset()

    sess_config = tf.ConfigProto(allow_soft_placement=False,
                                 log_device_placement=FLAGS.log_device_placement,
                                 intra_op_parallelism_threads=FLAGS.wk_intra_parallel,
                                 inter_op_parallelism_threads=FLAGS.wk_inter_parallel,
                                 graph_options=tf.GraphOptions(enable_bfloat16_sendrecv=FLAGS.use_bfloat16_transfer))
    server = tf.train.Server(cluster, job_name="worker", task_index=task_index, config=sess_config)

    worker_device = "/job:worker/task:%d" % task_index

    # Create graph
    print_message(str(datetime.now()) + " " + "%s: Create graph" % str(datetime.now()))
    device_setter = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)
    with tf.device(device_setter):
        # query_tensor = tf.placeholder(tf.int32, shape=(batch_size, query_rnn_length))
        # offer_tensors = tf.placeholder(tf.int32,
        #                               shape=(batch_size * (NEGATIVE_OFFER_NUMBER+1), offer_rnn_length))
        query_tensor = tf.placeholder(tf.int32, shape=(None, query_rnn_length))
        offer_tensors = tf.placeholder(tf.int32, shape=(None, offer_rnn_length))

        query_encoding = rnn_model_layer_query(query_tensor, query_word_number)
        offer_encodings = rnn_model_layer_offer(offer_tensors, offer_word_number)

        cosine_similarity = train_cosine_sim(query_encoding, offer_encodings)

        target = tf.zeros(batch_size, tf.int32)
        target = tf.one_hot(target, (NEGATIVE_OFFER_NUMBER + 1), 1, 0)
        print_message("target size: " + str(target.get_shape()))  # [100, 11]

        loss = tf.losses.softmax_cross_entropy(onehot_labels=target, logits=cosine_similarity)
        print_message("loss size: " + str(loss.get_shape()))  # []

        global_step = tf.Variable(0, name="global_step")
        if FLAGS.use_epoch_signal:
            global_worker_count = tf.Variable(0, name="global_worker_count")
            global_worker_count_op = global_worker_count.assign_add(1, use_locking=True)
            global_worker_count_reset_op = global_worker_count.assign(0, use_locking=True)

        #train_op = tf.contrib.layers.optimize_loss(
        #    loss,
        #    tf.contrib.framework.get_global_step(),
        #    optimizer='Adam',
        #    learning_rate=0.001)
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

        if is_chief and FLAGS.use_validation:
            # Calculate AUC score for validation and evaluation
            labels = tf.placeholder(tf.bool, shape=(None))
            eval_cosine_similarity = calculate_cosine_sim(query_encoding,
                                                          offer_encodings, name_scope="Evaluation_Cosine_Similarity")
            # Computes the approximate AUC via a Riemann sum.
            # Please refer the following url for details:
            # https://www.tensorflow.org/api_docs/python/tf/contrib/metrics/streaming_auc
            eval_auc, eval_update_op_auc = tf.contrib.metrics.streaming_auc(eval_cosine_similarity, labels)

        # Initialize global and local variables
        global_initializer = tf.global_variables_initializer()
        local_initializer = tf.local_variables_initializer()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_chief,
                             init_op=global_initializer,
                             init_fn=print_model,
                             local_init_op=local_initializer,
                             logdir=FLAGS.log_dir,
                             global_step=global_step,
                             saver=None if FLAGS.save_model_secs <= 0 else saver,
                             # Disable the saver of Supervisor which separates checkpoint files cross machines!
                             save_model_secs=None if FLAGS.save_model_secs <= 0 else FLAGS.save_model_secs,
                             summary_writer=None)
    with sv.prepare_or_wait_for_session(master=server.target, config=sess_config, start_standard_services=True) as sess:
        print_message(str(datetime.now()) + " Create session success")
        local_step = 0
        sess.run(global_initializer)

        # Start to train
        batch_num = train_query_offer_size//batch_size
        for epoch in range(100):
            if FLAGS.use_epoch_signal:
                current_global_worker_count = sess.run(global_worker_count)
                while current_global_worker_count == workers_count:
                    print_message(str(datetime.now()) +
                                  "Worker %d is prepared to start epoch %d, still wait for start signal..."
                                  % (task_index, epoch))
                    time.sleep(10)
                    current_global_worker_count = sess.run(global_worker_count)

            sess.run(local_initializer)
            start_time = time.time()
            local_total_loss = 0.0
            local_start_time = time.time()
            for batch_idx in range(task_index, batch_num, workers_count):
                local_step += 1
                query_tensor1, offer_tensors1 = pull_batch(batch_idx,
                                                           query_sentences, offer_sentences,
                                                           train_clicks_q, train_clicks_o)
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={query_tensor: query_tensor1,
                                                    offer_tensors: offer_tensors1})
                if local_step % 10 == 1:
                    local_end_time = time.time()
                    print_message(str(datetime.now()) + " " + "### Local_step " + str(local_step)
                                  + ", Epoch " + str(epoch)
                                  + ", Batch_idx " + str(batch_idx) + ", total batch number " + str(batch_num)
                                  + ", loss_value " + str(loss_value)
                                  + ", cost time:" + str(local_end_time - local_start_time) + " ###")
                    local_start_time = time.time()

                local_total_loss += loss_value
            # Save model
            # model_path = os.path.join(model_dir, "model.ckpt-deepintent-epoch-" + str(epoch))
            # save_path = saver.save(sess, model_path)
            # logger.info("Model saved in file: %s" % save_path)
            end_time = time.time()
            print_message(str(datetime.now()) + " " + "Epoch " + str(epoch)
                          + " local_total_loss: " + str(local_total_loss)
                          + ", training time: " + str(end_time - start_time) + "s")

            saver.save(sess, os.path.join(FLAGS.model_dir, "model.ckpt-epoch-" + str(epoch)), global_step)
            # Call saver explicitly will output checkpint files in current machine!

            if FLAGS.use_epoch_signal:
                sess.run(global_worker_count_op)  # This worker has finished one epoch, waiting others to finish

                current_global_worker_count = sess.run(global_worker_count)
                while current_global_worker_count != workers_count or current_global_worker_count == 0:
                    # There is a potential issue: if one worker can finish one epoch in 10s,
                    # then other workers may be blocked.
                    # But workers couldn't finish one epoch in 10s.
                    print_message(str(datetime.now()) +
                                  "Worker %d finished epoch %d, current success worker count: %d"
                                  % (task_index, epoch, current_global_worker_count))
                    time.sleep(10)
                    current_global_worker_count = sess.run(global_worker_count)

            if is_chief:
                if FLAGS.use_validation:
                    sess.run(local_initializer)
                    # valid_cosine_sim = sess.run(eval_cosine_similarity,
                    #                             feed_dict={query_tensor: valid_query_tensors,
                    #                                        offer_tensors: valid_offer_tensors,
                    #                                        labels: valid_labels})
                    # print_message(str(datetime.now()) + " " + ", Epoch " + str(epoch)
                    #               + "Valid_cosine_similarity: " + str(valid_cosine_sim))
                    valid_auc = sess.run(eval_update_op_auc, feed_dict={query_tensor: valid_query_tensors,
                                                                        offer_tensors: valid_offer_tensors,
                                                                        labels: valid_labels})
                    print_message(str(datetime.now()) + " " + ", Epoch " + "Validation AUC = " + str(valid_auc))
                if FLAGS.use_epoch_signal:
                    sess.run(global_worker_count_reset_op)

                """ # Evaluation part
                sess.run(local_initializer)

                eval_cos_sim = sess.run(eval_cosine_similarity,
                                        feed_dict={query_tensor: eval_query_tensors,
                                                   offer_tensors: eval_offer_tensors,
                                                   labels: eval_labels})
                print_message(str(datetime.now()) + " " + ", Epoch "
                              + "Evaluation cosine_similarity: " + str(eval_cos_sim))
                eval_auc = sess.run(eval_update_op_auc, feed_dict={query_tensor: eval_query_tensors,
                                                                   offer_tensors: eval_offer_tensors,
                                                                   labels: eval_labels})
                print_message(str(datetime.now()) + " " + ", Epoch " + "Evaluation AUC = %g" % eval_auc)
                """


def main(unparsed):
    parse_flags()

    begin_time = time.time()

    if FLAGS.dist_in_single_machine:
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")
    else:
        ps_hosts = [ip + ":" + str(FLAGS.ps_port) for ip in FLAGS.ps_hosts.split(",")]
        worker_hosts = []
        for ip in FLAGS.worker_hosts.split(","):
            worker_hosts.extend([ip + ":" + str(FLAGS.worker_port + i) for i in range(FLAGS.worker_per_host)])

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    if FLAGS.job_name == "ps":
        # if FLAGS.save_model_secs > 0 and FLAGS.task_index == 0: # Create a backend thread to remove old checkpoint files
        #    print_message("%s: Start a backend thread to monitoring: %s" % (str(datetime.now()), FLAGS.model_dir))
        #    t_cleaner = threading.Thread(target=clean_output, args=(FLAGS.model_dir, "model.ckpt-", 2, FLAGS.save_model_secs / 2))
        #    t_cleaner.setDaemon(True)
        #    t_cleaner.start()

        # Start the parameter server
        server_config = tf.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=FLAGS.log_device_placement,
                                       graph_options=tf.GraphOptions(enable_bfloat16_sendrecv=FLAGS.use_bfloat16_transfer))
        server = tf.train.Server(cluster, job_name="ps", task_index=FLAGS.task_index, config=server_config)
        print_message("%s: ps%d: listening ..." % (str(datetime.now()), FLAGS.task_index))
        server.join()
    else:
        # Create a directory for log and model
        # if not os.path.exists(FLAGS.log_dir):
        #     os.makedirs(FLAGS.log_dir)
        # if not os.path.exists(FLAGS.model_dir):
        #     os.makedirs(FLAGS.model_dir)

        # start host master worker
        train_dist(ps_hosts, worker_hosts)
        print_message(str(datetime.now()) + " " + "Host master worker finished, exit")

    print_message(str(datetime.now()) + " " + "Finish in %.3f seconds" % (time.time() - begin_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # IO arguments
    parser.add_argument("--log_dir", type=str, default="D:/Work/RnR-Selection/DL/DL_ltr/debug",
                        help="Summaries log directory.")
    parser.add_argument("--data_dir", type=str, default="D:/Dataset/DeepIntent/all", help="File path of input data.")
    parser.add_argument("--model_dir", type=str, default="D:/Work/RnR-Selection/DL/DL_ltr/debug",
                        help="Directory path storing model files.")
    parser.add_argument("--large_dataset", action='store_true', default=False, help="Whether use small dataset.")

    # Model arguments
    parser.add_argument("--attention", action='store_true', default=False, help="Whether use attention mechanism.")
    parser.add_argument("--bidirectionalRNN", action='store_true', default=False, help='Whether use bidirectionalRNN.')
    parser.add_argument("--save_model_secs", type=int, default=0,
                        help="Time interval of saving model into checkpoint and the unit is second.")

    # Distributed arguments
    parser.add_argument("--ps_hosts", type=str, default="localhost",
                        help="Comma-separated list of hostname:port pairs.")
    parser.add_argument("--ps_port", type=int, default=2200, help="parameter port for each host")
    parser.add_argument("--worker_hosts", type=str, default="localhost",
                        help="Comma-separated list of hostname:port pairs.")
    parser.add_argument("--worker_per_host", type=int, default=1, help="worker count per host")
    parser.add_argument("--worker_port", type=int, default=2300,
                        help="worker start port each host, if two workers per host, then [worker_port, worker_port+1]")
    parser.add_argument("--job_name", type=str, default="ps", help="One of 'ps', 'worker'.")
    parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job.")
    parser.add_argument("--use_bfloat16_transfer", action='store_true', default=False,
                        help="Use bfloat16 to reduce network traffic")
    parser.add_argument("--use_epoch_signal", action='store_true', default=False,
                        help="Use signal to control each work to run exact same epoch.")
    parser.add_argument("--wk_intra_parallel", type=int, default=0,
                        help="The execution of an individual op (for some op types) can be " +
                             "parallelized on a pool of intra_op_parallelism_threads. " +
                             "0 means the system picks an appropriate number.")
    parser.add_argument("--wk_inter_parallel", type=int, default=0,
                        help="Nodes that perform blocking operations are enqueued on a pool of" +
                             "inter_op_parallelism_threads available in each process." +
                             "0 means the system picks an appropriate number." +
                             "Note that the first Session created in the process sets the" +
                             "number of threads for all future sessions unless use_per_session_threads is" +
                             "true or session_inter_op_thread_pool is configured.")
    # parser.add_argument("--optimizer", type=str, default="AdaGrad",
    #                     help="Optimizer type, options are: AdaGrad, FTRL, RMSProp, GradientDecent")
    # parser.add_argument("--success_ratio", type=float, default=0.95,
    #                     help="master work will exit after success_ratio workers finished")
    # parser.add_argument("--session_run_timeout", type=int, default=1000, help="session single step timeout in ms")
    # parser.add_argument("--wait_for_exit_timeout", type=int, default=1800, help="wait seconds before force exiting")

    # Debug arguments
    parser.add_argument("--dist_in_single_machine", action='store_true', default=False,
                        help="Whether training in a machine, it is for testing the correct of codes.")
    parser.add_argument("--debug", action='store_true', default=False, help="Whether output inner states for debug.")
    parser.add_argument("--use_print", action='store_true', default=False, help="Whether use python print.")
    parser.add_argument("--log_device_placement", action='store_true', default=False,
                        help="Print placement of variables and operations.")
    parser.add_argument("--use_validation", action='store_true', default=False,
                        help="Whether use validation")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
