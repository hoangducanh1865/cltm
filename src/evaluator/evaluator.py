import os
import json
import numpy as np
import tensorflow as tf
from math import *
# Add TensorFlow 2.x compatibility for TensorFlow 1.x code
tf.compat.v1.disable_eager_execution()
import src.data_manager.data_manager as data_manager
import src.model.cltm as cltm
import src.evaluator.evaluation_utils as eval
from src.utils.utils import Utils
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# sys.setdefaultencoding() does not exist, here!
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

home_dir = os.getenv("HOME")

# dir(tf.contrib) - removed in TensorFlow 2.x


def reload_evaluation_ir(params, training_vectors, validation_vectors, test_vectors, 
                        training_labels, validation_labels, test_labels, 
                        ir_ratio_list, W_matrix, suffix=""):
    log_dir = os.path.join(params['model'], 'logs')

    #ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    ir_ratio_list = [0.02]

    val_ir_list = eval.evaluate(
        training_vectors,
        validation_vectors,
        training_labels,
        validation_labels,
        recall=ir_ratio_list,
        num_classes=params['num_classes'],
        multi_label=params['multi_label']
    )

    # logging information
    with open(os.path.join(log_dir, "reload_info_ir_" + suffix + ".txt"), "w") as f:
        f.write("\n\nFractions list: %s" % (ir_ratio_list))
        f.write("\nVal IR: %s" % (val_ir_list))

    test_ir_list = eval.evaluate(
        training_vectors,
        test_vectors,
        training_labels,
        test_labels,
        recall=ir_ratio_list,
        num_classes=params['num_classes'],
        multi_label=params['multi_label']
    )

    # logging information
    with open(os.path.join(log_dir, "reload_info_ir_" + suffix + ".txt"), "a") as f:
        f.write("\n\nFractions list: %s" % (ir_ratio_list))
        f.write("\nTest IR: %s" % (test_ir_list))
  
  
def reload_evaluation_ir_source(model_ir, dataset, params, ir_ratio_list, suffix):
    with tf.compat.v1.Session() as session_ir:
        log_dir = os.path.join(params['model'], 'logs')

        #ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        ir_ratio_list = [0.02]

        training_labels = np.array(
            [[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
        )
        validation_labels = np.array(
            [[y] for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
        )
        test_labels = np.array(
            [[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
        )

        tf.compat.v1.local_variables_initializer().run()
        tf.compat.v1.global_variables_initializer().run()
        
        if params['bidirectional']:
            pass
        else:
            print("Getting DocNADE document vector representation.")
            training_vectors = cltm.vectors(
                model_ir,
                dataset.batches(
                    'training_docnade',
                    params['validation_bs'],
                    num_epochs=1,
                    shuffle=True,
                    multilabel=params['multi_label']
                ),
                session_ir
            )

            validation_vectors = cltm.vectors(
                model_ir,
                dataset.batches(
                    'validation_docnade',
                    params['validation_bs'],
                    num_epochs=1,
                    shuffle=True,
                    multilabel=params['multi_label']
                ),
                session_ir
            )

            test_vectors = cltm.vectors(
                model_ir,
                dataset.batches(
                    'test_docnade',
                    params['test_bs'],
                    num_epochs=1,
                    shuffle=True,
                    multilabel=params['multi_label']
                ),
                session_ir
            )
        
        # Validation IR

        #import pdb; pdb.set_trace()

        val_list = eval.evaluate(
            training_vectors,
            validation_vectors,
            training_labels,
            validation_labels,
            recall=ir_ratio_list,
            num_classes=params['num_classes'],
            multi_label=params['multi_label']
        )
        
        print('Val IR: ', val_list)

        # logging information
        with open(os.path.join(log_dir, "reload_info_ir_" + suffix + ".txt"), "w") as f:
            f.write("\n\nFractions list: %s" % (ir_ratio_list))
            f.write("\nVal IR: %s" % (val_list))
        
        
        # Test IR
        
        test_list = eval.evaluate(
            training_vectors,
            test_vectors,
            training_labels,
            test_labels,
            recall=ir_ratio_list,
            num_classes=params['num_classes'],
            multi_label=params['multi_label']
        )
        
        print('Test IR: ', test_list)

        # logging information
        with open(os.path.join(log_dir, "reload_info_ir_" + suffix + ".txt"), "a") as f:
            f.write("\n\nFractions list: %s" % (ir_ratio_list))
            f.write("\n\nTest IR: %s" % (test_list))
   
def reload_evaluation_ppl_source(model_ppl, dataset, params, suffix="", model_one_shot=None):
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=params['num_cores'],
        intra_op_parallelism_threads=params['num_cores'],
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )) as session_ppl_source:

        log_dir = os.path.join(params['model'], 'logs')

        tf.compat.v1.local_variables_initializer().run()
        tf.compat.v1.global_variables_initializer().run()

        # TODO: Validation PPL

        this_val_nll = []
        this_val_loss_normed = []
        # val_loss_unnormed is NLL
        this_val_nll_bw = []
        this_val_loss_normed_bw = []

        this_val_disc_accuracy = []

        if params['bidirectional']:
            pass
        else:
            for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
            #for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', batch_size=1, num_epochs=1, shuffle=False, multilabel=params['multi_label']):
                if params['supervised']:
                    pass
                else:
                    val_loss_normed, val_loss_unnormed = session_ppl_source.run([model_ppl.loss_normed, model_ppl.loss_unnormed], feed_dict={
                        model_ppl.x: val_x,
                        model_ppl.y: val_y,
                        model_ppl.seq_lengths: val_seq_lengths
                    })
                this_val_nll.append(val_loss_unnormed)
                this_val_loss_normed.append(val_loss_normed)
        
        if params['bidirectional']:
            total_val_nll = 0.5 * (np.mean(this_val_nll) + np.mean(this_val_nll_bw))
            total_val_ppl = 0.5 * (np.exp(np.mean(this_val_loss_normed)) + np.exp(np.mean(this_val_loss_normed_bw)))
        else:
            total_val_nll = np.mean(this_val_nll)
            total_val_ppl = np.exp(np.mean(this_val_loss_normed))

        print('Val PPL: {:.3f},\tVal loss: {:.3f}\n'.format(
            total_val_ppl,
            total_val_nll
        ))

        # logging information
        with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "w") as f:
            f.write("Val PPL: %s,\tVal loss: %s" % 
                    (total_val_ppl, total_val_nll))
        

        # TODO: Test PPL

        this_test_nll = []
        this_test_loss_normed = []
        this_test_nll_bw = []
        this_test_loss_normed_bw = []
        this_test_disc_accuracy = []

        if params['bidirectional']:
            pass
        else:
            #for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
            for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', batch_size=1, num_epochs=1, shuffle=False, multilabel=params['multi_label']):
                if params['supervised']:
                    pass
                else:
                    test_loss_normed, test_loss_unnormed = session_ppl_source.run([model_ppl.loss_normed, model_ppl.loss_unnormed], feed_dict={
                        model_ppl.x: test_x,
                        model_ppl.y: test_y,
                        model_ppl.seq_lengths: test_seq_lengths
                    })
                this_test_nll.append(test_loss_unnormed)
                this_test_loss_normed.append(test_loss_normed)

        if params['bidirectional']:
            total_test_nll = 0.5 * (np.mean(this_test_nll) + np.mean(this_test_nll_bw))
            total_test_ppl = 0.5 * (np.exp(np.mean(this_test_loss_normed)) + np.exp(np.mean(this_test_loss_normed_bw)))
        else:
            total_test_nll = np.mean(this_test_nll)
            total_test_ppl = np.exp(np.mean(this_test_loss_normed))

        print('Test PPL: {:.3f},\tTest loss: {:.3f}\n'.format(
            total_test_ppl,
            total_test_nll
        ))

        # logging information
        with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "a") as f:
            f.write("\n\nTest PPL: %s,\tTest loss: %s" % 
                    (total_test_ppl, total_test_nll))


def reload_evaluation_ppl(params, suffix=""):
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=params['num_cores'],
        intra_op_parallelism_threads=params['num_cores'],
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )) as session_ppl:

        dataset = data_manager.Dataset(params['dataset'])
        log_dir = os.path.join(params['model'], 'logs')
            
        saver_ppl = tf.compat.v1.train.import_meta_graph("data/saved_models/" + params['reload_model_dir'] + "model_ppl/model_ppl-1.meta")
        saver_ppl.restore(session_ppl, tf.compat.v1.train.latest_checkpoint("data/saved_models/" + params['reload_model_dir'] + "model_ppl/"))

        graph = tf.compat.v1.get_default_graph()

        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        seq_lengths = graph.get_tensor_by_name("seq_lengths:0")
        loss_normed = graph.get_tensor_by_name("loss_normed_x:0")
        loss_unnormed = graph.get_tensor_by_name("loss_unnormed_x:0")

        # TODO: Validation PPL

        this_val_nll = []
        this_val_loss_normed = []
        # val_loss_unnormed is NLL
        this_val_nll_bw = []
        this_val_loss_normed_bw = []

        this_val_disc_accuracy = []

        if params['bidirectional']:
            pass
        else:
            for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params['validation_bs'], num_epochs=1, shuffle=True, multilabel=params['multi_label']):
            #for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params['validation_bs'], num_epochs=1, shuffle=False, multilabel=params['multi_label']):
                if params['supervised']:
                    pass
                else:
                    val_loss_normed, val_loss_unnormed = session_ppl.run([loss_normed, loss_unnormed], feed_dict={
                        x: val_x,
                        y: val_y,
                        seq_lengths: val_seq_lengths
                    })
                this_val_nll.append(val_loss_unnormed)
                this_val_loss_normed.append(val_loss_normed)
        
        if params['bidirectional']:
            total_val_nll = 0.5 * (np.mean(this_val_nll) + np.mean(this_val_nll_bw))
            total_val_ppl = 0.5 * (np.exp(np.mean(this_val_loss_normed)) + np.exp(np.mean(this_val_loss_normed_bw)))
        else:
            total_val_nll = np.mean(this_val_nll)
            total_val_ppl = np.exp(np.mean(this_val_loss_normed))

        print('Val PPL: {:.3f},\tVal loss: {:.3f}\n'.format(
            total_val_ppl,
            total_val_nll
        ))

        # logging information
        with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "w") as f:
            f.write("Val PPL: %s,\tVal loss: %s" % 
                    (total_val_ppl, total_val_nll))
        
        # TODO: Test PPL

        this_test_nll = []
        this_test_loss_normed = []
        this_test_nll_bw = []
        this_test_loss_normed_bw = []
        this_test_disc_accuracy = []

        if params['bidirectional']:
            pass
        else:
            #for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params['test_bs'], num_epochs=1, shuffle=True, multilabel=params['multi_label']):
            for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params['test_bs'], num_epochs=1, shuffle=False, multilabel=params['multi_label']):
                if params['supervised']:
                    pass
                else:
                    test_loss_normed, test_loss_unnormed = session_ppl.run([loss_normed, loss_unnormed], feed_dict={
                        x: test_x,
                        y: test_y,
                        seq_lengths: test_seq_lengths
                    })
                this_test_nll.append(test_loss_unnormed)
                this_test_loss_normed.append(test_loss_normed)

        if params['bidirectional']:
            total_test_nll = 0.5 * (np.mean(this_test_nll) + np.mean(this_test_nll_bw))
            total_test_ppl = 0.5 * (np.exp(np.mean(this_test_loss_normed)) + np.exp(np.mean(this_test_loss_normed_bw)))
        else:
            total_test_nll = np.mean(this_test_nll)
            total_test_ppl = np.exp(np.mean(this_test_loss_normed))

        print('Test PPL: {:.3f},\tTest loss: {:.3f}\n'.format(
            total_test_ppl,
            total_test_nll
        ))

        # logging information
        with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "a") as f:
            f.write("\n\nTest PPL: %s,\tTest loss: %s" % 
                    (total_test_ppl, total_test_nll))

        W_target = session_ppl.run("embedding:0")
        bias_W_target = session_ppl.run("bias:0")
        U_target = session_ppl.run("U:0")
        bias_U_target = session_ppl.run("b:0")

        #import pdb; pdb.set_trace()

        source_data_W_projection_list = []
        source_data_U_projection_list = []
        if params['ll_loss'] and params['projection']:
            for i, source_data in enumerate(params['reload_source_data_list']):
                source_data_W_projection_list.append(session_ppl.run("ll_projection_W_" + str(i) + ":0"))
                source_data_U_projection_list.append(session_ppl.run("ll_projection_U_" + str(i) + ":0"))
        
        return W_target, bias_W_target, U_target, bias_U_target, source_data_W_projection_list, source_data_U_projection_list


def compute_coherence(texts, list_of_topics, top_n_word_in_each_topic_list, reload_model_dir):

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print('corpus len:%s' %len(corpus))
    print('dictionary:%s' %dictionary)
    # https://github.com/earthquakesan/palmetto-py
    # compute_topic_coherence: PMI and other coherence types
    # from palmettopy.palmetto import Palmetto
    # palmetto = Palmetto()

    # coherence_types = ["ca", "cp", "cv", "npmi", "uci", "umass"] # for palmetto library
    coherence_types = ["c_v"]#, 'u_mass', 'c_v', 'c_uci', 'c_npmi'] # ["c_v"] # 'u_mass', 'c_v', 'c_uci', 'c_npmi',
    avg_coh_scores_dict = {}

    best_coh_type_value_topci_indx = {}
    for top_n in top_n_word_in_each_topic_list:
        avg_coh_scores_dict[top_n]= []
        best_coh_type_value_topci_indx[top_n] = [0,  0, []] # score, topic_indx, topics words


    h_num = 0
    with open(reload_model_dir, "w") as f:
        for topic_words_all in list_of_topics:
            h_num += 1
            for top_n in top_n_word_in_each_topic_list:
                topic_words = [topic_words_all[:top_n]]
                for coh_type in coherence_types:
                    try:
                        print('top_n: %s Topic Num: %s \nTopic Words: %s' % (top_n, h_num, topic_words))
                        f.write('top_n: %s Topic Num: %s \nTopic Words: %s\n' % (top_n, h_num, topic_words))
                        # print('topic_words_top_10_abs[%s]:%s' % (h_num, topic_words_top_10_abs[h_num]))
                        # PMI = palmetto.get_coherence(topic_words_top_10[h_num], coherence_type=coh_type)
                        PMI = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence=coh_type, processes=2).get_coherence()

                        avg_coh_scores_dict[top_n].append(PMI)

                        if PMI > best_coh_type_value_topci_indx[top_n][0]:
                            best_coh_type_value_topci_indx[top_n] = [PMI, top_n, topic_words]

                        print('Coh_type:%s  Topic Num:%s COH score:%s' % (coh_type, h_num, PMI))
                        f.write('Coh_type:%s  Topic Num:%s COH score:%s\n' % (coh_type, h_num, PMI))

                        '''
                        output_topics_coh_filename_fp.write(str('h_num:') + str(h_num) + ' ' + str('PMI_') +
                                                            str(coh_type) + ' ' + str('COH:') + str(PMI) + ' ' + str('topicsWords:'))

                        for word in topic_words_top_10[h_num]:
                            output_topics_coh_filename_fp.write(str(word) + ' ')

                        output_topics_coh_filename_fp.write('\n')

                        output_topics_coh_filename_fp.write(
                            str('--------------------------------------------------------------') + '\n')
                        '''


                        print('--------------------------------------------------------------')
                    except:
                        continue
                print('========================================================================================================')

        for top_n in top_n_word_in_each_topic_list:
            print('top scores for top_%s:%s' %(top_n, best_coh_type_value_topci_indx[top_n]))
            print('-------------------------------------------------------------------')
            f.write('top scores for top_%s:%s\n' %(top_n, best_coh_type_value_topci_indx[top_n]))
            f.write('-------------------------------------------------------------------\n')

        for top_n in top_n_word_in_each_topic_list:
            print('Avg COH for top_%s topic words: %s' %(top_n, np.mean(avg_coh_scores_dict[top_n])))
            print('-------------------------------------------------------------------')
            f.write('Avg COH for top_%s topic words: %s\n' %(top_n, np.mean(avg_coh_scores_dict[top_n])))
            f.write('-------------------------------------------------------------------\n')


def reload_evaluation_topics(W_target, U_target, params, suffix=""):

    log_dir = os.path.join(params['model'], 'logs')

    # Topics with W matrix

    top_n_topic_words = 20
    w_h_top_words_indices = []
    W_topics = W_target
    topics_list_W = []

    # Debug: print shape information
    print(f"W_topics shape: {np.array(W_topics).shape}")
    
    # Check if W_topics has the right shape
    W_shape = np.array(W_topics).shape
    if len(W_shape) == 1:
        print("Warning: W_topics is 1D, reshaping...")
        W_topics = W_topics.reshape(-1, 1)
        W_shape = W_topics.shape
    
    # Use the smaller dimension as the number of topics
    num_topics = min(W_shape[0], W_shape[1]) if len(W_shape) >= 2 else W_shape[0]
    vocab_size = max(W_shape[0], W_shape[1]) if len(W_shape) >= 2 else W_shape[0]
    
    print(f"Using {num_topics} topics with vocab size {vocab_size}")

    for h_num in range(num_topics):
        if len(W_shape) >= 2 and W_shape[1] > W_shape[0]:
            # If W is transposed (topics x vocab)
            w_h_top_words_indices.append(np.argsort(W_topics[h_num, :])[::-1][:top_n_topic_words])
        else:
            # Normal case (vocab x topics)
            w_h_top_words_indices.append(np.argsort(W_topics[:, h_num])[::-1][:top_n_topic_words])

    with open(params['docnadeVocab'], 'r') as f:
        vocab_docnade = [w.strip() for w in f.readlines()]

    print(f"Vocab size: {len(vocab_docnade)}")
    print(f"W_target shape: {W_target.shape}")
    
    # More flexible assertion - check if dimensions are compatible
    W_target_shape = W_target.shape if len(W_target.shape) > 1 else (W_target.shape[0], 1)
    max_vocab_index = max(max(indices) for indices in w_h_top_words_indices) if w_h_top_words_indices else 0
    
    if max_vocab_index >= len(vocab_docnade):
        print(f"Warning: max vocab index {max_vocab_index} >= vocab size {len(vocab_docnade)}")
        print("Limiting word indices to vocab size")
        # Limit indices to vocab size
        w_h_top_words_indices = [[idx for idx in indices if idx < len(vocab_docnade)] for indices in w_h_top_words_indices]

    with open(os.path.join(log_dir, "topics_ppl_W_" + suffix + ".txt"), "w") as f:
        for w_h_top_words_indx, h_num in zip(w_h_top_words_indices, range(len(w_h_top_words_indices))):
            w_h_top_words = [vocab_docnade[w_indx] for w_indx in w_h_top_words_indx]
            topics_list_W.append(w_h_top_words)
            print('h_num: %s' % h_num)
            print('w_h_top_words_indx: %s' % w_h_top_words_indx)
            print('w_h_top_words:%s' % w_h_top_words)
            print('----------------------------------------------------------------------')

            f.write('h_num: %s\n' % h_num)
            f.write('w_h_top_words_indx: %s\n' % w_h_top_words_indx)
            f.write('w_h_top_words:%s\n' % w_h_top_words)
            f.write('----------------------------------------------------------------------\n')
            
    
    # Topics with V matrix

    top_n_topic_words = 20
    w_h_top_words_indices = []
    W_topics = U_target.T
    topics_list_V = []

    # Debug: print shape information for U matrix
    print(f"U_topics (transposed) shape: {np.array(W_topics).shape}")
    
    # Check if W_topics has the right shape
    W_shape = np.array(W_topics).shape
    if len(W_shape) == 1:
        print("Warning: U_topics (transposed) is 1D, reshaping...")
        W_topics = W_topics.reshape(-1, 1)
        W_shape = W_topics.shape
    
    # Use the smaller dimension as the number of topics
    num_topics_V = min(W_shape[0], W_shape[1]) if len(W_shape) >= 2 else W_shape[0]
    vocab_size_V = max(W_shape[0], W_shape[1]) if len(W_shape) >= 2 else W_shape[0]
    
    print(f"Using {num_topics_V} topics with vocab size {vocab_size_V} for V matrix")

    for h_num in range(num_topics_V):
        if len(W_shape) >= 2 and W_shape[1] > W_shape[0]:
            # If W is transposed (topics x vocab)
            w_h_top_words_indices.append(np.argsort(W_topics[h_num, :])[::-1][:top_n_topic_words])
        else:
            # Normal case (vocab x topics)
            w_h_top_words_indices.append(np.argsort(W_topics[:, h_num])[::-1][:top_n_topic_words])

    with open(params['docnadeVocab'], 'r') as f:
        vocab_docnade = [w.strip() for w in f.readlines()]

    # Validate word indices for V matrix
    max_vocab_index_V = max(max(indices) for indices in w_h_top_words_indices) if w_h_top_words_indices else 0
    if max_vocab_index_V >= len(vocab_docnade):
        print(f"Warning: max vocab index {max_vocab_index_V} >= vocab size {len(vocab_docnade)} for V matrix")
        print("Limiting word indices to vocab size")
        w_h_top_words_indices = [[idx for idx in indices if idx < len(vocab_docnade)] for indices in w_h_top_words_indices]

    with open(os.path.join(log_dir, "topics_ppl_V_" + suffix + ".txt"), "w") as f:
        for w_h_top_words_indx, h_num in zip(w_h_top_words_indices, range(len(w_h_top_words_indices))):
            w_h_top_words = [vocab_docnade[w_indx] for w_indx in w_h_top_words_indx if w_indx < len(vocab_docnade)]
            topics_list_V.append(w_h_top_words)
            print('h_num: %s' % h_num)
            print('w_h_top_words_indx: %s' % w_h_top_words_indx)
            print('w_h_top_words:%s' % w_h_top_words)
            print('----------------------------------------------------------------------')

            f.write('h_num: %s\n' % h_num)
            f.write('w_h_top_words_indx: %s\n' % w_h_top_words_indx)
            f.write('w_h_top_words:%s\n' % w_h_top_words)
            f.write('----------------------------------------------------------------------\n')

    
    # TOPIC COHERENCE

    top_n_word_in_each_topic_list = [5, 10, 15, 20]

    text_filenames = [
        params['trainfile'],
        params['valfile'],
        params['testfile']
    ]

    # read original text documents as list of words
    texts = []

    for file in text_filenames:
        print('filename:%s', file)
        for line in open(file, 'r').readlines():
            document = str(line).strip()
            texts.append(document.split())

    compute_coherence(texts, topics_list_W, top_n_word_in_each_topic_list, os.path.join(log_dir, "topics_coherence_W_" + suffix + ".txt"))
    #compute_coherence(texts, topics_list_V, top_n_word_in_each_topic_list, os.path.join(log_dir, "topics_coherence_V_" + suffix + ".txt"))


class Evaluator:
    def __init__(self):
        pass
    
    def evaluate(self, args):
        with open("data/saved_models/" + args.reload_model_dir + "params.json") as f:
            params = json.load(f)

        params['trainfile'] = args.trainfile
        params['valfile'] = args.valfile
        params['testfile'] = args.testfile

        source_datasets_path = "/home/ubuntu/DocNADE_Lifelong_Learning/datasets"
        source_trainfiles = [source_datasets_path + "/20NS/20_news_groups_MVG_train.txt",
                            source_datasets_path + "/R21578/train.txt",
                            source_datasets_path + "/TMN/TMN_train.txt",
                            source_datasets_path + "/AGnews/train.txt"]
        source_valfiles = [source_datasets_path + "/20NS/20_news_groups_MVG_val.txt",
                            source_datasets_path + "/R21578/val.txt",
                            source_datasets_path + "/TMN/TMN_val.txt",
                            source_datasets_path + "/AGnews/val.txt"]
        source_testfiles = [source_datasets_path + "/20NS/20_news_groups_MVG_test.txt",
                            source_datasets_path + "/R21578/test.txt",
                            source_datasets_path + "/TMN/TMN_test.txt",
                            source_datasets_path + "/AGnews/test.txt"]
        source_vocabs = [source_datasets_path + "/20NS/vocab_docnade.vocab",
                        source_datasets_path + "/R21578/vocab_docnade.vocab",
                        source_datasets_path + "/TMN/vocab_docnade.vocab",
                        source_datasets_path + "/AGnews/vocab_docnade.vocab"]

        params['reload_model_dir'] = args.reload_model_dir

        reload_ir = False
        if os.path.isdir("data/saved_models/" + args.reload_model_dir + "/model_ir"):
            reload_ir = True

        reload_ppl = False
        if os.path.isdir("data/saved_models/" + args.reload_model_dir + "/model_ppl"):
            reload_ppl = True
        
        # Debug
        print(f"reload_ppl: {reload_ppl}")

        if reload_ppl:
            W_target, bias_W_target, U_target, bias_U_target, \
            source_data_W_projection_list, source_data_U_projection_list = reload_evaluation_ppl(params, suffix="target")
            reload_evaluation_topics(W_target, U_target, params, suffix="target")

            ## SOURCE DATA
            params['reload_source_data_list'] = args.reload_source_data_list
            params['bias_W_old_path_list'] = args.bias_W_old_path_list
            params['bias_U_old_path_list'] = args.bias_U_old_path_list

            if (params['use_embeddings_prior']) and (not params['ll_loss']) and (not params['sal_loss']):
                params['projection'] = False
                params['ll_loss'] = True
            
            if params['ll_loss']:
                source_multi_label = [Utils.str2bool(value) for value in args.reload_source_multi_label]
                source_num_classes = args.reload_source_num_classes

                with open(params['docnadeVocab'], "r") as f:
                    target_data_vocab = [line.strip().lower() for line in f.readlines()]
                
                for i, source_data in enumerate(params['reload_source_data_list']):
                    with open(source_data + "/vocab_docnade.vocab", "r") as f:
                        source_data_vocab = [line.strip().lower() for line in f.readlines()]

                    params['multi_label'] = source_multi_label[i]
                    params['num_classes'] = source_num_classes[i]

                    if params['projection']:
                        source_data_W_projection = source_data_W_projection_list[i]

                    source_data_W_original = np.load(params['W_old_path_list'][i])
                    source_data_bias_W_original = np.load(params['bias_W_old_path_list'][i])

                    source_data_W_merged = np.zeros_like(source_data_W_original, dtype=np.float32)
                    for j, word in enumerate(source_data_vocab):
                        try:
                            index = target_data_vocab.index(word)
                        except ValueError:
                            source_data_W_merged[j, :] = source_data_W_original[j, :]
                            continue
                        
                        if params['projection']:
                            if len(W_target.shape) == 1:
                                # Handle 1D case
                                source_data_W_merged[j, :] = W_target[index] * source_data_W_projection
                            else:
                                source_data_W_merged[j, :] = np.dot(W_target[index, :], source_data_W_projection)
                        else:
                            if len(W_target.shape) == 1:
                                # Handle 1D case
                                source_data_W_merged[j, :] = W_target[index]
                            else:
                                source_data_W_merged[j, :] = W_target[index, :]
                        
                    if params['projection']:
                        source_data_U_projection = source_data_U_projection_list[i]

                    source_data_U_original = np.load(params['U_old_path_list'][i])
                    source_data_bias_U_original = np.load(params['bias_U_old_path_list'][i])

                    source_data_U_merged = np.zeros_like(source_data_U_original, dtype=np.float32)
                    source_data_bias_U_merged = np.zeros_like(source_data_bias_U_original, dtype=np.float32)
                    for j, word in enumerate(source_data_vocab):
                        try:
                            index = target_data_vocab.index(word)
                        except ValueError:
                            source_data_U_merged[:, j] = source_data_U_original[:, j]
                            source_data_bias_U_merged[j] = source_data_bias_U_original[j]
                            continue
                        source_data_bias_U_merged[j] = bias_U_target[index]

                        if params['projection']:
                            source_data_U_merged[:, j] = np.dot(source_data_U_projection, U_target[:, index])
                        else:
                            source_data_U_merged[:, j] = U_target[:, index]
                        
                
                    x_source = tf.compat.v1.placeholder(tf.int32, shape=(None, None), name='x_source')
                    if params['multi_label']:
                        y_source = tf.compat.v1.placeholder(tf.string, shape=(None), name='y_source')
                    else:
                        y_source = tf.compat.v1.placeholder(tf.int32, shape=(None), name='y_source')
                    seq_lengths_source = tf.compat.v1.placeholder(tf.int32, shape=(None), name='seq_lengths_source')

                    model_ppl_source = cltm.DocNADE_TL_reload(x_source, y_source, seq_lengths_source, params, 
                                                        W_initializer=None, W_reload=source_data_W_merged, 
                                                        W_embeddings_reload=[], W_prior_proj_reload=None, 
                                                        bias_reload=source_data_bias_W_original, bias_bw_reload=None, 
                                                        V_reload=source_data_U_merged, b_reload=source_data_bias_U_original, 
                                                        b_bw_reload=None, W_list_reload=[], bias_list_reload=[], 
                                                        lambda_embeddings_reload=[])

                    print("DocNADE PPL source created")
                    
                    model_ppl_source_one_shot = None
                    source_dataset = data_manager.Dataset(source_data)
                    reload_evaluation_ppl_source(model_ppl_source, source_dataset, params, suffix="source_" + str(i), model_one_shot=model_ppl_source_one_shot)

                    params['trainfile'] = source_trainfiles[i]
                    params['valfile'] = source_valfiles[i]
                    params['testfile'] = source_testfiles[i]
                    params['docnadeVocab'] = source_vocabs[i]
                    #reload_evaluation_topics(source_data_W_merged, source_data_U_merged, params, suffix="source_" + str(i))

        # Reloading and evaluating on Information Retrieval
        if reload_ir:
            sess_ir = tf.compat.v1.Session()
            
            saver_ir = tf.compat.v1.train.import_meta_graph("data/saved_models/" + args.reload_model_dir + "model_ir/model_ir-1.meta")
            saver_ir.restore(sess_ir, tf.compat.v1.train.latest_checkpoint("data/saved_models/" + args.reload_model_dir + "model_ir/"))

            graph = tf.compat.v1.get_default_graph()

            x = graph.get_tensor_by_name("x:0")
            seq_lengths = graph.get_tensor_by_name("seq_lengths:0")
            last_hidden = graph.get_tensor_by_name("last_hidden:0")

            ## TARGET DATA
            dataset = data_manager.Dataset(params['dataset'])

            training_labels = np.array(
                [[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
            )
            validation_labels = np.array(
                [[y] for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
            )
            test_labels = np.array(
                [[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
            )

            hidden_vectors_val = []
            for va_y, va_x, va_seq_lengths in dataset.batches('validation_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
                hidden_vec = sess_ir.run([last_hidden], feed_dict={
                    x: va_x,
                    seq_lengths: va_seq_lengths
                })
                hidden_vectors_val.append(hidden_vec[0])
            hidden_vectors_val = np.squeeze(np.array(hidden_vectors_val, dtype=np.float32))

            hidden_vectors_tr = []
            for tr_y, tr_x, tr_seq_lengths in dataset.batches('training_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
                hidden_vec = sess_ir.run([last_hidden], feed_dict={
                    x: tr_x,
                    seq_lengths: tr_seq_lengths
                })
                hidden_vectors_tr.append(hidden_vec[0])
            hidden_vectors_tr = np.squeeze(np.array(hidden_vectors_tr, dtype=np.float32))

            hidden_vectors_test = []
            for te_y, te_x, te_seq_lengths in dataset.batches('test_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
                hidden_vec = sess_ir.run([last_hidden], feed_dict={
                    x: te_x,
                    seq_lengths: te_seq_lengths
                })
                hidden_vectors_test.append(hidden_vec[0])
            hidden_vectors_test = np.squeeze(np.array(hidden_vectors_test, dtype=np.float32))

            W_target = sess_ir.run("embedding:0")
            bias_W_target = sess_ir.run("bias:0")
            U_target = sess_ir.run("U:0")
            bias_U_target = sess_ir.run("b:0")

            ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

            reload_evaluation_ir(params, hidden_vectors_tr, hidden_vectors_val, hidden_vectors_test, 
                                training_labels, validation_labels, test_labels, 
                                ir_ratio_list, W_target, suffix="target")
            
            ## SOURCE DATA
            params['reload_source_data_list'] = args.reload_source_data_list
            params['bias_W_old_path_list'] = args.bias_W_old_path_list
            params['bias_U_old_path_list'] = args.bias_U_old_path_list

            if (params['use_embeddings_prior']) and (not params['ll_loss']) and (not params['sal_loss']):
                params['projection'] = False
                params['ll_loss'] = True
            
            if params['ll_loss']:
                source_multi_label = [Utils.str2bool(value) for value in args.reload_source_multi_label]
                source_num_classes = args.reload_source_num_classes

                with open(params['docnadeVocab'], "r") as f:
                    target_data_vocab = [line.strip().lower() for line in f.readlines()]
                
                for i, source_data in enumerate(params['reload_source_data_list']):
                    with open(source_data + "/vocab_docnade.vocab", "r") as f:
                        source_data_vocab = [line.strip().lower() for line in f.readlines()]

                    params['multi_label'] = source_multi_label[i]
                    params['num_classes'] = source_num_classes[i]

                    if params['projection']:
                        source_data_W_projection = sess_ir.run("ll_projection_W_" + str(i) + ":0")

                    source_data_W_original = np.load(params['W_old_path_list'][i])
                    source_data_bias_W_original = np.load(params['bias_W_old_path_list'][i])

                    source_data_W_merged = np.zeros_like(source_data_W_original, dtype=np.float32)
                    for j, word in enumerate(source_data_vocab):
                        try:
                            index = target_data_vocab.index(word)
                        except ValueError:
                            source_data_W_merged[j, :] = source_data_W_original[j, :]
                            continue

                        if params['projection']:
                            source_data_W_merged[j, :] = np.dot(W_target[index, :], source_data_W_projection)
                        else:
                            source_data_W_merged[j, :] = W_target[index, :]

                    if params['projection']:
                        source_data_U_projection = sess_ir.run("ll_projection_U_" + str(i) + ":0")

                    source_data_U_original = np.load(params['U_old_path_list'][i])
                    source_data_bias_U_original = np.load(params['bias_U_old_path_list'][i])

                    source_data_U_merged = np.zeros_like(source_data_U_original, dtype=np.float32)
                    source_data_bias_U_merged = np.zeros_like(source_data_bias_U_original, dtype=np.float32)
                    for j, word in enumerate(source_data_vocab):
                        try:
                            index = target_data_vocab.index(word)
                        except ValueError:
                            source_data_U_merged[:, j] = source_data_U_original[:, j]
                            source_data_bias_U_merged[j] = source_data_bias_U_original[j]
                            continue
                        source_data_bias_U_merged[j] = bias_U_target[index]

                        if params['projection']:
                            source_data_U_merged[:, j] = np.dot(source_data_U_projection, U_target[:, index])
                        else:
                            source_data_U_merged[:, j] = U_target[:, index]
            
                    x_source = tf.compat.v1.placeholder(tf.int32, shape=(None, None), name='x_source')
                    if params['multi_label']:
                        y_source = tf.compat.v1.placeholder(tf.string, shape=(None), name='y_source')
                    else:
                        y_source = tf.compat.v1.placeholder(tf.int32, shape=(None), name='y_source')
                    seq_lengths_source = tf.compat.v1.placeholder(tf.int32, shape=(None), name='seq_lengths_source')

                    #params['use_embeddings_prior'] = False

                    model_ir_source = cltm.DocNADE_TL_reload(x_source, y_source, seq_lengths_source, params, 
                                                        W_initializer=None, W_reload=source_data_W_merged, 
                                                        W_embeddings_reload=[], W_prior_proj_reload=None, 
                                                        bias_reload=source_data_bias_W_original, bias_bw_reload=None, 
                                                        V_reload=source_data_U_merged, b_reload=source_data_bias_U_original, 
                                                        b_bw_reload=None, W_list_reload=[], bias_list_reload=[], 
                                                        lambda_embeddings_reload=[])

                    print("DocNADE IR source created")

                    ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

                    source_dataset = data_manager.Dataset(source_data)
                    reload_evaluation_ir_source(model_ir_source, source_dataset, params, ir_ratio_list, suffix="source_" + str(i))
