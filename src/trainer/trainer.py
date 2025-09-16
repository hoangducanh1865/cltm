import os
import sys
import json
import datetime
import numpy as np
import tensorflow as tf
from math import *
# Add TensorFlow 2.x compatibility for TensorFlow 1.x code
tf.compat.v1.disable_eager_execution()
import src.data_manager.data_manager as data_manager
import src.model.cltm as cltm
import src.evaluator.evaluation_utils as eval

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


class Trainer:
    def __init__(self):
        pass
    
    @staticmethod
    def train(model, dataset, dataset_old, params, x_old_loss_values):
        log_dir = os.path.join(params.model, 'logs')
        model_dir_ir = os.path.join(params.model, 'model_ir')
        model_dir_ppl = os.path.join(params.model, 'model_ppl')
        model_dir_supervised = os.path.join(params.model, 'model_supervised')

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=params.num_cores,
            intra_op_parallelism_threads=params.num_cores,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
        )) as session:
            avg_loss = tf.compat.v1.placeholder(tf.float32, [], 'loss_ph')
            tf.compat.v1.summary.scalar('loss', avg_loss)

            validation = tf.compat.v1.placeholder(tf.float32, [], 'validation_ph')
            validation_accuracy = tf.compat.v1.placeholder(tf.float32, [], 'validation_acc')
            tf.compat.v1.summary.scalar('validation', validation)
            tf.compat.v1.summary.scalar('validation_accuracy', validation_accuracy)

            summary_writer = tf.compat.v1.summary.FileWriter(log_dir, session.graph)
            summaries = tf.compat.v1.summary.merge_all()
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

            tf.compat.v1.local_variables_initializer().run()
            tf.compat.v1.global_variables_initializer().run()

            losses = []

            # This currently streams from disk. You set num_epochs=1 and
            # wrap this call with something like itertools.cycle to keep
            # this data in memory.
            # shuffle: the order of words in the sentence for DocNADE
            if params.bidirectional:
                pass
            else:
                training_data = dataset.batches('training_docnade', params.batch_size, shuffle=True, multilabel=params.multi_label)

                training_data_old_list = []
                training_doc_ids_old_list = []
                for i, dataset_temp in enumerate(dataset_old):
                    training_data_old_list.append(dataset_temp.batches('training_docnade', params.batch_size, shuffle=True, multilabel=params.source_multi_label[i]))
                    training_doc_ids_old_list.append(dataset_temp.batches('training_document_indices', params.batch_size, shuffle=True, multilabel=params.source_multi_label[i]))

            best_val_IR = 0.0
            best_val_nll = np.inf
            best_val_ppl = np.inf
            best_val_disc_accuracy = 0.0

            best_test_IR = 0.0
            best_test_nll = np.inf
            best_test_ppl = np.inf
            best_test_disc_accuracy = 0.0
            
            #if params.bidirectional or params.initialize_docnade:
            #    patience = 30
            #else:
            #    patience = params.patience
            
            patience = params.patience

            patience_count = 0
            patience_count_ir = 0
            best_train_nll = np.inf

            training_labels = np.array(
                [[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
            )
            validation_labels = np.array(
                [[y] for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
            )
            test_labels = np.array(
                [[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
            )

            sal_docs_taken_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)
            sal_docs_total_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)
            
            for step in range(params.num_steps + 1):
                this_loss = -1.

                if params.bidirectional:
                    pass
                else:
                    y, x, seq_lengths = next(training_data)

                    x_old_list_temp = []
                    x_old_doc_ids_list = []
                    seq_lengths_old_list = []
                    for (training_data_old, training_doc_ids_old) in zip(training_data_old_list, training_doc_ids_old_list):
                        y_old, x_old, seq_lengths_old = next(training_data_old)
                        y_old_doc_ids, x_old_doc_ids, seq_lengths_old_doc_ids = next(training_doc_ids_old)
                        y_old_doc_ids = np.array(y_old_doc_ids, dtype=np.float32)
                        x_old_doc_ids = np.squeeze(x_old_doc_ids)

                        x_old_list_temp.append(x_old)
                        x_old_doc_ids_list.append(x_old_doc_ids)
                        seq_lengths_old_list.append(seq_lengths_old)

                    if len(x_old_list_temp) > 1:
                        max_doc_len = 0
                        for x_old_temp in x_old_list_temp:
                            if x_old_temp.shape[1] > max_doc_len:
                                max_doc_len = x_old_temp.shape[1]

                        #print("Max doc length: ", max_doc_len)

                        x_old_list = []
                        for x_old_temp in x_old_list_temp:
                            if x_old_temp.shape[1] != max_doc_len:
                                pad_len = max_doc_len - x_old_temp.shape[1]
                                x_old_list.append(np.pad(x_old_temp, ((0,0), (0,pad_len)), 'constant', constant_values=(0,0)))
                            else:
                                x_old_list.append(x_old_temp)
                        x_old_list = np.array(x_old_list)
                    else:
                        x_old_list = x_old_list_temp
                    
                    #import pdb; pdb.set_trace()

                    if x_old_loss_values is None:
                        x_old_loss_input = np.sum(y_old_doc_ids)
                    else:
                        x_old_loss_input = np.mean(x_old_loss_values[x_old_doc_ids])
                
                    if params.supervised:
                        print("Error: params.supervised == ", params.supervised)
                        sys.exit()
                    else:
                        if params.sal_loss and (params.sal_gamma == "manual"):
                            _, loss, loss_unnormed, sal_gamma_mask_list = session.run([model.opt, model.loss_normed, model.loss_unnormed, model.sal_gamma_mask_list], feed_dict={
                                model.x: x,
                                model.y: y,
                                model.x_old: x_old_list,
                                model.x_old_doc_ids: x_old_doc_ids_list,
                                model.seq_lengths: seq_lengths,
                                model.seq_lengths_old: seq_lengths_old_list,
                                model.x_old_loss: x_old_loss_input
                            })
                            this_loss = loss
                            losses.append(this_loss)
                            
                            for i, sal_gamma_mask in enumerate(sal_gamma_mask_list):
                                sal_docs_taken_list[i] += np.sum(sal_gamma_mask)
                                sal_docs_total_list[i] += len(sal_gamma_mask)
                        else:
                            _, loss, loss_unnormed = session.run([model.opt, model.loss_normed, model.loss_unnormed], feed_dict={
                                model.x: x,
                                model.y: y,
                                model.x_old: x_old_list,
                                model.x_old_doc_ids: x_old_doc_ids_list,
                                model.seq_lengths: seq_lengths,
                                model.seq_lengths_old: seq_lengths_old_list,
                                model.x_old_loss: x_old_loss_input
                            })
                            this_loss = loss
                            losses.append(this_loss)

                if (step % params.log_every == 0):
                    print('{}: {:.6f}'.format(step, this_loss))

                if step and (step % params.validation_ppl_freq) == 0:
                    if params.sal_loss:
                        if params.sal_gamma == "manual":
                            doc_str = ""
                            for (taken, total) in zip(sal_docs_taken_list, sal_docs_total_list):
                                doc_str += "(" + str(taken) + "/" + str(total) + ")"
                            
                            print("SAL docs: %s" % doc_str)
                            with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                                f.write("SAL docs: %s\n" % (doc_str))

                    sal_docs_taken_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)
                    sal_docs_total_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)

                    this_val_nll = []
                    this_val_loss_normed = []
                    # val_loss_unnormed is NLL
                    this_val_nll_bw = []
                    this_val_loss_normed_bw = []

                    this_val_disc_accuracy = []

                    if params.bidirectional:
                        pass
                    else:
                        for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        #for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=False, multilabel=params.multi_label):
                            if params.supervised:
                                pass
                            else:
                                val_loss_normed, val_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict={
                                    model.x: val_x,
                                    model.y: val_y,
                                    model.seq_lengths: val_seq_lengths
                                })
                            this_val_nll.append(val_loss_unnormed)
                            this_val_loss_normed.append(val_loss_normed)
                    
                    if params.bidirectional:
                        pass
                    else:
                        total_val_nll = np.mean(this_val_nll)
                        total_val_ppl = np.exp(np.mean(this_val_loss_normed))

                    if total_val_ppl < best_val_ppl:
                        best_val_ppl = total_val_ppl
                        print('saving: {}'.format(model_dir_ppl))
                        saver.save(session, model_dir_ppl + '/model_ppl', global_step=1)

                    # Early stopping
                    if total_val_nll < best_val_nll:
                        best_val_nll = total_val_nll
                        patience_count = 0
                    else:
                        patience_count += 1

                    print('This val PPL: {:.3f} (best val PPL: {:.3f},  best val loss: {:.3f}'.format(
                        total_val_ppl,
                        best_val_ppl or 0.0,
                        best_val_nll
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,	val PPL: %s,	 best val PPL: %s,	best val loss: %s\n" % 
                                (step, total_val_ppl, best_val_ppl, best_val_nll))

                    if patience_count > patience:
                        print("Early stopping criterion satisfied.")
                        break
                
                if step and (step % params.validation_ir_freq) == 0:
                    #import pdb; pdb.set_trace()
                    if params.sal_loss:
                        if params.sal_gamma == "manual":
                            doc_str = ""
                            for (taken, total) in zip(sal_docs_taken_list, sal_docs_total_list):
                                doc_str += "(" + str(taken) + "/" + str(total) + ")"
                            
                            print("SAL docs: %s" % doc_str)
                            with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                                f.write("SAL docs: %s\n" % (doc_str))

                    sal_docs_taken_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)
                    sal_docs_total_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)

                    if params.supervised:
                        pass
                    else:
                        if params.bidirectional:
                            pass
                        else:
                            validation_vectors = cltm.vectors(
                                model,
                                dataset.batches(
                                    'validation_docnade',
                                    params.validation_bs,
                                    num_epochs=1,
                                    shuffle=True,
                                    multilabel=params.multi_label
                                ),
                                session
                            )

                            training_vectors = cltm.vectors(
                                model,
                                dataset.batches(
                                    'training_docnade',
                                    params.validation_bs,
                                    num_epochs=1,
                                    shuffle=True,
                                    multilabel=params.multi_label
                                ),
                                session
                            )

                        val = eval.evaluate(
                            training_vectors,
                            validation_vectors,
                            training_labels,
                            validation_labels,
                            recall=[0.02],
                            num_classes=params.num_classes,
                            multi_label=params.multi_label
                        )[0]

                        if val > best_val_IR:
                            best_val_IR = val
                            print('saving: {}'.format(model_dir_ir))
                            saver.save(session, model_dir_ir + '/model_ir', global_step=1)
                            patience_count_ir = 0
                        else:
                            patience_count_ir += 1
                        
                        print('This val IR: {:.3f} (best val IR: {:.3f})'.format(
                            val,
                            best_val_IR or 0.0
                        ))

                        # logging information
                        with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                            f.write("Step: %i,	val IR: %s,	best val IR: %s\n" % 
                                    (step, val, best_val_IR))
                    
                    if patience_count_ir > patience:
                        print("Early stopping criterion satisfied.")
                        break
                
                if step and (step % params.test_ppl_freq) == 0:
                    this_test_nll = []
                    this_test_loss_normed = []
                    this_test_nll_bw = []
                    this_test_loss_normed_bw = []
                    this_test_disc_accuracy = []

                    if params.bidirectional:
                        pass
                    else:
                        #for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params.test_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params.test_bs, num_epochs=1, shuffle=False, multilabel=params.multi_label):
                            if params.supervised:
                                pass
                            else:
                                test_loss_normed, test_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict={
                                    model.x: test_x,
                                    model.y: test_y,
                                    model.seq_lengths: test_seq_lengths
                                })
                            this_test_nll.append(test_loss_unnormed)
                            this_test_loss_normed.append(test_loss_normed)

                    if params.bidirectional:
                        pass
                    else:
                        total_test_nll = np.mean(this_test_nll)
                        total_test_ppl = np.exp(np.mean(this_test_loss_normed))

                    if total_test_ppl < best_test_ppl:
                        best_test_ppl = total_test_ppl

                    if total_test_nll < best_test_nll:
                        best_test_nll = total_test_nll

                    print('This test PPL: {:.3f} (best test PPL: {:.3f},  best test loss: {:.3f})'.format(
                        total_test_ppl,
                        best_test_ppl or 0.0,
                        best_test_nll
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,	test PPL: %s,	best test PPL: %s,	best test loss: %s\n" % 
                                (step, total_test_ppl, best_test_ppl, best_test_nll))

                
                if step >= 1 and (step % params.test_ir_freq) == 0:
                    if params.supervised:
                        pass
                    else:
                        if params.bidirectional:
                            pass
                        else:
                            test_vectors = cltm.vectors(
                                model,
                                dataset.batches(
                                    'test_docnade',
                                    params.test_bs,
                                    num_epochs=1,
                                    shuffle=True,
                                    multilabel=params.multi_label
                                ),
                                session
                            )

                            training_vectors = cltm.vectors(
                                model,
                                dataset.batches(
                                    'training_docnade',
                                    params.test_bs,
                                    num_epochs=1,
                                    shuffle=True,
                                    multilabel=params.multi_label
                                ),
                                session
                            )

                        test = eval.evaluate(
                            training_vectors,
                            test_vectors,
                            training_labels,
                            test_labels,
                            recall=[0.02],
                            num_classes=params.num_classes,
                            multi_label=params.multi_label
                        )[0]

                        if test > best_test_IR:
                            best_test_IR = test
                        
                        print('This test IR: {:.3f} (best test IR: {:.3f})'.format(
                            test,
                            best_test_IR or 0.0
                        ))

                        # logging information
                        with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                            f.write("Step: %i,	test IR: %s,	best test IR: %s\n" % 
                                (step, test, best_test_IR))
    
    @staticmethod
    def loadGloveModel(gloveFile=None, params=None):
        if gloveFile is None:
            if params.hidden_size == 50:
                gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.50d.txt")
            elif params.hidden_size == 100:
                gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.100d.txt")
            elif params.hidden_size == 200:
                gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.200d.txt")
            elif params.hidden_size == 300:
                gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.300d.txt")
            else:
                print('Invalid dimension [%d] for Glove pretrained embedding matrix!!' %params.hidden_size)
                exit()

        print("Loading Glove Model")
        f = open(gloveFile, 'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.", len(model), " words loaded!")
        return model
    
    @staticmethod
    def get_prior_matrix(prior_embedding_path, prior_vocab, docnade_vocab, hidden_size):
        prior_embedding_matrix = np.load(prior_embedding_path)
        
        W_old_indices = []
        W_new_indices = []
        W_old_matrix = []
        prior_matrix = np.zeros((len(docnade_vocab), hidden_size), dtype=np.float32)
        for i, word in enumerate(docnade_vocab):
            try:
                index = prior_vocab.index(word)
            except ValueError:
                continue
            prior_matrix[i, :] = prior_embedding_matrix[index, :]
            W_old_matrix.append(prior_embedding_matrix[index, :])
            W_old_indices.append(index)
            W_new_indices.append(i)
        
        return prior_matrix, np.array(W_old_matrix, dtype=np.float32), W_old_indices, W_new_indices
    
    def fit(self, args):
        # Setup placeholders
        x = tf.compat.v1.placeholder(tf.int32, shape=(None, None), name='x')
        x_bw = tf.compat.v1.placeholder(tf.int32, shape=(None, None), name='x_bw')
        if args.multi_label:
            y = tf.compat.v1.placeholder(tf.string, shape=(None), name='y')
        else:
            y = tf.compat.v1.placeholder(tf.int32, shape=(None), name='y')
        seq_lengths = tf.compat.v1.placeholder(tf.int32, shape=(None), name='seq_lengths')

        x_old = tf.compat.v1.placeholder(tf.int32, shape=(len(args.sal_threshold), None, None), name='x_old')
        x_old_doc_ids = tf.compat.v1.placeholder(tf.int32, shape=(len(args.sal_threshold), None), name='x_old_doc_ids')
        seq_lengths_old = tf.compat.v1.placeholder(tf.int32, shape=(len(args.sal_threshold), None), name='seq_lengths_old')
        x_old_loss = tf.compat.v1.placeholder(tf.float32, shape=(), name='x_old_loss')

        now = datetime.datetime.now()

        if args.bidirectional:
            args.model += "_iDocNADE"
        else:
            args.model += "_DocNADE"

        if args.supervised:
            args.model += "_supervised"

        if args.use_embeddings_prior:
            args.model += "_emb_lambda_" + str(args.lambda_embeddings) + "_" + "_".join([str(lamb) for lamb in args.lambda_embeddings_list])

        if args.W_pretrained_path or args.U_pretrained_path:
            args.model += "_pretr_reload_"

        if args.pretraining_target:
            args.model += "_pretr_targ_" + str(args.pretraining_epochs)

        if args.bias_sharing:
            args.model += "_bias_sharing_"
        
        args.model +=  "_act_" + str(args.activation) + "_hid_" + str(args.hidden_size) \
                        + "_vocab_" + str(args.vocab_size) + "_lr_" + str(args.learning_rate)

        if args.sal_loss:
            if args.sal_gamma == "automatic":
                #args.model += "_SAL_loss_" + str(args.sal_loss) + "_" + str(args.sal_gamma) + "_" + str(args.sal_gamma_init)
                args.model += "_SAL_loss_" + str(args.sal_loss) + "_" + str(args.sal_gamma) + "_" + "_".join([str(val) for val in args.sal_gamma_init])
            else:
                args.model += "_SAL_loss_" + str(args.sal_loss) + "_" + str(args.sal_gamma) + "_" + "_".join([str(val) for val in args.sal_threshold]) + "_" + "_".join([str(val) for val in args.sal_gamma_init])

        if args.ll_loss:
            #args.model += "_LL_loss_" + str(args.ll_loss) + "_" + str(args.ll_lambda) + "_" + str(args.ll_lambda_init)
            args.model += "_LL_loss_" + str(args.ll_loss) + "_" + str(args.ll_lambda) + "_".join([str(lamb) for lamb in args.ll_lambda_init])

        if args.projection:
            args.model += "_projection"
        
        args.model += "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)
        
        if not os.path.isdir(args.model):
            os.mkdir(args.model)

        docnade_vocab = args.docnadeVocab
        with open(docnade_vocab, 'r') as f:
            vocab_docnade = [w.strip() for w in f.readlines()]

        with open(os.path.join(args.model, 'params.json'), 'w') as f:
            f.write(json.dumps(vars(args)))

        dataset = data_manager.Dataset(args.dataset)
        #dataset_old = data.Dataset(args.dataset_old)
        dataset_old_list = []

        for old_dataset in args.dataset_old:
            dataset_old_list.append(data_manager.Dataset(old_dataset))

        if args.initialize_docnade:
            glove_embeddings = Trainer.loadGloveModel(params=args)
        
        docnade_embedding_matrix = None
        if args.initialize_docnade:
            missing_words = 0
            docnade_embedding_matrix = np.zeros((len(vocab_docnade), args.hidden_size), dtype=np.float32)
            for i, word in enumerate(vocab_docnade):
                if str(word).lower() in glove_embeddings.keys():
                    if len(glove_embeddings[str(word).lower()]) == 0:
                        docnade_embedding_matrix[i, :] = np.zeros((args.hidden_size), dtype=np.float32)
                        missing_words += 1
                    else:
                        docnade_embedding_matrix[i, :] = np.array(glove_embeddings[str(word).lower()], dtype=np.float32)
                else:
                    docnade_embedding_matrix[i, :] = np.zeros((args.hidden_size), dtype=np.float32)
                    missing_words += 1

            docnade_embedding_matrix = tf.convert_to_tensor(docnade_embedding_matrix)
            print("Total missing words:%d out of %d" %(missing_words, len(vocab_docnade)))

        W_pretrained_matrix = None
        if args.W_pretrained_path:
            W_pretrained_matrix = np.load(args.W_pretrained_path)
            print("pretrained W loaded.")

        U_pretrained_matrix = None
        if args.U_pretrained_path:
            U_pretrained_matrix = np.load(args.U_pretrained_path)
            print("pretrained U loaded.")

        W_old_indices_list = []
        W_new_indices_list = []
        W_old_matrices_list = []
        W_embeddings_matrices_list = []
        if args.use_embeddings_prior or args.ll_loss:
            for i, W_old_path in enumerate(args.W_old_path_list):
                with open(args.W_old_vocab_path_list[i], "r") as f:
                    temp_vocab = [str(word).lower().strip() for word in f.readlines()]

                prior_matrix, W_old_matrix, W_old_indices, W_new_indices = Trainer.get_prior_matrix(W_old_path, temp_vocab, vocab_docnade, args.hidden_size)
                W_embeddings_matrices_list.append(prior_matrix)
                W_old_matrices_list.append(W_old_matrix)
                W_old_indices_list.append(W_old_indices)
                W_new_indices_list.append(W_new_indices)
            print("Loaded W_embeddings_matrices_list and W_embeddings_indices_list.")

            args.lambda_embeddings_list = np.array(args.lambda_embeddings_list, dtype=np.float32)
            
        
        U_old_matrices_list = []
        if args.ll_loss:
            for i, (U_old_path, W_old_indices) in enumerate(zip(args.U_old_path_list, W_old_indices_list)):
                prior_matrix = np.load(U_old_path)
                prior_matrix = np.take(prior_matrix, W_old_indices, axis=1)
                U_old_matrices_list.append(prior_matrix)
            print("Loaded U_old_list.")

            args.ll_lambda_init = np.array(args.ll_lambda_init, dtype=np.float32)
        
        
        x_old_loss_values = None
        args.sal_threshold_list = []
        if args.sal_gamma == "manual":
            for sal_threshold in args.sal_threshold:
                args.sal_threshold_list.append(np.ones((args.batch_size), dtype=np.float32) * sal_threshold)


        if args.bidirectional:
            print("Error: args.bidirectional == ", args.bidirectional)
            sys.exit()
        else:
            model = cltm.DocNADE_TL(x, y, x_old, x_old_doc_ids, seq_lengths, seq_lengths_old, args,  x_old_loss, \
                                W_old_list=W_old_matrices_list, U_old_list=U_old_matrices_list, \
                                W_embeddings_matrices_list=W_embeddings_matrices_list, W_old_indices_list=W_old_indices_list, \
                                lambda_embeddings_list=args.lambda_embeddings_list, W_new_indices_list=W_new_indices_list, \
                                W_pretrained=W_pretrained_matrix, U_pretrained=U_pretrained_matrix)
            print("DocNADE created")
        
        Trainer.train(model, dataset, dataset_old_list, args, x_old_loss_values)
