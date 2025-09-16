import argparse
import os
import tensorflow as tf


class Utils:
    def __init__(self):
        pass
    
    @staticmethod
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    @staticmethod
    def setup_environment():
        """Setup TensorFlow environment and random seeds."""
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        tf.compat.v1.disable_eager_execution()
        
        seed = 42
        import numpy as np
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    @staticmethod
    def ensure_dir_exists(dir_path):
        """Create directory if it doesn't exist."""
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    @staticmethod
    def get_model_save_paths(base_model_dir):
        """Get standardized model save paths."""
        return {
            'log_dir': os.path.join(base_model_dir, 'logs'),
            'model_dir_ir': os.path.join(base_model_dir, 'model_ir'),
            'model_dir_ppl': os.path.join(base_model_dir, 'model_ppl'),
            'model_dir_supervised': os.path.join(base_model_dir, 'model_supervised')
        }
    
    @staticmethod
    def create_session_config(num_cores):
        """Create TensorFlow session configuration."""
        return tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=num_cores,
            intra_op_parallelism_threads=num_cores,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
        )
    
    @staticmethod
    def build_model_directory_name(args):
        """Build unique model directory name based on parameters."""
        import datetime
        
        now = datetime.datetime.now()
        model_name = args.model

        if args.bidirectional:
            model_name += "_iDocNADE"
        else:
            model_name += "_DocNADE"

        if args.supervised:
            model_name += "_supervised"

        if args.use_embeddings_prior:
            model_name += "_emb_lambda_" + str(args.lambda_embeddings) + "_" + "_".join([str(lamb) for lamb in args.lambda_embeddings_list])

        if args.W_pretrained_path or args.U_pretrained_path:
            model_name += "_pretr_reload_"

        if args.pretraining_target:
            model_name += "_pretr_targ_" + str(args.pretraining_epochs)

        if args.bias_sharing:
            model_name += "_bias_sharing_"
        
        model_name += "_act_" + str(args.activation) + "_hid_" + str(args.hidden_size) \
                    + "_vocab_" + str(args.vocab_size) + "_lr_" + str(args.learning_rate)

        if args.sal_loss:
            if args.sal_gamma == "automatic":
                model_name += "_SAL_loss_" + str(args.sal_loss) + "_" + str(args.sal_gamma) + "_" + "_".join([str(val) for val in args.sal_gamma_init])
            else:
                model_name += "_SAL_loss_" + str(args.sal_loss) + "_" + str(args.sal_gamma) + "_" + "_".join([str(val) for val in args.sal_threshold]) + "_" + "_".join([str(val) for val in args.sal_gamma_init])

        if args.ll_loss:
            model_name += "_LL_loss_" + str(args.ll_loss) + "_" + str(args.ll_lambda) + "_".join([str(lamb) for lamb in args.ll_lambda_init])

        if args.projection:
            model_name += "_projection"
        
        model_name += "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)
        
        return model_name
    
    @staticmethod
    def save_params_to_json(args, model_dir):
        """Save parameters to JSON file."""
        import json
        
        Utils.ensure_dir_exists(model_dir)
        with open(os.path.join(model_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(vars(args)))