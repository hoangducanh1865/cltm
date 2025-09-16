import os
import argparse
import numpy as np
import tensorflow as tf
from src.trainer.trainer import Trainer
from src.evaluator.evaluator import Evaluator
from src.utils.utils import Utils

# Setup environment
Utils.setup_environment()

home_dir = os.getenv("HOME")


def main(args):
	if args.reload:
		evaluator = Evaluator()
		evaluator.evaluate(args)
	else:
		trainer = Trainer()
		trainer.fit(args)


def parse_args():
	parser = argparse.ArgumentParser()
	
	# Model and data arguments
	parser.add_argument('--model', type=str, required=True,
						help='path to model output directory')
	parser.add_argument('--dataset', type=str, required=True,
						help='path to the input dataset')
	parser.add_argument('--docnadeVocab', type=str, default="False",
						help='path to vocabulary file used by DocNADE')
	
	# Model architecture arguments
	parser.add_argument('--vocab-size', type=int, default=2000,
						help='the vocab size')
	parser.add_argument('--hidden-size', type=int, default=50,
						help='size of the hidden layer')
	parser.add_argument('--activation', type=str, default='tanh',
						help='which activation to use: sigmoid|tanh')
	parser.add_argument('--num-classes', type=int, default=-1,
						help='number of classes')
	
	# Training arguments
	parser.add_argument('--learning-rate', type=float, default=0.0004,
						help='initial learning rate')
	parser.add_argument('--num-steps', type=int, default=50000,
						help='the number of steps to train for')
	parser.add_argument('--batch-size', type=int, default=64,
						help='the batch size')
	parser.add_argument('--patience', type=int, default=10,
						help='early stopping patience')
	
	# System arguments
	parser.add_argument('--num-cores', type=int, default=2,
						help='the number of CPU cores to use')
	parser.add_argument('--num-samples', type=int, default=None,
						help='softmax samples (default: full softmax)')
	
	# Logging and evaluation arguments
	parser.add_argument('--log-every', type=int, default=10,
						help='print loss after this many steps')
	parser.add_argument('--validation-ppl-freq', type=int, default=500,
						help='validation perplexity frequency')
	parser.add_argument('--test-ppl-freq', type=int, default=100,
						help='test perplexity frequency')
	parser.add_argument('--test-ir-freq', type=int, default=100,
						help='test information retrieval frequency')
	parser.add_argument('--validation-ir-freq', type=int, default=500,
						help='validation information retrieval frequency')
	parser.add_argument('--validation-bs', type=int, default=64,
						help='validation batch size')
	parser.add_argument('--test-bs', type=int, default=64,
						help='test batch size')
	
	# Model type arguments
	parser.add_argument('--supervised', type=str, default="False",
						help='whether to use supervised model or not')
	parser.add_argument('--bidirectional', type=str, default="False",
						help='whether to use bidirectional DocNADE model or not')
	parser.add_argument('--initialize-docnade', type=str, default="False",
						help='whether to embedding matrix of docnade')
	parser.add_argument('--combination-type', type=str, default="concat",
						help='combination type for bidirectional docnade')
	parser.add_argument('--generative-loss-weight', type=float, default=0.0,
						help='weight for generative loss in total loss')
	parser.add_argument('--projection', type=str, default="False",
						help='whether to project prior embeddings or not')
	parser.add_argument('--deep', type=str, default="False",
						help='whether to make model deep or not')
	parser.add_argument('--deep-hidden-sizes', nargs='+', type=int,
						help='sizes of the hidden layers')
	parser.add_argument('--multi-label', type=str, default="False",
						help='whether dataset is multi-label or not')
	
	# Reload arguments
	parser.add_argument('--reload', type=str, default="False",
						help='whether to reload model or not')
	parser.add_argument('--reload-model-dir', type=str,
						help='path for model to be reloaded')
	parser.add_argument('--model-type', type=str,
						help='type of model to be reloaded')
	parser.add_argument('--shuffle-reload', type=str, default="True",
						help='whether dataset is shuffled or not')
	
	# File path arguments
	parser.add_argument('--trainfile', type=str, required=True,
						help='path to train text file')
	parser.add_argument('--valfile', type=str, required=True,
						help='path to validation text file')
	parser.add_argument('--testfile', type=str, required=True,
						help='path to test text file')
	parser.add_argument('--W-pretrained-path', type=str, default="",
						help='path for pretrained W matrix')
	parser.add_argument('--U-pretrained-path', type=str, default="",
						help='path for pretrained U matrix')
	
	# Continual learning arguments
	parser.add_argument('--sal-loss', type=str, default="False",
						help='whether to include SAL loss')
	parser.add_argument('--sal-gamma', type=str, default="automatic",
						help='"automatic" or "manual"')
	parser.add_argument('--sal-gamma-init', type=float, nargs='+', default=[],
						help='initialization value for SAL gamma variable')
	parser.add_argument('--ll-loss', type=str, default="False",
						help='whether to include LL loss')
	parser.add_argument('--ll-lambda', type=str, default="automatic",
						help='"automatic" or "manual"')
	parser.add_argument('--ll-lambda-init', type=float, nargs='+', default=[],
						help='"automatic" or "manual"')
	parser.add_argument('--dataset-old', type=str, nargs='+', required=True,
						help='path to the old datasets')
	parser.add_argument('--pretraining-target', type=str, default="False",
						help='whether to do pretraining on target data or not')
	parser.add_argument('--pretraining-epochs', type=int, default=50,
						help='number of epochs for pretraining')
	parser.add_argument('--bias-sharing', type=str, default="True",
						help='whether to share encoding and decoding bias with old dataset or not')
	parser.add_argument('--sal-threshold', type=float, nargs='+', default=[],
						help='threshold on NLL for old dataset')
	parser.add_argument('--W-old-path-list', type=str, nargs='+', default=[],
						help='path to the W matrices of source datasets')
	parser.add_argument('--U-old-path-list', type=str, nargs='+', default=[],
						help='path to the U matrices of source datasets')
	parser.add_argument('--W-old-vocab-path-list', type=str, nargs='+', default=[],
						help='path to the vocab of source datasets')
	parser.add_argument('--use-embeddings-prior', type=str, default="False",
						help='whether to use embeddings as prior or not')
	parser.add_argument('--lambda-embeddings', type=str, default="",
						help='make embeddings lambda trainable or not')
	parser.add_argument('--lambda-embeddings-list', type=float, nargs='+', default=[],
						help='list of lambda for every embedding prior')
	parser.add_argument('--reload-source-data-list', type=str, nargs='+', default=[],
						help='list of source datasets')
	parser.add_argument('--bias-W-old-path-list', type=str, nargs='+', default=[],
						help='path to the bias of W matrices of source datasets')
	parser.add_argument('--bias-U-old-path-list', type=str, nargs='+', default=[],
						help='path to the bias of U matrices of source datasets')
	parser.add_argument('--reload-source-multi-label', type=str, nargs='+', default=[], required=True,
						help='whether source datasets are multi-label or not')
	parser.add_argument('--reload-source-num-classes', type=int, nargs='+', default=[], required=True,
						help='number of classes in source datasets')

	args = parser.parse_args()
	
	# Convert string boolean arguments
	bool_args = [
		'reload', 'supervised', 'initialize_docnade', 'bidirectional', 'projection',
		'deep', 'multi_label', 'shuffle_reload', 'sal_loss', 'll_loss',
		'pretraining_target', 'bias_sharing', 'use_embeddings_prior'
	]
	
	for arg_name in bool_args:
		if hasattr(args, arg_name.replace('_', '-')) or hasattr(args, arg_name):
			attr_name = arg_name.replace('-', '_')
			if hasattr(args, attr_name):
				setattr(args, attr_name, Utils.str2bool(getattr(args, attr_name)))
	
	# Convert source multi-label arguments
	args.source_multi_label = [Utils.str2bool(value) for value in args.reload_source_multi_label]
 
	return args


if __name__ == '__main__':
	main(parse_args())
