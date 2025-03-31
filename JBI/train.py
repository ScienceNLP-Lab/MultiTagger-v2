import numpy as np
from functools import partial
from data import collate_fn, data_load
from models.losses import BCEWithLabelSmoothing, FocalWithLogitsLoss, AsymmetricLossOptimized, AsymmetricLossOptimized_LabelSmoothing
from models.models import BertForMultiLabelClassification, Specter2ForMultiLabelClassification, BertForMultiLabelClassificationHeroCon
import torch
from torch.utils.data import DataLoader
import tqdm
import random
import math
from ast import literal_eval
import time, os, json, csv
from argparse import ArgumentParser
from metrics import calculate_multilabel_metrics
from sklearn.metrics import precision_recall_curve

import torch._dynamo
torch._dynamo.reset()

def set_random_seed(seed_input: int):
	random.seed(seed_input)
	np.random.seed(seed_input)
	torch.manual_seed(seed_input)


def set_device(args):
	if args.use_gpu:
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device('cpu')
	args.device = device
	return args


def round_using_optimal_f1_threshold(predictions_logits, labels, list_of_label_names, optimal_thesholds):
	"""
	Round multi-label predictions for optimal F1, where each label is rounded using its own optimal threshold.
	:param predictions_logits: list of lists or similar - prediction logits from model
	:param labels: list of lists or similar - true labels
	:param list_of_label_names: list of strings with name of each label
	:param optimal_thesholds: a dict containing optimal threshold cut-offs
	:return: list of lists or similar - binary predictions from model
	"""
	labels = np.array([np.array(xi).astype(np.int32) for xi in labels])
	predictions_logits = np.array([np.array(xi) for xi in predictions_logits])
	predictions_binary = np.zeros(predictions_logits.shape)
	if not optimal_thesholds:
		optimal_thesholds = dict()
		for i in range(len(list_of_label_names)):
			precision, recall, thresholds = precision_recall_curve(labels[:, i], predictions_logits[:, i])
			f1_scores = np.divide(2 * recall * precision, recall + precision, out=np.zeros_like(recall + precision), where=(recall + precision!=0))
			max_f1_thresh = thresholds[np.argmax(f1_scores)]
			optimal_thesholds[list_of_label_names[i]] = max_f1_thresh
			predictions_binary[:, i] = np.where(predictions_logits[:, i] > max_f1_thresh, 1, 0)
	else:
		for i in range(len(list_of_label_names)):
			try:
				predictions_binary[:, i] = np.where(predictions_logits[:, i] > optimal_thesholds[list_of_label_names[i]], 1, 0)
			except KeyError:
				predictions_binary[:, i] = np.where(predictions_logits[:, i] > 0.5, 1, 0)  # uses 0.5 if no key available
	return predictions_binary, optimal_thesholds


def evaluate(model, loss_fn, data, config, batch_num, list_name_=[], epoch=0, name='', optimal_thesholds=None):
	progress = tqdm.tqdm(total=batch_num, ncols=75, desc='{} {}'.format(name, epoch))
	list_name = list_name_.copy()
	sid = []
	running_loss = 0.0

	true_labels = []
	probability_predictions = []
	binary_predictions = []

	if config.label_split:
		split_true_labels = []
		split_probability_predictions = []
		if config.label_split == 'cohort':
			number_of_split_labels = 4
		elif config.label_split == 'generalized_rct' or config.label_split == 'animals' or config.label_split == 'humans' or config.label_split == 'veterinary':
			number_of_split_labels = 1
		elif config.label_split == 'combination':
			number_of_split_labels = 7
		list_name = list_name[:-number_of_split_labels]

	with torch.no_grad():
		for batch in DataLoader(data, batch_size=config.eval_batch_size, shuffle=False, collate_fn=partial(collate_fn, config=config)):
			model.eval()
			target = batch.labels

			if config.contrastive_loss == 'HeroCon' or config.contrastive_loss == 'WeighCon':
				losses, combine = model(batch)
			else:
				combine = model(batch)

			loss = loss_fn(combine, target) # losses use logits
			running_loss += loss.item()

			combine = torch.sigmoid(combine)

			if config.use_gpu:
				target = target.cpu().data
				combine = combine.cpu().data
			
			if config.label_split:
				combine = combine.numpy()
				num_cols = combine.shape[1]
				num_tar = target.shape[1]
				split_preds = combine[:, -number_of_split_labels:]
				combine = combine[:, :num_cols-number_of_split_labels]
				split_labels = target[:, -number_of_split_labels:]
				target = target[:, :num_tar-number_of_split_labels]
				
				split_true_labels.extend(split_labels.tolist()[:config.eval_batch_size])
				split_probability_predictions.extend(split_preds.tolist()[:config.eval_batch_size])
			
			true_labels.extend(target.tolist()[:config.eval_batch_size])
			probability_predictions.extend(combine.tolist()[:config.eval_batch_size])

			sid_single = batch.PMID
			
			sid.extend(sid_single[:config.eval_batch_size])
			progress.update(1)
		progress.close()
	
	avg_running_loss = running_loss/len(data)
	if config.optimal_f1_threshold:
		binary_predictions, optimal_thesholds = round_using_optimal_f1_threshold(probability_predictions, true_labels, list_name, optimal_thesholds)
	else:
		optimal_thesholds = None
		probability_predictions = np.array([np.array(xi) for xi in probability_predictions])
		binary_predictions = np.zeros(probability_predictions.shape)
		for i in range(len(list_name)):
			binary_predictions[:, i] = np.where(probability_predictions[:, i] > 0.5, 1, 0)
	
	
	base_performance_metrics, all_performance_metrics = calculate_multilabel_metrics(probability_predictions, binary_predictions, true_labels, list_name)

	true_final = []
	preds_final = []

	for i in true_labels:
		if sum(i) > 0.1:
			new_item_targets = [list_name[j] for j in range(len(i)) if i[j] == 1] 
			true_final.append(new_item_targets)
		else:
			true_final.append(["0"])
	
	for i in binary_predictions:
		if sum(i) > 0.1:
			new_item_valid = [list_name[j] for j in range(len(i)) if i[j] == 1] 
			preds_final.append(new_item_valid)
		else:
			preds_final.append(["0"])
	
	model.train()

	if isinstance(probability_predictions, np.ndarray):
		probability_predictions = probability_predictions.tolist()
	if isinstance(binary_predictions, np.ndarray):
		binary_predictions = binary_predictions.tolist()

	return sid, avg_running_loss, true_final, preds_final, base_performance_metrics, all_performance_metrics, probability_predictions, binary_predictions, true_labels, optimal_thesholds


def train(config):
	torch.autograd.profiler.profile(False)
	torch.autograd.profiler.emit_nvtx(False)
	torch.autograd.set_detect_anomaly(False)
	
	train_dataset, val_dataset, test_dataset, list_name, config = data_load(config)  # list_name is list of labels
	num_labels = len(list_name)

	if config.save:
		if config.continue_training:
			log_dir = config.checkpoint
		else:
			base_folder = os.path.join('experiment_logs', config.experiment_name)
			timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
			log_dir = os.path.join(base_folder, timestamp)
			os.makedirs(log_dir, exist_ok=True)
			config.checkpoint = log_dir
		log_fpath = os.path.join(log_dir, 'log.txt')
		best_model_fpath = os.path.join(log_dir, 'best_model.pth')
		with open(log_fpath, 'w', encoding='utf-8') as w:
			w.write(str(config) + '\n')

	if config.contrastive_loss == 'HeroCon' or config.contrastive_loss == 'WeighCon':
		model = BertForMultiLabelClassificationHeroCon(config, num_labels, clip=config.cl_clipping_val)
	else:
		if config.bert_model_name == "allenai/specter2_base" and config.bert_adapter:
			model = Specter2ForMultiLabelClassification(config, num_labels)
		else:
			model = BertForMultiLabelClassification(config, num_labels)
	
	model.train()
	batch_num = len(train_dataset) // config.batch_size
	eval_steps = math.floor((len(train_dataset) / config.batch_size) / config.eval_steps)
	total_steps = batch_num * config.max_epoch
	val_batch_num = len(val_dataset) // config.eval_batch_size + (len(val_dataset) % config.eval_batch_size != 0)
	if config.base_batch_size != config.batch_size:
		iters_to_accumulate = config.base_batch_size / config.batch_size
	else:
		iters_to_accumulate = 1

	model.to(config.device)
	
	param_groups = [
		{
			'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
			'lr': config.bert_learning_rate
		},
		{
			'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
			'lr': config.linear_learning_rate
		},
	]

	if config.optimizer == "RAdam":
		if config.regularization == 'decoupled_weight_decay':
			optimizer = torch.optim.RAdam(params=param_groups, eps=config.epsilon, weight_decay=config.optimizer_weight_decay, decoupled_weight_decay=True)
		else:
			optimizer = torch.optim.RAdam(params=param_groups, eps=config.epsilon)
	else:
		optimizer = torch.optim.AdamW(params=param_groups, eps=config.epsilon, weight_decay=config.optimizer_weight_decay)

	if config.lr_scheduler == 'linear':
		scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			total_iters=total_steps,
		)
	elif config.lr_scheduler == 'cosine':
		# See https://discuss.pytorch.org/t/how-to-implement-torch-optim-lr-scheduler-cosineannealinglr/28797/5 for implementation example
		# Delete after debugging
		if config.lr_scheduler_restarts == 'per_epoch':
			# Schedules over each batch - lowers the learning rate to its minimum in each epoch and then restart from the base learning rate
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
				optimizer,
				T_max = batch_num // iters_to_accumulate,
			)
		elif config.lr_scheduler_restarts == 'delayed':
			# Waits 5 epochs, then starts lr scheduler. Schedules over entire training
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
				optimizer,
				T_max = config.max_epoch - 5,
			)
		else:
			# Schedules over entire training
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
				optimizer,
				T_max = config.max_epoch,
			)
	
	scaler = torch.cuda.amp.GradScaler()

	if config.loss_function == 'bce':
		loss_fn = torch.nn.BCEWithLogitsLoss()
	elif config.loss_function == 'bce_ls': # bce w/ label smoothing
		loss_fn = BCEWithLabelSmoothing(num_labels, alpha=config.ls_alpha)
	elif config.loss_function == 'fl':  # focal loss
		loss_fn = FocalWithLogitsLoss()
	elif config.loss_function == 'asl':  # asymmetric loss
		loss_fn = AsymmetricLossOptimized()
	elif config.loss_function == 'asl_ls':  # asymmetric loss w/ label smoothing
		loss_fn = AsymmetricLossOptimized_LabelSmoothing(num_labels, alpha=config.ls_alpha)

	best_metric, best_epoch, start_epoch = 0, 0, 0

	if config.continue_training:
		checkpoint = torch.load(best_model_fpath, weights_only=True)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scaler.load_state_dict(checkpoint['scaler_state_dict'])
		if config.lr_scheduler:
			scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
		start_epoch = checkpoint['epoch']
		start_step = checkpoint['step']
		start = False

	break_flag = False
	n_iter = 0
	for epoch in range(start_epoch, config.max_epoch):
		running_loss = 0.0
		step = 0
		progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))
		optimizer.zero_grad(set_to_none=True)
		dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=partial(collate_fn, config=config))
		for batch_idx, batch in enumerate(dataloader):
			if config.continue_training:
				if not start:
					if step != start_step:
						step += 1
						continue
					else:
						start = True
			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(device_type='cuda', dtype=torch.float16):
				if config.contrastive_loss == 'HeroCon' or config.contrastive_loss == 'WeighCon':
					cl_losses, prediction = model(batch, training=True, sup_contrastive_mode=config.contrastive_loss, focusing=config.cl_focusing, clipping=config.cl_clipping)
					loss = loss_fn(prediction, batch.labels)
					loss = (loss + cl_losses[0] + cl_losses[1]).mean()  # asl + unsup + sup
				else:
					prediction = model(batch)
					loss = loss_fn(prediction, batch.labels)
				loss = loss / iters_to_accumulate

			scaler.scale(loss).backward()
			if ((batch_idx + 1) % iters_to_accumulate == 0) or (batch_idx + 1 == len(dataloader)):
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				scaler.step(optimizer)
				scaler.update()
				if config.lr_scheduler == 'linear' or config.lr_scheduler_restarts == 'per_epoch':
					scheduler.step()
			n_iter += 1
			running_loss += loss.item()
			progress.update(1)
			step += 1
			if step % eval_steps == 0:
				print('\nINFO: Training loss is ', running_loss/step)
				if epoch >= config.eval_after_epoch:
					sid, eval_avg_loss, target_result, valid_result, base_performance_metrics, all_performance_metrics, probability_predictions, binary_predictions, true_labels, optimal_thesholds = evaluate(
						model, loss_fn, val_dataset, config, val_batch_num, list_name_=list_name, epoch=epoch, name='VALID'
						)
					print('\nINFO: Validation loss is ', eval_avg_loss)
					print(f'\nINFO: Performance is {base_performance_metrics}')
					if config.save:
						result = json.dumps({'epoch': epoch, 'train_loss': running_loss/len(train_dataset), 'eval_loss': eval_avg_loss, 'performance_metrics': all_performance_metrics})
						with open(log_fpath, 'a', encoding='utf-8') as w:
							w.write(result + '\n')
						print('\nINFO: Log file: ', log_fpath)
						if config.select_model_using == 'macro':
							comparison_metric = all_performance_metrics['macro f1']
						elif config.select_model_using == 'micro':
							comparison_metric = all_performance_metrics['micro f1']
						elif config.select_model_using == 'core':
							comparison_metric = all_performance_metrics['core collections f1']
						if comparison_metric > best_metric:
							best_epoch = epoch
							best_metric = comparison_metric
							best_model = model
							best_thresholds = optimal_thesholds
							if config.lr_scheduler:
								torch.save({'config': config, 'epoch': epoch, 'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'lr_scheduler': scheduler.state_dict(), 'optimal_thesholds': optimal_thesholds}, best_model_fpath)
							else:
								torch.save({'config': config, 'epoch': epoch, 'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'optimal_thesholds': optimal_thesholds}, best_model_fpath)

							# saves predictions of best model
							performance_save_path = os.path.join(config.checkpoint, 'val_performance.txt')
							with open(performance_save_path, 'w', encoding='utf-8') as w:
								w.write(str(all_performance_metrics))

							if config.label_split:
								if config.label_split == 'cohort':
									number_of_split_labels = 4
								elif config.label_split == 'generalized_rct' or config.label_split == 'animals' or config.label_split == 'humans' or config.label_split == 'veterinary':
									number_of_split_labels = 1
								elif config.label_split == 'combination':
									number_of_split_labels = 7
								saved_list_name = list_name[:-number_of_split_labels]
							else:
								saved_list_name = list_name
							predictions_save_path = os.path.join(config.checkpoint, 'val_predictions.csv')
							with open(predictions_save_path, "w", newline='') as f:
								writer = csv.writer(f)
								header = ['pmid', 'named_true_label', 'named_pred', 'logits', 'binary_pred', 'binary_true_label', 'label_names']
								writer.writerow(header)
								for i in range(len(sid)):
									content = [sid[i], target_result[i], valid_result[i], probability_predictions[i], binary_predictions[i], true_labels[i], saved_list_name]
									writer.writerow(content)

					# Early Stopping - if the macro f1 hasn't improved in the provided number of epochs, then stop training
					if config.early_stopping:
						if epoch - best_epoch >= config.early_stopping:
							break_flag = True  # necessary to break inner & outer loop
							break
		if config.lr_scheduler == 'cosine':
			if config.lr_scheduler_restarts == 'per_epoch':
				# Reset scheduler per epoch
				scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
					optimizer,
					T_max = batch_num // iters_to_accumulate,
				)
			elif config.lr_scheduler_restarts == 'delayed' and epoch > 4:
				# Waits 5 epochs, then starts lr scheduler
				scheduler.step()
			elif config.lr_scheduler_restarts != 'delayed':
				# Schedules over entire training, stepping every epoch
				scheduler.step()
		if break_flag == True:
			break

		progress.close()

	if config.save:
		best = json.dumps({'best epoch': best_epoch})
		with open(log_fpath, 'a', encoding='utf-8') as w:
			w.write(best + '\n')
	
	# Get performance of best model on test set
	test_batch_num = len(test_dataset) // config.eval_batch_size + (len(test_dataset) % config.eval_batch_size != 0)
	sid, eval_avg_loss, target_result, valid_result, base_performance_metrics, all_performance_metrics, probability_predictions, binary_predictions, true_labels, optimal_thesholds = evaluate(best_model, loss_fn, test_dataset, config, test_batch_num, list_name_=list_name, epoch=epoch, name='TEST', optimal_thesholds=best_thresholds)
	if config.save:
		performance_save_path = os.path.join(config.checkpoint, 'test_performance.txt')
		with open(performance_save_path, 'w', encoding='utf-8') as w:
			w.write(str(all_performance_metrics))

		if config.label_split:
			if config.label_split == 'cohort':
				number_of_split_labels = 4
			elif config.label_split == 'generalized_rct' or config.label_split == 'animals' or config.label_split == 'humans' or config.label_split == 'veterinary':
				number_of_split_labels = 1
			elif config.label_split == 'combination':
				number_of_split_labels = 7
			saved_list_name = list_name[:-number_of_split_labels]
		else:
			saved_list_name = list_name
		predictions_save_path = os.path.join(config.checkpoint, 'test_predictions.csv')
		with open(predictions_save_path, "w", newline='') as f:
			writer = csv.writer(f)
			header = ['pmid', 'named_true_label', 'named_pred', 'logits', 'binary_pred', 'binary_true_label', 'label_names']
			writer.writerow(header)
			for i in range(len(sid)):
				content = [sid[i], target_result[i], valid_result[i], probability_predictions[i], binary_predictions[i], true_labels[i], saved_list_name]
				writer.writerow(content)



def test(config):
	train_dataset, val_dataset, test_dataset, list_name, config = data_load(config)  # list_name is list of labels
	num_labels = len(list_name)

	if config.contrastive_loss == 'HeroCon' or config.contrastive_loss == 'WeighCon':
		model = BertForMultiLabelClassificationHeroCon(config, num_labels, clip=config.cl_clipping_val)
	else:
		if config.bert_model_name == "allenai/specter2_base" and config.bert_adapter:
			model = Specter2ForMultiLabelClassification(config, num_labels)
		else:
			model = BertForMultiLabelClassification(config, num_labels)

	checkpoint_path = os.path.join(config.checkpoint, 'best_model.pth')
	checkpoint = torch.load(checkpoint_path)
	best_thresholds = checkpoint['optimal_thesholds']
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(config.device)

	if config.train_val_test == 'val':
		dataset = val_dataset
	else:
		dataset = test_dataset

	if config.loss_function == 'bce':
		loss_fn = torch.nn.BCEWithLogitsLoss()
	elif config.loss_function == 'bce_ls':
		loss_fn = BCEWithLabelSmoothing(num_labels, alpha=config.ls_alpha)
	elif config.loss_function == 'fl':  # focal loss
		loss_fn = FocalWithLogitsLoss()  # default parameters
	elif config.loss_function == 'asl':  # asymmetric loss
		loss_fn = AsymmetricLossOptimized()  # default parameters
	elif config.loss_function == 'asl_ls':  # asyemmetric loss with label smoothing
		loss_fn = AsymmetricLossOptimized_LabelSmoothing(num_labels, alpha=config.ls_alpha)  # default parameter

	eval_batch_num = len(dataset) // config.eval_batch_size + (len(dataset) % config.eval_batch_size != 0)
	sid, eval_avg_loss, target_result, valid_result, base_performance_metrics, all_performance_metrics, probability_predictions, binary_predictions, true_labels, optimal_thesholds = evaluate(model, loss_fn, dataset, config, eval_batch_num, list_name_=list_name, name='TEST', optimal_thesholds=best_thresholds)

	if config.save:
		if config.label_split:
			if config.label_split == 'cohort':
				number_of_split_labels = 4
			elif config.label_split == 'generalized_rct' or config.label_split == 'animals' or config.label_split == 'humans' or config.label_split == 'veterinary':
				number_of_split_labels = 1
			elif config.label_split == 'combination':
				number_of_split_labels = 7
			list_name = list_name[:-number_of_split_labels]

		performance_save_path = os.path.join(config.checkpoint, f'{config.train_val_test}_performance.txt')
		with open(performance_save_path, 'w', encoding='utf-8') as w:
			w.write(str(all_performance_metrics))

		predictions_save_path = os.path.join(config.checkpoint, f'{config.train_val_test}_predictions.csv')
		with open(predictions_save_path, "w", newline='') as f:
			writer = csv.writer(f)
			header = ['pmid', 'named_true_label', 'named_pred', 'logits', 'binary_pred', 'binary_true_label', 'label_names']
			writer.writerow(header)
			for i in range(len(sid)):
				content = [sid[i], target_result[i], valid_result[i], probability_predictions[i], binary_predictions[i], true_labels[i], list_name]
				writer.writerow(content)


if __name__ == '__main__':
	parser = ArgumentParser()

	# General hyperparameters
	parser.add_argument('--experiment_name', type=str, help='experiment name - directory name to save files')
	parser.add_argument('--set_seed', type=int, default=42, help='set random seed')
	parser.add_argument('--use_gpu', type=bool, default=1, help='if or not use GPU, choose from True or False')   
	parser.add_argument('--save', type=int, default=1, help='create log file or not, choose either 1 or 0')
	parser.add_argument('--checkpoint', type=str, default='', help='location of the directory containing checkpoint file')
	parser.add_argument('--continue_training', type=str, default='', help='To continue training, add some string here; requires checkpoint')

	# Hyperparameters related to labels, data and feature augmentation
	parser.add_argument('--label_file', type=str, default='data/labels_human/split_stratified_data.csv', help='path to the labels file')   
	parser.add_argument('--data_file', type=str, default='data/pubmed/pubmed_data.csv', help='path to the data file')
	parser.add_argument('--train_val_test', type=str, default='train', help='what to do - train (fine-tune) or validate (eval) or test (eval)')
	parser.add_argument('--label_split', type=str, default='', help='how to split labels - cohort or humans or animals or veterinary')
	parser.add_argument('--max_length', type=int, default=512, help='int of max input length for the model')
	parser.add_argument('--verbalize', type=str, default='short', help='whether to verbalize non-missing data')
	parser.add_argument('--verbalize_missing', type=str, default='', help='whether to verbalize missing data')
	parser.add_argument('--remove_feature', type=str, default='', help='name a single feature to remove for ablation')
	parser.add_argument('--full_text', type=str, default='', help='Prefer full-text articles for training')
	parser.add_argument('--align_full_text_only_comparison', type=str, default='', help='Restrict to full-text articles only for eval')

	# Hyperparameters related to the optimizer
	parser.add_argument('--optimizer', type=str, default='RAdam', help='Choice of optimizer - Adam or RAdam')
	parser.add_argument('--optimizer_weight_decay', type=float, default=0, help='weight decay rate used by optimizer')
	parser.add_argument('--epsilon', type=float, default=1e-2, help='optimizer epsilon value; added to denom to improve numerical stability')
	parser.add_argument('--regularization', type=str, default='', help='type of regulaization - decoupled weight decay')
	parser.add_argument('--lr_scheduler', type=str, default='', help='learning rate scheduler')
	parser.add_argument('--lr_scheduler_restarts', type=str, default='', help='learning rate scheduler')

	# Hyperparameters related to loss functions
	parser.add_argument('--loss_function', type=str, default='asl_ls', help='loss function to use for classification')
	parser.add_argument('--ls_alpha', type=float, default=0.05, help='alpha to control level of label smoothing')
	
	# Hyperparameters related to training, model independent
	parser.add_argument('--select_model_using', type=str, default='macro', help='macro, micro, or core - all f1 averages')
	parser.add_argument('--optimal_f1_threshold', type=int, default=1, help='(0 or 1) round using optimal threshold')
	parser.add_argument('--base_batch_size', type=int, default=32, help='batch size')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size during training')
	parser.add_argument('--eval_batch_size', type=int, default=32, help='batch size during evaluation')
	parser.add_argument('--eval_steps', type=int, default=1, help='eval steps')
	parser.add_argument('--max_epoch', type=int, default=25, help='number of epoch')
	parser.add_argument('--eval_after_epoch', type=int, default=0, help='start evaluating on validation set after provided epoch number')
	parser.add_argument('--early_stopping', type=int, default=4, help='early stopping') 

	# Hyperparameters related to the model - architecture or internal parameters
	parser.add_argument('--bert_model_name', type=str, default='allenai/specter2_base', help='pretrained language model name, choose from huggingface')
	parser.add_argument('--bert_adapter', type=int, default=0, help='whether to use classifier adapter for specter2 - 0 or 1')
	parser.add_argument('--bert_dropout', type=float, default=0.1, help='dropout rate')
	parser.add_argument('--layer_weight_decay', type=float, default=0, help='weight decay rate of linear neural network')
	parser.add_argument('--layer_bert_weight_decay', type=float, default=0, help='weight decay rate of the bert model')
	parser.add_argument('--linear_learning_rate', type=float, default=1e-2, help='learning rate of linear layer in neural network') 
	parser.add_argument('--bert_learning_rate', type=float, default=1e-4, help='learning rate of bert finetuning')
	
	# Hyperparameters related to contrastive learning experiments
	parser.add_argument('--contrastive_loss', type=str, default='', help='which CL to use: HeroCon, WeighCon; leave empty for none')
	parser.add_argument('--cl_temp', type=float, default=0.05, help='temperature for cl function')
	parser.add_argument('--cl_alpha', type=float, default=0, help='hyperparameter to increase effect of unsupervised cl')
	parser.add_argument('--adnce_w1', type=float, default=0, help='adnce hyperparameter - mu - central region weight allocation')
	parser.add_argument('--adnce_w2', type=float, default=0, help='adnce hyperparameter - sigma - height of weight in central region')
	parser.add_argument('--cl_beta', type=float, default=0, help='hyperparameter to increase effect of supervised cl')
	parser.add_argument('--cl_focusing', type=int, default=0, help='hyperparameter to focus supervised cl')
	parser.add_argument('--cl_clipping', type=int, default=0, help='hyperparameter to clip supervised cl')
	parser.add_argument('--cl_clipping_val', type=float, default=0.95, help='hyperparameter to clip supervised cl - cutoff value')

	config = parser.parse_args()
	
	set_random_seed(config.set_seed)
	config = set_device(config)
	
	print("config: ", config)

	if config.train_val_test == 'val' or config.train_val_test == 'test':
		if not config.checkpoint:
			raise Exception('No checkpoint argument detected. Please fine-tune a checkpoint and add location as an argument.')
		test(config)
	else:
		train(config)
