import numpy as np
# from transformers import AdamW, get_linear_schedule_with_warmup
from models.bert_model import BertForMultiLabelClassification, MultiLabelBertForCL, self_contrastive_loss, strict_contrastive_loss, jaccard_similarity_contrastive_loss
from data import collate_fn, data_load
import torch
from torch.utils.data import DataLoader
import tqdm
import random
import math
import pickle
import time, os, json, csv
from argparse import ArgumentParser
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report, precision_recall_curve
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# from came_pytorch import CAME

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


def calculate_multilabel_metrics(predictions_logits, predictions, labels, label_list):
	"""
	Calculate a variety of performance metrics (precision, recall, f1, auc) for multi-label data.
	:param predictions_logits: list of lists or similar - prediction logits from model
	:param predictions: list of lists or similar - binary predictions from model
	:param labels: list of lists or similar - true labels
	:param label_list: list of strings with name of each label
	:return: dictionary of metrics
	"""
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	predictions_logits_arr = sigmoid(np.array(predictions_logits))
	labels_arr = np.array([np.array(xi) for xi in labels])
    # Averaged metrics - micro and macro
	precision_micro_average = precision_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
	precision_macro_average = precision_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
	recall_micro_average = recall_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
	recall_macro_average = recall_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
	f1_micro_average = f1_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
	f1_macro_average = f1_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
	roc_auc_micro = roc_auc_score(labels, predictions_logits_arr, average = 'micro')
	roc_auc_macro = roc_auc_score(labels, predictions_logits_arr, average = 'macro')
	metrics = {'micro f1': f1_micro_average,
				'macro f1': f1_macro_average,
				'micro roc_auc': roc_auc_micro,
				'macro roc_auc': roc_auc_macro,
				'micro precision': precision_micro_average,
				'macro precision': precision_macro_average,
				'micro recall': recall_micro_average,
				'macro recall': recall_macro_average,
				}
	# Individual label metrics
	clf_dict = classification_report(labels, predictions, target_names=label_list, digits=4, zero_division=0, output_dict=True)
	for i, lab in enumerate(label_list):
		# P, R, F1
		metrics[f'{lab} precision'] = clf_dict[lab]['precision']
		metrics[f'{lab} recall'] = clf_dict[lab]['recall']
		metrics[f'{lab} f1'] = clf_dict[lab]['f1-score']
		# AUC for each label
		lab_preds = np.transpose(predictions_logits_arr[:, i])
		lab_labels = np.transpose(labels_arr[:, i])
		metrics[f'{lab} auc'] = roc_auc_score(lab_labels, lab_preds, average = None)
	return clf_dict, metrics


def round_using_optimal_f1_threshold(config, predictions_logits, labels, list_of_label_names, log_dir):
	"""
	Round multi-label predictions for optimal F1, where each label is rounded using its own optimal threshold.
	:param predictions_logits: list of lists or similar - prediction logits from model
	:param labels: list of lists or similar - true labels
	:param label_list: list of strings with name of each label
	:param log_dir: string of base directory, where to save optimal thresholds associated with model
	:return: list of lists or similar - binary predictions from model
	"""
	optimal_thesholds = dict()
	labels = np.array([np.array(xi) for xi in labels])
	predictions_logits = np.array([np.array(xi) for xi in predictions_logits])
	predictions_binary = np.zeros(predictions_logits.shape)
	threshold_file = os.path.join(log_dir, 'best_thresholds.pkl')
	if config.train_val_test == 'train' or config.train_val_test == 'val':
		try:
			with open(threshold_file, 'rb') as file:
				optimal_thesholds = pickle.load(file)
			for i in range(len(list_of_label_names)):
				predictions_binary[:, i] = np.where(predictions_logits[:, i] > optimal_thesholds[list_of_label_names[i]], 1, 0)
		except FileNotFoundError:
			for i in range(len(list_of_label_names)):
				precision, recall, thresholds = precision_recall_curve(labels[:, i], predictions_logits[:, i])
				f1_scores = np.divide(2 * recall * precision, recall + precision, out=np.zeros_like(recall + precision), where=(recall + precision!=0))
				max_f1_thresh = thresholds[np.argmax(f1_scores)]
				optimal_thesholds[list_of_label_names[i]] = max_f1_thresh
				predictions_binary[:, i] = np.where(predictions_logits[:, i] > max_f1_thresh, 1, 0)
	else:
		with open(threshold_file, 'rb') as file:
			optimal_thesholds = pickle.load(file)
		for i in range(len(list_of_label_names)):
			predictions_binary[:, i] = np.where(predictions_logits[:, i] > optimal_thesholds[list_of_label_names[i]], 1, 0)
	return predictions_binary, optimal_thesholds


def evaluate(model, loss_fn, data, config, batch_num, list_name=[], epoch=0, name='', log_dir=None):
	progress = tqdm.tqdm(total=batch_num, ncols=75, desc='{} {}'.format(name, epoch))
	binary_predictions = []
	sid = []
	running_loss = 0.0
	true_labels = []
	probability_predictions = []

	with torch.no_grad():
		for batch in DataLoader(data, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn):
			model.eval()
			target = batch.labels
			combine = model(batch)
			combine = torch.sigmoid(combine)
			if not config.align_for_comparison_with_v1:
				loss = loss_fn(combine, target)
				running_loss += loss.item()

			if config.use_gpu:
				target = target.cpu().data
				combine = combine.cpu().data
			
			if config.align_for_comparison_with_v1:
				combine = combine.numpy()
				idx2remove = [51, 44, 9, 21, 57, 58, 5, 38, 48, 56]  # indices of labels not used in v1
				filtered_combine = np.delete(combine, idx2remove, 1)
				combine = filtered_combine
			
			true_labels.extend(target.tolist()[:config.eval_batch_size])
			probability_predictions.extend(combine.tolist()[:config.eval_batch_size])

			sid_single = batch.PMID
			
			sid.extend(sid_single[:config.eval_batch_size])
			progress.update(1)
		progress.close()
	
	avg_running_loss = running_loss/len(data)
	if config.train_val_test != 'train':
		binary_predictions, optimal_thesholds = round_using_optimal_f1_threshold(config, probability_predictions, true_labels, list_name, log_dir)
	else:
		optimal_thesholds = 0
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
	
	return sid, avg_running_loss, true_final, binary_predictions, preds_final, base_performance_metrics, all_performance_metrics, optimal_thesholds


def train(config):
	torch.autograd.profiler.profile(False)
	torch.autograd.profiler.emit_nvtx(False)
	torch.autograd.set_detect_anomaly(False)

	if config.bert_model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
		model_name = "PubMedBERT"
	base_folder = model_name + "_verbalize-" + str(config.verbalize) + "_verbalizemissing-" + str(config.verbalize_missing) + "_contrastive_loss-" + str(config.contrastive_loss)
	
	if config.save:
		if not os.path.exists(base_folder):
			os.mkdir(base_folder)
		timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
		log_home_dir = os.path.join(base_folder, timestamp)
		os.mkdir(log_home_dir)
	
	train_dataset, val_dataset, test_dataset, list_name = data_load(config)  # list_name is list of labels
	num_labels = len(list_name)

	if config.save:
		timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
		log_dir = os.path.join(log_home_dir, timestamp)
		os.mkdir(log_dir)
		log_file = os.path.join(log_dir, 'log.txt')
		best_thresholds = os.path.join(log_dir, 'best_thresholds.pkl')
		best_macro_f1_model = os.path.join(log_dir, 'best_f1.mdl')
		best_model_predictions = os.path.join(log_dir, 'best_predictions.csv')
		with open(log_file, 'w', encoding='utf-8') as w:
			w.write(str(config) + '\n')

	if config.use_gpu and config.gpu_device >= 0:
		torch.cuda.set_device(config.gpu_device)

	if not config.contrastive_loss:
		model = BertForMultiLabelClassification(config, num_labels)
	else:
		model = MultiLabelBertForCL(config, num_labels)
	
	# model = torch.compile(model)
	model.train()
	batch_num = len(train_dataset) // config.batch_size
	eval_steps = math.floor((len(train_dataset) / config.batch_size) / config.eval_steps)
	total_steps = batch_num * config.max_epoch
	test_batch_num = len(val_dataset) // config.eval_batch_size + (len(val_dataset) % config.eval_batch_size != 0)

	if config.use_gpu:
		model.cuda()
	param_groups = [
		{
			'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
			'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
		},
		{
			'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
			'lr': config.learning_rate, 'weight_decay': config.weight_decay
		},
	]

	optimizer = torch.optim.AdamW(params=param_groups, eps=config.learning_rate)

	if config.lr_scheduler:
		scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			total_iters=total_steps,
		)
	scaler = torch.cuda.amp.GradScaler()

	loss_fn = torch.nn.BCEWithLogitsLoss()

	best_macro_f1, best_epoch = 0, 0

	for epoch in range(config.max_epoch):
		running_loss = 0.0
		step = 0
		progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))
		optimizer.zero_grad(set_to_none=True)
		for batch_idx, batch in enumerate(DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)):
			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(device_type='cuda', dtype=torch.float16):
				prediction = model(batch)
				loss = loss_fn(prediction, batch.labels)
				if config.contrastive_loss:
					z1, z2, z3 = model(batch, cl_emb=True)
					if config.contrastive_loss == 'unsup':
						loss = (loss * (1 - config.cl_alpha)) + (self_contrastive_loss(model, z1, z2, z3) * config.cl_alpha)
					if config.contrastive_loss == 'scl':
						loss = (loss * (1 - config.cl_alpha)) + (strict_contrastive_loss(model, z1, z2, z3) * config.cl_alpha)
					if config.contrastive_loss == 'jscl':
						loss = (loss * (1 - config.cl_alpha)) + (jaccard_similarity_contrastive_loss(model, z1, z2, z3, batch) * config.cl_alpha)

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			scaler.step(optimizer)
			scaler.update()
			if config.lr_scheduler:
				scheduler.step()
			running_loss += loss.item()
			progress.update(1)
			step += 1
			if step % eval_steps == 0:
				print('\nINFO: Training loss is ', running_loss/step)
				if epoch >= 6:
					sid, eval_avg_loss, target_result, binary_predictions, valid_result, base_performance_metrics, all_performance_metrics, optimal_thesholds = evaluate(
						model, loss_fn, val_dataset, config, test_batch_num, list_name, epoch, 'TEST', log_dir
						)
					print(base_performance_metrics)
					if config.save:
						result = json.dumps({'epoch': epoch, 'train_loss': running_loss/len(train_dataset), 'eval_loss': eval_avg_loss, 'performance_metrics': all_performance_metrics})
						with open(log_file, 'a', encoding='utf-8') as w:
							w.write(result + '\n')
						print('\nINFO: Log file: ', log_file)
						if all_performance_metrics['macro f1'] > best_macro_f1:
							# with open(best_thresholds, 'wb') as file:
							# 	pickle.dump(optimal_thesholds, file)
							best_epoch = epoch
							best_macro_f1 = all_performance_metrics['macro f1']
							torch.save(dict(model=model.state_dict(), config=config), best_macro_f1_model)
							# saves predictions of best model
							with open(best_model_predictions, "w", newline='') as f:
								writer = csv.writer(f)
								header = ['pmid', 'true label', 'prediction']
								writer.writerow(header)
								for i in range(len(sid)):
									content = [sid[i], target_result[i], valid_result[i]]
									writer.writerow(content)
					# Early Stopping - if the macro f1 hasn't improved in the provided number of epochs, then stop training
					if config.early_stopping:
						if epoch - best_epoch >= config.early_stopping:
							break

		progress.close()

	if config.save:
		best = json.dumps({'best epoch': best_epoch})
		with open(log_file, 'a', encoding='utf-8') as w:
			w.write(best + '\n')


def test(config):
	train_dataset, val_dataset, test_dataset, list_name = data_load(config)  # list_name is list of labels
	num_labels = len(list_name)

	if config.align_for_comparison_with_v1:
		num_labels = 59
		# # 'scientific_integrity_review', 'published_erratum', 'clinical_trial_protocol', 'expression_of_concern', 'veterinary_observational_study', 'veterinary_randomized_controlled_trial', 'clinical_conference', 'newspaper_article', 'retraction_of_publication', 'veterinary_clinical_trial'

	if not config.contrastive_loss:
		model = BertForMultiLabelClassification(config, num_labels)
	else:
		model = MultiLabelBertForCL(config, num_labels)

	checkpoint_path = os.path.join(config.checkpoint, 'best_f1.mdl')
	model.load_state_dict(torch.load(checkpoint_path)['model'], strict=False)
	if config.use_gpu:
		model.cuda()

	if config.train_val_test == 'val':
		dataset = val_dataset
	else:
		dataset = test_dataset
	loss_fn = torch.nn.BCEWithLogitsLoss()
	test_batch_num = len(dataset) // config.eval_batch_size + (len(dataset) % config.eval_batch_size != 0)
	sid, eval_avg_loss, target_result, binary_predictions, valid_result, base_performance_metrics, all_performance_metrics, optimal_thesholds = evaluate(model, loss_fn, dataset, config, test_batch_num, list_name, log_dir=config.checkpoint)

	if config.train_val_test == 'val':
		best_thresholds = os.path.join(config.checkpoint, 'best_thresholds.pkl')
		with open(best_thresholds, 'wb') as file:
			pickle.dump(optimal_thesholds, file)

	if config.save:
		if config.align_for_comparison_with_v1:
			performance_save_path = os.path.join(config.checkpoint, f'v1_aligned_{config.train_val_test}_performance.txt')
		else:
			performance_save_path = os.path.join(config.checkpoint, f'{config.train_val_test}_performance.txt')
		with open(performance_save_path, 'w', encoding='utf-8') as w:
			w.write(str(all_performance_metrics))
		if config.align_for_comparison_with_v1:
			predictions_save_path = os.path.join(config.checkpoint, f'v1_aligned_{config.train_val_test}_predictions.txt')
		else:
			predictions_save_path = os.path.join(config.checkpoint, f'{config.train_val_test}_predictions.txt')
		with open(predictions_save_path, "w", newline='') as f:
			writer = csv.writer(f)
			header = ['pmid', 'true label', 'binary_prediction', 'labeled_prediction']
			writer.writerow(header)
			for i in range(len(sid)):
				content = [sid[i], target_result[i], binary_predictions[i], valid_result[i]]
				writer.writerow(content)


if __name__ == '__main__':
	parser = ArgumentParser()
	
	parser.add_argument('--label_file', type=str, help='path to the labels file')   

	parser.add_argument('--data_file', type=str, help='path to the data file')

	parser.add_argument('--train_val_test', type=str, help='what to do - train (fine-tune) or validate (eval) or test (eval)')

	parser.add_argument('--undersampling', type=float, help='proportion of majority classes')

	parser.add_argument('--undersampling_min_thresh', type=int, default=700, help='proportion of majority classes')  # 1k (min threshold set during initiial stratification) x 70% (train proportion) = 700

	parser.add_argument('--verbalize', type=str, help='whether to verbalize non-missing data')

	parser.add_argument('--verbalize_missing', type=str, help='whether to verbalize missing data')

	parser.add_argument('--contrastive_loss', type=str, default='', help='which CL to use: unsup, scl, jscl; leave empty for none')

	parser.add_argument('--cl_temp', type=float, default=0.05, help='temperature for cl function')

	parser.add_argument('--cl_alpha', type=float, default=0.5, help='alpha for cl function - higher values places more emphasis on learning from cl vs. label prediction')

	parser.add_argument('--remove_feature', type=str, default='', help='name a single feature to remove for ablation')   

	parser.add_argument('--use_gpu', type=bool, default=1, help='if or not use GPU, choose from True or False')   

	parser.add_argument('--gpu_device', type=int, default=0, help='number of GPU devices')

	parser.add_argument('--save', type=int, default=1, help='create log file or not, choose either 1 or 0')

	parser.add_argument('--bert_model_name', type=str, help='pretrained language model name, choose from huggingface')

	parser.add_argument('--bert_dropout', type=float, default=0.1, help='dropout rate')

	parser.add_argument('--batch_size', type=int, default=4, help='batch size')

	parser.add_argument('--eval_batch_size', type=int, default=4, help='batch size')

	parser.add_argument('--eval_steps', type=int, default=3, help='batch size')

	parser.add_argument('--max_epoch', type=int, default=20, help='number of epoch')

	parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate of linear neural network') 

	parser.add_argument('--bert_weight_decay', type=float, default=0, help='weight decay rate of the bert model') 

	parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate of linear neural network') 

	parser.add_argument('--bert_learning_rate', type=float, default=1e-5, help='learning rate of bert finetuning')   

	parser.add_argument('--early_stopping', type=int, default=4, help='early stopping')   

	parser.add_argument('--lr_scheduler', type=bool, default=True, help='learning rate scheduler')

	parser.add_argument('--align_for_comparison_with_v1', type=str, default='', help='align data/labels for comparison with multitagger v1')

	parser.add_argument('--checkpoint', type=str, default='', help='location of the directory containing checkpoint file')
	
	config = parser.parse_args()
	print("config: ", config)
	
	if config.train_val_test == 'train':
		train(config)
	elif config.train_val_test == 'val' or config.train_val_test == 'test':
		if not config.checkpoint:
			raise Exception('No checkpoint argument detected. Please fine-tune a checkpoint and add location as an argument.')
		test(config)
