import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from torchmetrics.classification import BinaryCalibrationError


def remove_if_no_criteria_present(predictions_logits, predictions, labels, label_list):
	"""Removes criteria without any true positive instances or predictions within the inputted test set."""
	labs_preds = np.vstack((predictions, labels))
	all_zero_columns = np.where(np.all(labs_preds == 0, axis=0))[0]
	if all_zero_columns.size > 0:
		predictions_logits = np.delete(predictions_logits, all_zero_columns, axis=1)
		predictions = np.delete(predictions, all_zero_columns, axis=1)
		labels = np.delete(labels, all_zero_columns, axis=1)
		label_list = [x for i, x in enumerate(label_list) if i not in all_zero_columns]
	return predictions_logits, predictions, labels, label_list


def calculate_multilabel_metrics(predictions_logits, predictions, labels, label_list):
	"""
	Calculate a variety of performance metrics (precision, recall, f1, auc) for multi-label data.
	:param predictions_logits: list of lists or similar - prediction logits from model
	:param predictions: list of lists or similar - binary predictions from model
	:param labels: list of lists or similar - true labels
	:param label_list: list of strings with name of each label
	:return: dictionary of metrics
	"""
	predictions_logits, predictions, labels, label_list = remove_if_no_criteria_present(predictions_logits, predictions, labels, label_list)
	predictions_logits_arr = np.array([np.array(xi) for xi in predictions_logits])
	labels_arr = np.array([np.array(xi) for xi in labels])
    # Averaged metrics - micro and macro
	precision_micro_average = precision_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
	precision_macro_average = precision_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
	recall_micro_average = recall_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
	recall_macro_average = recall_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
	f1_micro_average = f1_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
	f1_macro_average = f1_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
	try:
		roc_auc_micro = roc_auc_score(labels, predictions_logits_arr, average = 'micro')
	except ValueError:
		roc_auc_micro = 'Value Error'
	try:
		roc_auc_macro = roc_auc_score(labels, predictions_logits_arr, average = 'macro')
	except ValueError:
		roc_auc_macro = 'Value Error'
	metrics = {'micro f1': f1_micro_average,
				'macro f1': f1_macro_average,
				'micro roc_auc': roc_auc_micro,
				'macro roc_auc': roc_auc_macro,
				'micro precision': precision_micro_average,
				'macro precision': precision_macro_average,
				'micro recall': recall_micro_average,
				'macro recall': recall_macro_average,
				}
	## Individual label metrics
	clf_dict = classification_report(labels, predictions, target_names=label_list, digits=4, zero_division=0, output_dict=True)
	# list of PT/SDs in core collection
	core_collections_list = ['case_reports', 'case-control_studies', 'clinical_studies_as_topic', 'clinical_study', 'clinical_trial', 'clinical_trial_protocol', 'cohort_studies', 'cross-over_studies', 'cross-sectional_studies', 'diagnostic_test_accuracy', 'double-blind_method', 'evaluation_study', 'follow-up_studies', 'longitudinal_studies', 'meta-analysis', 'multicenter_study', 'prospective_studies', 'random_allocation', 'randomized_controlled_trial_humans', 'retrospective_studies', 'systematic_review', 'systematic_reviews_as_topic', 'validation_study']
	core_collection_macro_precision = []
	core_collection_macro_recall = []
	core_collection_macro_f1 = []
	core_collection_macro_auc = []
	if roc_auc_macro == 'Value Error':
		roc_auc_macro_list = []
	for i, lab in enumerate(label_list):
		# P, R, F1
		metrics[f'{lab} precision'] = clf_dict[lab]['precision']
		metrics[f'{lab} recall'] = clf_dict[lab]['recall']
		metrics[f'{lab} f1'] = clf_dict[lab]['f1-score']
		# AUC for each label
		lab_preds = np.transpose(predictions_logits_arr[:, i])
		lab_labels = np.transpose(labels_arr[:, i])
		try:
			metrics[f'{lab} auc'] = roc_auc_score(lab_labels, lab_preds, average = None)
		except ValueError:
			metrics[f'{lab} auc'] = 'Value Error'
		if roc_auc_macro == 'Value Error':
			if metrics[f'{lab} auc'] != 'Value Error':
				roc_auc_macro_list.append(metrics[f'{lab} auc'])
			else:
				roc_auc_macro_list.append(0.5)
		
		# Core Collection Macro Averages
		if lab in core_collections_list:
			core_collection_macro_precision.append(clf_dict[lab]['precision'])
			core_collection_macro_recall.append(clf_dict[lab]['recall'])
			core_collection_macro_f1.append(clf_dict[lab]['f1-score'])
			if metrics[f'{lab} auc'] != 'Value Error':
				core_collection_macro_auc.append(metrics[f'{lab} auc'])
	
	if roc_auc_macro == 'Value Error':
		roc_auc_macro = sum(roc_auc_macro_list) / len(roc_auc_macro_list)
		metrics['macro roc_auc'] = roc_auc_macro
	
	metrics['core collections precision'] = sum(core_collection_macro_precision) / len(core_collection_macro_precision)
	metrics['core collections recall'] = sum(core_collection_macro_recall) / len(core_collection_macro_recall)
	metrics['core collections f1'] = sum(core_collection_macro_f1) / len(core_collection_macro_f1)
	metrics['core collections auc'] = sum(core_collection_macro_auc) / len(core_collection_macro_auc)

	## Calculate ECE
	ece_l1_metric = BinaryCalibrationError(n_bins=15, norm='l1')
	ece_l2_metric = BinaryCalibrationError(n_bins=15, norm='l2')
	ece_max_metric = BinaryCalibrationError(n_bins=15, norm='max')

	flat_logits = torch.tensor([x for xs in predictions_logits for x in xs], dtype=torch.float)
	flat_true_labels = torch.tensor([x for xs in labels for x in xs], dtype=torch.float)

	metrics['ece_l1'] = ece_l1_metric(flat_logits, flat_true_labels).item()
	metrics['ece_l2'] = ece_l2_metric(flat_logits, flat_true_labels).item()
	metrics['ece_max'] = ece_max_metric(flat_logits, flat_true_labels).item()

	return clf_dict, metrics

