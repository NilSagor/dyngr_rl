import torch 
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any, Tuple


def compute_ap(predictions, labels):
    pred_np = predictions.detach().cpu().numpy()
    label_np = labels.detach().cpu().numpy()

    if len(np.unique(label_np)) == 1:
        return 1.0 if label_np[0] == 1 else 0.0
    return average_precision_score(label_np, pred_np)

def compute_auc(predictions, labels):
    pred_np = predictions.detach().cpu().numpy()
    label_np = labels.detach().cpu().numpy()

    if len(np.unique(label_np)) == 1:
        return 1.0 if label_np[0] == 1 else 0.0
    
    return roc_auc_score(label_np, pred_np)

def compute_mrr(predictions, labels, num_negatives=1):
    batch_size = predictions.size(0) // (num_negatives+1)
    predictions = predictions.view(batch_size, num_negatives+1)

    ranks = torch.argsort(torch.argsort(predictions, descending=True), dim=1)
    positive_ranks = ranks[:, 0]+1

    reciprocal_ranks = 1.0/positive_ranks.float()

    return reciprocal_ranks.mean().item()


def compute_accuracy(predictions, labels, threshold=0.5):
    binary_preds = (predictions>threshold).float()
    correct = (binary_preds == labels).float().sum()
    accuracy = correct/len(labels)
    return accuracy.item()

def compute_precision_recall_f1(predictions,labels, threshold=0.5):
    binary_preds = (predictions>threshold).float()
    tp = ((binary_preds == 1) & (labels==1)).float().sum()
    fp = ((binary_preds == 1) & (labels==0)).float().sum()
    fn = ((binary_preds == 0) & (labels==1)).float().sum()

    precision = tp/(tp+fp+1e-8)
    recall = tp/(tp+fn+1e-8)
    f1 = 2* (precision*recall)/(precision*recall+1e-8)
    return precision.item(), recall.item(), f1.item()

def evaluate_model(model, dataloader, device, metrics=["accuracy", "ap", "auc"]):
    model.eval()
    all_predictions = []
    all_lables = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()} 

        if hasattr(model, 'predict'):
            predictions = model.predict(batch)
        else:
            logits=model(batch)
            predictions = torch.sigmoid(logits)
        labels = batch['labels']

        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    results={}

    if 'accuracy' in metrics:
        results['accuracy'] = compute_accuracy(all_predictions, all_lables)

    if 'ap' in metrics:
        results['ap'] = compute_ap(all_predictions, all_lables)
    
    if 'auc' in metrics:
        results['auc'] = compute_auc(all_predictions, all_lables)
    
    if 'mrr' in metrics:
        results['mrr'] = compute_mrr(all_predictions, all_lables)

    if 'precision' in metrics or 'recall' in metrics or 'f1' in metrics:
        precision, recall, f1 = compute_precision_recall_f1(all_predictions, all_lables)
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
    return results

def temporal_link_prediction_evaluation(model, test_data, device, num_negative_samples):
    model.eval()
    src_nodes = test_data['src_nodes'].to(device)
    dst_nodes = test_data['dst_nodes'].to(device)
    timestamps = test_data['timestamps'].to(device)
    
    batch_size = len(src_nodes)

    all_dst_nodes = []
    all_labels = []

    for i in range(batch_size):
        all_dst_nodes.append(dst_nodes[i:i+1])
        all_labels.append(torch.tensor([1.0], device=device))

        neg_dst = torch.randint(0, model.num_nodes, (num_negative_samples,), device=device)
        all_dst_nodes.append(neg_dst)
        all_labels.append(torch.zeros(num_negative_samples, device=device))
    
    # concatenate
    all_dst_nodes = torch.cat(all_dst_nodes)
    all_labels = torch.cat(all_labels)

    all_src_nodes = src_nodes.repeat_interleave(num_negative_samples+1)
    all_timestamps = timestamps.repeat_interleave(num_negative_samples+1)

    batch = {
        "src_nodes": all_src_nodes,
        "dst_nodes": all_dst_nodes,
        "timestamps": all_timestamps,
        "labels": all_labels,
    }

    with torch.no_grad():
        if hasattr(model, 'predict'):
            predictions = model.predict(batch)
        else:
            logits = model(batch)
            predictions = torch.sigmoid(logits)

    results = {
        "mrr" : compute_mrr(predictions, all_labels, num_negative_samples),
        "ap": compute_ap(predictions, all_labels),
        "auc": compute_auc(predictions, all_labels)
    }
    
    return results


def ranking_evaluation(predictions, labels, k_values):
    # sort predictions
    sorted_indices = torch.argsort(predictions, descending=True)
    sorted_labels = labels[sorted_indices]
    results = {}
    for k in k_values:
        if k > len(sorted_labels):
            continue

        # recall@k
        relevant_at_k = sorted_labels[:k].sum().item()
        total_relevant = labels.sum().item()
        recall_k = relevant_at_k/total_relevant if total_relevant > 0 else 0
        results[f'recall@k{k}'] = recall_k
        # Precision@k
        precision_k = relevant_at_k / k
        results[f'precision@{k}'] = precision_k
    return results