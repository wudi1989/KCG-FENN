from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, \
    roc_auc_score,precision_recall_curve,auc,accuracy_score
import numpy as np
import torch
import torch.nn.functional as F

def eval_all(pred,label):
    pred_all = np.array(pred)
    label_all = np.array(label)

    precision, recall, _ = precision_recall_curve(label, pred)
    AUPR=auc(recall, precision)
    AUC = roc_auc_score(label_all, pred_all)

    for index,i in enumerate(pred):
        if i>0.5:
            pred[index]=1
        else:
            pred[index]=0

    Acc=accuracy_score(pred,label)
    Rec = recall_score(label, pred, average='binary')
    Pre = precision_score(label, pred, average='binary')
    F1 = f1_score(label, pred)

    return Acc,AUPR,AUC,F1,Pre,Rec

def evaluate_confidence(alphas, labels):
    probs = alphas[:, 1] / torch.sum(alphas, dim=1)
    confidences = torch.max(alphas / torch.sum(alphas, dim=1, keepdim=True), dim=1)[0]
    predictions = (probs > 0.5).long()

    num_bins = 15
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(labels)

    for i in range(num_bins):
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

def evaluate_confidence_standard_model(probs, labels):
    probs = torch.tensor(probs, dtype=torch.float)
    labels = torch.tensor(labels)

    confidences = torch.max(probs, 1 - probs)
    predictions = (probs > 0.5).long()

    num_bins = 15
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    for i in range(num_bins):
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].float().mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()