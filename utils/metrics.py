import torch

import torch

class Metrics:
    def __init__(self, num_classes, k=5):
        self.num_classes = num_classes
        self.k = k
        self.reset()
        
    def reset(self):
        self.total = torch.zeros(self.num_classes)
        self.tp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.top_k_correct = 0
        
    def update(self, preds, targets):
        # preds: (N, C) logits
        # targets: (N,) class indices
        _, pred = preds.topk(self.k, dim=1)  # (N, k)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))  # (N, k)
        self.top_k_correct += correct.any(dim=1).sum().item()
            
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)
            
        for i in range(self.num_classes):
            mask = targets == i
            self.tp[i] += (preds[mask] == i).sum().item() # TP
            self.fn[i] += (preds[mask] != i).sum().item() # FN
            self.fp[i] += (preds[~mask] == i).sum().item() # FP
            self.total[i] += mask.sum().item()
    
    def compute(self):
        # create mask for classes that appear in the dataset
        present_classes = self.total > 0
        
        # initialize metrics tensors
        accuracies = torch.zeros(self.num_classes)
        precisions = torch.zeros(self.num_classes)
        recalls = torch.zeros(self.num_classes)
        f1s = torch.zeros(self.num_classes)
        
        # calculate metrics only for present classes
        if present_classes.any():
            # accuracy
            accuracies[present_classes] = self.tp[present_classes] / self.total[present_classes]
            
            # precision
            denom = self.tp + self.fp
            precision_mask = present_classes & (denom > 0)
            precisions[precision_mask] = self.tp[precision_mask] / denom[precision_mask]
            
            # recall
            denom = self.tp + self.fn
            recall_mask = present_classes & (denom > 0)
            recalls[recall_mask] = self.tp[recall_mask] / denom[recall_mask]
            
            # f1
            denom = 2 * self.tp + self.fp + self.fn
            f1_mask = present_classes & (denom > 0)
            f1s[f1_mask] = 2 * self.tp[f1_mask] / denom[f1_mask]
        
        # calculate means only for present classes
        total_present = present_classes.sum().item()
        mean_accuracy = accuracies[present_classes].mean().item() if total_present > 0 else 0
        mean_precision = precisions[present_classes].mean().item() if total_present > 0 else 0
        mean_recall = recalls[present_classes].mean().item() if total_present > 0 else 0
        mean_f1 = f1s[present_classes].mean().item() if total_present > 0 else 0
        
        # calculate top-k accuracy
        total_samples = self.total.sum().item()
        top_k_accuracy = self.top_k_correct / total_samples if total_samples > 0 else 0
        
        return {
            'class_accuracies': [acc * 100 for acc in accuracies.tolist()],
            'class_precisions': [prec * 100 for prec in precisions.tolist()],
            'class_recalls': [rec * 100 for rec in recalls.tolist()],
            'class_f1s': [f * 100 for f in f1s.tolist()],
            'mean_accuracy': mean_accuracy * 100,
            'mean_precision': mean_precision * 100,
            'mean_recall': mean_recall * 100,
            'mean_f1': mean_f1 * 100,
            'top_k_accuracy': top_k_accuracy * 100,
            'present_classes': present_classes.tolist()
        }
