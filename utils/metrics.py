from torch import Tensor
import numpy as np


class SimpleClassificationEvaluator:
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.n_correct, self.n_tot = 0, 0

    def reset(self) -> None:
        self.n_correct, self.n_tot = 0, 0

    def update(self, pred_labels: Tensor, gt_labels: Tensor) -> None:
        """ Add labels of a batch of images

        Args:
            pred_labels: [K]
            gt_labels: [K]
        """
        assert pred_labels.shape == gt_labels.shape
        self.n_correct += (pred_labels == gt_labels).sum().item()
        self.n_tot += pred_labels.shape[0]

    def Accuracy(self) -> float:
        return self.n_correct / self.n_tot


class ClassificationEvaluator:
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.conf_mat = np.zeros((n_classes, n_classes))

    def reset(self) -> None:
        self.conf_mat = np.zeros((self.n_classes, self.n_classes))

    def update(self, pred_labels: Tensor, gt_labels: Tensor) -> None:
        """ Add labels of a batch of images and update confusion matrix

        Args:
            pred_labels: [K]
            gt_labels: [K]
        """
        assert pred_labels.shape == gt_labels.shape
        indices = gt_labels * self.n_classes + pred_labels
        conf_mat = indices.bincount(minlength=self.n_classes**2).reshape(self.n_classes, self.n_classes)
        self.conf_mat += conf_mat.cpu().numpy()

    def Accuracy(self) -> float:
        return np.diag(self.conf_mat).sum() / self.conf_mat.sum()

    def Macro_F1(self) -> tuple[float, float, float]:
        """ Calculate precision, recall and macro f1 """
        TP = np.diag(self.conf_mat)
        FN = np.sum(self.conf_mat, axis=1) - TP
        FP = np.sum(self.conf_mat, axis=0) - TP
        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        f1score = 2 * precision * recall / (precision + recall + 1e-12)
        return precision, recall, f1score.mean()


def _test():
    import torch
    pred = torch.tensor([0, 0, 1, 2, 1, 1, 2, 1, 2])
    gt = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])
    evaluator = ClassificationEvaluator(n_classes=3)
    evaluator.update(pred[0:5], gt[0:5])
    evaluator.update(pred[5:], gt[5:])
    print(evaluator.Accuracy())
    print(evaluator.Macro_F1())

    evaluator = SimpleClassificationEvaluator(n_classes=3)
    evaluator.update(pred[0:3], gt[0:3])
    evaluator.update(pred[3:], gt[3:])
    print(evaluator.Accuracy())


if __name__ == '__main__':
    _test()
