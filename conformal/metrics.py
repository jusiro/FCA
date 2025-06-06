
import numpy as np
from sklearn.metrics import confusion_matrix


def aca(output, target):

    # Confusion matrix
    cm = confusion_matrix(target, np.argmax(output, -1))
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    # Accuracy per class - and average
    aca = np.round(np.mean(np.diag(cm_norm) * 100), 2)

    return aca


def accuracy(output, target, topk=(1,)):
    output, target = output.to("cpu"), target.to("cpu")

    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_conformal(pred_sets, labels, alpha=0.1):

    size = set_size(pred_sets)
    coverage = empirical_set_coverage(pred_sets, labels)
    class_cov_gap = avg_class_coverage_gap(pred_sets, labels, alpha=alpha)
    # class_cov = avg_class_coverage(pred_sets, labels)

    """
    #size_classwise = np.array([set_size([pred_sets[i] for i in range(len(pred_sets)) if labels[i] == i_label]) for i_label in np.unique(labels)])
    # each row one label, each column a number of bins
    setsizes_classwise = np.concatenate([np.expand_dims(np.array([[len(pred_sets[i]) for i in range(len(pred_sets)) if labels[i] == i_label].count(i_size+1) for i_label in np.unique(labels)]), 1) for i_size in np.unique(labels)], axis=1)
    setsizes_classwise = (setsizes_classwise / np.expand_dims(np.sum(setsizes_classwise, -1), 1))

    # occurence_classwise
    extend_label, extend_preds = [], []
    for i in range(len(pred_sets)):
        for i_pred in pred_sets[i]:
            extend_label.append(labels[i].item()), extend_preds.append(i_pred)
    cm = np.float32(confusion_matrix(extend_label, extend_preds))
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))
    """

    return [coverage, size, class_cov_gap]


def set_size(pred_sets):
    """
        Compute the size of the predicted sets.
        arguments:
            pred_sets [numpy.array]: predicted sets
        returns:
            sizes [numpy.array]: mean size of the predicted sets
    """
    mean_size = np.mean([len(pred_set) for pred_set in pred_sets])
    return mean_size


def empirical_set_coverage(pred_sets, labels):
    """
        Compute the empirical coverage of the predicted sets.
        arguments:
            pred_sets [numpy.array]: predicted sets
            labels [numpy.array]: true labels
        returns:
            coverage [float]: empirical coverage
    """
    coverage = np.mean([label in pred_set for label, pred_set in zip(labels, pred_sets)])
    return coverage


def avg_class_coverage_gap(pred_sets, labels, alpha=0.1):

    # Get sample-wise accuracy
    correct = np.int8([labels[i] in pred_sets[i] for i in range(len(labels))])

    violation = []
    for i_label in list(np.unique(labels)):
        idx = np.argwhere(labels == i_label)
        violation.append(abs(correct[idx].mean() - (1 - alpha)))

    # Get mean violation
    covgap = 100 * np.mean(violation)

    return covgap


def avg_class_coverage(pred_sets, labels):

    # Get sample-wise accuracy
    correct = np.int8([labels[i] in pred_sets[i] for i in range(len(labels))])

    clas_cov = []
    for i_label in list(np.unique(labels)):
        idx = np.argwhere(labels == i_label)
        clas_cov.append(correct[idx].mean())

    # Get mean violation
    clas_cov = np.mean(clas_cov)

    return clas_cov