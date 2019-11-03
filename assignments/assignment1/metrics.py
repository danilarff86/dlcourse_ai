import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = np.sum(prediction & ground_truth, dtype=np.float32)
    fp = np.sum(prediction & (ground_truth == False), dtype=np.float32)
    fn = np.sum((prediction == False) & ground_truth, dtype=np.float32)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = np.sum(prediction == ground_truth, dtype=np.float32) / prediction.shape[0]
    f1 = 2 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return np.sum(prediction == ground_truth, dtype=np.float32) / prediction.shape[0]
