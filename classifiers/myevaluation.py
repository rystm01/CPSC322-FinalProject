# need accuracy_score, binary_precision_score,binary_recall_score,binary_f1_score

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float or int): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    """
    if not y_true:  # Check if y_true is empty
        return 0.0 if normalize else 0

    correct_count = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    if normalize:
        return correct_count / len(y_true)
    else:
        return correct_count
    
    