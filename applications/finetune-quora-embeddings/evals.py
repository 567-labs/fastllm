from sklearn.metrics import accuracy_score, precision_score, recall_score
import itertools


def threshold_decorator(threshold=0.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            labels, predictions = args
            coerced_predictions = [
                1 if item >= threshold else 0 for item in predictions
            ]
            return func(*[labels, coerced_predictions], **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    # Small Unit Test for sanity check
    predictions = [0.2, 0.3, 0.9, 0.5]
    labels = [0, 0, 1, 1]

    thresholds = [0.7]

    # Define the metrics
    metrics = {
        "accuracy": accuracy_score,  # This is the number of correct predictions by the model ( TP + TN )/ (# of samples)
        "precision": precision_score,  # This measures the number of positive class predicitons (TP) / (TP + FP)
        "recall": recall_score,  # This measure the number of negative class predictions (TP) / ( TP + FN )
    }

    evals = {}

    for threshold, metric_name in itertools.product(thresholds, metrics.keys()):
        evals[f"{metric_name}({threshold})"] = threshold_decorator(threshold)(
            metrics[metric_name]
        )

    # Predictions would be [0,0,0,1] which translates to TP : 1, TN : 2 , FP : 0, FN: 1
    for name, f in evals.items():
        score = f(labels, predictions)
        print(f"{name} - {score}")
        if name == "accuracy(0.7)":
            assert score == 0.75

        if name == "precison(0.7)":
            assert score == 1

        if name == "recall(0.7)":
            assert score == 0.5
