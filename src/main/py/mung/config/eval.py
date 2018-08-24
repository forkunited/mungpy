from mung.eval import ConfusionMatrix, ClassificationReport, F1, PearsonR, SpearmanR, MAE, Accuracy

# Expects config of the form:
# {
#    name : [ID FOR METRIC],
#    type :[ConfusionMatrix|ClassificationReport|F1|SpearmanR|PearsonR|MAE|Accuracy],
#    (Optional) labels : [ [LIST OF LABELS] ]
#    (Optional for PearsonR and SpearmanR) include_p : [INDICATOR OF WHETHER TO INCLUDE P-VALUE]
#    (Optional for ClassificationReport|ConfusionMatrix) target_names : [ [LIST OF LABEL NAMES ]]
#    (Optional for Accuracy) tolerance : [PREDICTIONS CONSIDERED ACCURATE UP TO THE TOLERANCE]
# }
def load_metric(config):
    evaluations = []

    metric_name = config["name"]
    labels = None
    if "labels" in config:
        labels = config["labels"]
    
    if config["type"] == "ConfusionMatrix":
        target_names = None
        if "target_names" in config:
            target_names = config["target_names"]
        return ConfusionMatrix(metric_name, labels=labels, target_names=target_names)
    elif config["type"] == "ClassificationReport":
        target_names = None
        if "target_names" in config:
            target_names = config["target_names"]
        return ClassificationReport(metric_name, labels=labels, target_names=target_names)
    elif config["type"] == "F1":
        return F1(metric_name, labels=labels)
    elif config["type"] == "SpearmanR":
        include_p = True
        if "include_p" in config:
            include_p = bool(int(config["include_p"]))
        return SpearmanR(metric_name, include_p=include_p)
    elif config["type"] == "PearsonR":
        include_p = True
        if "include_p" in config:
            include_p = bool(int(config["include_p"]))
        return PearsonR(metric_name, include_p=include_p)
    elif config["type"] == "MAE":
        return MAE(metric_name)
    elif config["type"] == "Accuracy":
        tolerance = 0.0
        if "tolerance" in config:
            tolerance = config["tolerance"]
        return Accuracy(metric_name, tolerance=tolerance)
    else:
        raise ValueError("Invalid metric type in config (" + str(config["type"]) + ")")

# Expects config of the form:
# {
#    data_metrics : [
#        data : {
#            [ID] : [TARGET DATA PARAMETER]
#        }
#        score_metrics : [LIST OF METRICS FOR MODEL SCORES]
#        prediction_metrics : [LIST OF METRICS FOR MODEL PREDICTIONS]
#    ]
# }
def load_data_metrics(config):
    data_metrics = dict()
    for data_metrics_config in config["data_metrics"]:
        score_metrics = [load_metric(metric_config) for metric_config in data_metrics_config["score_metrics"]]
        prediction_metrics = [load_metric(metric_config) for metric_config in data_metrics_config["prediction_metrics"]]
        for data_name, target_parameter in data_metrics_config["data"].iteritems():
            data_metrics[data_name] = { "score_metrics" : score_metrics, "prediction_metrics" : prediction_metrics, "target_parameter" : target_parameter }
    return data_metrics
