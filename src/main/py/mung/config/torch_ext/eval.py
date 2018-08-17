from mung.torch_ext.eval import DataParameter, ModulePrediction
from mung.eval import ConfusionMatrix, ClassificationReport, F1, PearsonR, SpearmanR, MAE

# Expects config of the form:
# {
#   evaluations : [
#     name : [ID FOR EVALUATION]
#     type : [ModulePrediction]
#     data : [NAME OF DATA SUBSET]
#     parameters : {
#       rand : [INDICATES WHETHER PREDICTIONS ARE RANDOMIZED]
#       (Optional) include_data : [INDICATES WHETHER OR NOT TO INCLUDE THE DATA SET IN OUTPUT]
#       (Optional) include_score : [INDICATES WHETHER OR NOT TO INCLUDE THE MODEL SCORES IN OUTPUT]
#       (Optional) metrics : [
#         name : [ID FOR METRIC],
#         type :[ConfusionMatrix|ClassificationReport|F1|SpearmanR|PearsonR|MAE],
#         (Optional) labels : [ [LIST OF LABELS] ]
#         (Optional for PearsonR and SpearmanR) include_p : [INDICATOR OF WHETHER TO INCLUDE P-VALUE]
#         (Optional for ClassificationReport|ConfusionMatrix) target_names : [ [LIST OF LABEL NAMES ]]
#       ]
#     }
#   ]
# }
def load_evaluations(config, data_sets, data_parameter, gpu=False):
    evaluations = []

    for eval_config in config["evaluations"]:
        data = data_sets[eval_config["data"]]
        if eval_config["type"] == "ModulePrediction":
            params_config = eval_config["parameters"]
            rand = bool(int(params_config["rand"]))
            include_data = False
            if "include_data" in params_config:
                include_data = bool(int(params_config["include_data"]))
            include_score= False
            if "include_score" in params_config:
                include_score = bool(int(params_config["include_score"]))
            metrics = []
            if "metrics" in params_config:
                for metric_config in params_config["metrics"]:
                    metric_name = metric_config["name"]
                    labels = None
                    if "labels" in metric_config:
                        labels = metric_config["labels"]
                    if metric_config["type"] == "ConfusionMatrix":
                        target_names = None
                        if "target_names" in metric_config:
                            target_names = metric_config["target_names"]
                        metrics.append(ConfusionMatrix(metric_name, labels=labels, target_names=target_names))
                    elif metric_config["type"] == "ClassificationReport":
                        target_names = None
                        if "target_names" in metric_config:
                            target_names = metric_config["target_names"]
                        metrics.append(ClassificationReport(metric_name, labels=labels, target_names=target_names))
                    elif metric_config["type"] == "F1":
                        metrics.append(F1(metric_name, labels=labels))
                    elif metric_config["type"] == "SpearmanR":
                        include_p = True
                        if "include_p" in metric_config:
                            include_p = bool(int(metric_config["include_p"]))
                        metrics.append(SpearmanR(metric_name, include_p=include_p))
                    elif metric_config["type"] == "PearsonR":
                        include_p = True
                        if "include_p" in metric_config:
                            include_p = bool(int(metric_config["include_p"]))
                        metrics.append(PearsonR(metric_name, include_p=include_p))
                    elif metric_config["type"] == "MAE":
                        metrics.append(MAE(metric_name))
                    else:
                        raise ValueError("Invalid metric type in config (" + str(metric_config["type"]) + ")")
            evaluations.append(ModulePrediction(eval_config["name"], data, data_parameter, rand=rand, metrics=metrics, include_data=include_data, include_score=include_score))
        else:
            raise ValueError("Invalid evaluation type in config (" + str(eval_config["type"]) + ")")

    return evaluations
