from mung.torch_ext.eval import DataParameter, ModulePrediction
from mung.eval import ConfusionMatrix, ClassificationReport, F1

# Expects config of the form:
# {
#   evaluations : [
#     name : [ID FOR EVALUATION]
#     type : [ModulePrediction]
#     data : [NAME OF DATA SUBSET]
#     parameters : {
#       rand : [INDICATES WHETHER PREDICTIONS ARE RANDOMIZED]
#       (Optional) metrics : [
#         name : [ID FOR METRIC],
#         type :[ConfusionMatrix|ClassificationReport|F1],
#         (Optional) labels : [ [LIST OF LABELS] ]
#         (Optional for ClassificationReport) target_names : [ [LIST OF LABEL NAMES ]]
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
            metrics = []
            if "metrics" in params_config:
                for metric_config in params_config["metrics"]:
                    metric_name = metric_config["name"]
                    labels = None
                    if "labels" in metric_config:
                        labels = metric_config["labels"]
                    if metric_config["type"] == "ConfusionMatrix":
                        metrics.append(ConfusionMatrix(metric_name, labels=labels))
                    elif metric_config["type"] == "ClassificationReport":
                        target_names = None
                        if "target_names" in metric_config:
                            target_names = metric_config["target_names"]
                        metrics.append(ClassificationReport(metric_name, labels=labels, target_names=target_names))
                    elif metric_config["type"] == "F1":
                        metrics.append(F1(metric_name, labels=labels))
                    else:
                        raise ValueError("Invalid metric type in config (" + str(metric_config["type"]) + ")")
            evaluations.append(ModulePrediction(eval_config["name"], data, data_parameter, rand=rand, metrics=metrics))
        else:
            raise ValueError("Invalid evaluation type in config (" + str(eval_config["type"]) + ")")

    return evaluations
