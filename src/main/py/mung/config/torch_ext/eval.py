from mung.config.eval import load_metric
from mung.torch_ext.eval import ModulePrediction

# Expects config of the form:
# {
#   evaluations : [
#     name : [ID FOR EVALUATION]
#     type : [ModulePrediction]
#     data : [NAME OF DATA SUBSET]
#     (ModulePrediction) target_parameter : [NAME OF TARGET VIEW WITHIN DATA]
#     parameters : {
#       rand : [INDICATES WHETHER PREDICTIONS ARE RANDOMIZED]
#       (Optional) metrics : [CONFIG FOR METRICS]
#     }
#   ]
# }
def load_evaluations(config, data_sets, data_parameter, gpu=False):
    evaluations = []

    for eval_config in config["evaluations"]:
        data = data_sets[eval_config["data"]]
        target_parameter = eval_config["target_parameter"]
        if eval_config["type"] == "ModulePrediction":
            params_config = eval_config["parameters"]
            rand = bool(int(params_config["rand"]))
            metrics = []
            if "metrics" in params_config:
                for metric_config in params_config["metrics"]:
                    metrics.append(load_metric(metric_config))
            evaluations.append(ModulePrediction(eval_config["name"], data, data_parameter, target_parameter, rand=rand, metrics=metrics))
        else:
            raise ValueError("Invalid evaluation type in config (" + str(eval_config["type"]) + ")")

    return evaluations
