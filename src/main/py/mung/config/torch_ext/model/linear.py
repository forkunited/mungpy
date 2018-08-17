from mung.torch_ext.model.linear import DataParameter, MultinomialLogisticRegression, LogisticRegression, \
    LinearRegression, OrdinalLogisticRegression, OrdisticRegression

# Expects config of the form:
# {
#   data_parameter : {
#     input : [INPUT PARAMETER NAME]
#     output : [OUTPUT PARAMETER NAME]
#   }
#   name : [ID FOR MODEL]
#   arch_type : [LinearRegression|LogisticRegression|MultinomialLogisticRegression|OrdinalLogisticRegression|OrdisticRegression]
#   (MultinomialLogisticRegression|OrdinalLogisticRegression|OrdisticRegression) label_count : [NUMBER OF OUTPUT CLASSES]
#   bias : [INDICATOR OF WHETHER OR NOT TO INCLUDE BIAS TERM]
# }
def load_linear_model(config, D, gpu=False):
    data_parameter = DataParameter.make(**config["data_parameter"])

    name = config["name"]
    input_size = D[data_parameter[DataParameter.INPUT]].get_feature_set().get_token_count()
    bias = bool(int(config["bias"]))

    if config["arch_type"] == "MultinomialLogisticRegression":
        label_count =  config["label_count"]
        model = MultinomialLogisticRegression(name, input_size, label_count, bias=bias)
    elif config["arch_type"] == "LogisticRegression":
        model = LogisticRegression(name, input_size, bias=bias)
    elif config["arch_type"] == "OrdinalLogisticRegression":
        label_count =  config["label_count"]
        model = OrdinalLogisticRegression(name, input_size, label_count, bias=bias)
    elif config["arch_type"] == "OrdisticRegression":
        label_count = config["label_count"]
        model = OrdisticRegression(name, input_size, label_count, bias=bias)
    else: # LinearRegression
        model = LinearRegression(name, input_size, bias=bias)
    if gpu:
        model = model.cuda()

    return data_parameter, model