from mung.torch_ext.model.linear import DataParameter, MultinomialLogisticRegression, LogisticRegression, \
    LinearRegression, OrdinalLogisticRegression, OrdisticRegression, PairwiseOrdinalLogisticRegression

# Expects config of the form:
# {
#   data_parameter : {
#     input : [INPUT PARAMETER NAME]
#     output : [OUTPUT PARAMETER NAME]
#   }
#   name : [ID FOR MODEL]
#   arch_type : [LinearRegression|LogisticRegression|MultinomialLogisticRegression|OrdinalLogisticRegression|OrdisticRegression|PairwiseOrdinalLogisticRegression]
#   (MultinomialLogisticRegression|OrdinalLogisticRegression|OrdisticRegression|PairwiseOrdinalLogisticRegression) label_count : [NUMBER OF OUTPUT CLASSES]
#   (Optional for PairwiseOrdinalLogisticRegression) confidence_ordinals : [INDICATOR OF WHETHER TO PREDICT ORDINALS BASED ON CONFIDENCES]
#   (Optional for PairwiseOrdinalLogisticRegression) constant_scalar : [INDICATOR OF WHETHER s PARAMETER IS CONSTANT ACROSS THRESHOLDS]
#   (Optional for PairwiseOrdinalLogisticRegression) hidden_sizes : [LIST OF SIZES FOR HIDDEN LAYERS]
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
        hidden_sizes = []
        if "hidden_sizes" in config:
            hidden_sizes = config["hidden_sizes"]
        output_size = D[data_parameter[DataParameter.OUTPUT]].get_feature_set().get_token_count()
        model = LogisticRegression(name, input_size, output_size=output_size, bias=bias, hidden_sizes=hidden_sizes)
    elif config["arch_type"] == "OrdinalLogisticRegression":
        label_count =  config["label_count"]
        model = OrdinalLogisticRegression(name, input_size, label_count, bias=bias)
    elif config["arch_type"] == "OrdisticRegression":
        label_count = config["label_count"]
        model = OrdisticRegression(name, input_size, label_count, bias=bias)
    elif config["arch_type"] == "PairwiseOrdinalLogisticRegression":
        ordinal_rescaling = 1.0
        if "ordinal_rescaling" in config:
            ordinal_rescaling = float(config["ordinal_rescaling"])
        confidence_ordinals = False
        if "confidence_ordinals" in config:
            confidence_ordinals = bool(int(config["confidence_ordinals"]))
        constant_scalar = True
        if "constant_scalar" in config:
            constant_scalar = bool(int(config["constant_scalar"]))
        hidden_sizes = []
        if "hidden_sizes" in config:
            hidden_sizes = config["hidden_sizes"]

        label_count = config["label_count"]
        model = PairwiseOrdinalLogisticRegression(name, input_size, label_count, bias=bias, ordinal_rescaling=ordinal_rescaling, confidence_ordinals=confidence_ordinals, constant_scalar=constant_scalar, hidden_sizes=hidden_sizes)
    else: # LinearRegression
        model = LinearRegression(name, input_size, bias=bias)
    if gpu:
        model = model.cuda()

    return data_parameter, model