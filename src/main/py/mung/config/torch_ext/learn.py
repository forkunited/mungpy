from mung.torch_ext.learn import Trainer

# Expects config of the form:
# {
#   max_evaluation : [INICATES WHETHER TO MAXIMIZE OVER EVALUATIONS (OR MINIMIZE)]
#   data : [NAME OF DATA SUBSET TO TRAIN ON]
#   (Optional) data_size : [SIZE OF DATA SUBSET TO TRAIN ON]
#   iterations : [TRAINING ITERATIONS]
#   batch_size : [BATCH SIZE]
#   optimizer_type : [OPTIMIZER TYPE]
#   learning_rate : [LEARNING RATE]
#   weight_decay : [WEIGHT DECAY]
#   gradient_clipping : [GRADIENT CLIPPING]
#   log_interval : [LOG INTERVAL]
#   (Optional) curriculum : {
#       steps : [NUMBER OF CURRICULUM STEPS TO SPLIT DATA INTO]
#       final_iterations : [NUMBER OF ITERATIONS AT FINAL STEP OF CURRICULUM]
#   }
# }
def train_from_config(config, data_parameter, loss_criterion, logger, evaluations, model, data_sets, best_part_fn=None, curriculum_key_fn_constructor=None):
    data = data_sets[config["data"]]
    if "data_size" in config:
        data.shuffle()
        data = data.get_subset(0, int(config["data_size"]))

    trainer = Trainer(data_parameter, loss_criterion, logger, \
            evaluations[0], other_evaluations=evaluations[1:len(evaluations)], \
            max_evaluation=bool(int(config["max_evaluation"])))
    
    iterations = int(config["iterations"])
    batch_size = int(config["batch_size"])
    optimizer_type = config["optimizer_type"]
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    gradient_clipping = float(config["gradient_clipping"])
    log_interval = int(config["log_interval"])

    best_part = None
    best_iteration = None
    if "curriculum" not in config or curriculum_key_fn_constructor is None:
        model, best_part, best_iteration = trainer.train(model, data, iterations, \
            batch_size=batch_size, optimizer_type=optimizer_type, lr=learning_rate, weight_decay=weight_decay, \
            grad_clip=gradient_clipping, log_interval=log_interval, best_part_fn=best_part_fn)
    else:
        curriculum_key_fn = curriculum_key_fn_constructor(data)
        data.sort(curriculum_key_fn)
        data_size = data.get_size()
        steps = int(config["curriculum"]["steps"])
        for i in range(steps):
            if i == steps-1:
                iterations = config["curriculum"]["final_iterations"]

            step_data = data.get_subset(0, (i+1)*data_size/steps)
            step_data.shuffle()
            model, best_part, best_iteration = trainer.train(model, step_data, iterations, \
                batch_size=batch_size, optimizer_type=optimizer_type, lr=learning_rate, weight_decay=weight_decay, \
                grad_clip=gradient_clipping, log_interval=log_interval, best_part_fn=best_part_fn)

    return model, best_part, best_iteration



def cotrain_from_config(config, data_parameters, loss_criteria, loggers, evaluations, models, data_sets, best_part_fns=None, curriculum_key_fn_constructor=None):
    data = data_sets[config["data"]]
    if "data_size" in config:
        data.shuffle()
        data = data.get_subset(0, int(config["data_size"]))

    trainers = []
    for i in range(len(models)):
        trainers.append(Trainer(data_parameters[i], loss_criteria[i], loggers[i], \
            evaluations[i][0], other_evaluations=evaluations[i][1:len(evaluations)], \
            max_evaluation=bool(int(config["max_evaluation"][i]))))
        
    iterations = int(config["iterations"])
    batch_size = int(config["batch_size"])
    optimizer_type = config["optimizer_type"]
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    gradient_clipping = float(config["gradient_clipping"])
    log_interval = int(config["log_interval"])
    swap_interval = int(config["swap_interval"])

    best_part = None
    best_iteration = None
    if "curriculum" not in config or curriculum_key_fn_constructor is None:
        current_iter = 0
        while current_iter < iterations:
            for i in range(len(models)):
                best_part_fn = None
                if best_part_fns is not None:
                    best_part_fn = best_part_fns[i]
                model, best_part, best_iteration = trainers[i].train(models[i], data, swap_interval, \
                    batch_size=batch_size, optimizer_type=optimizer_type, lr=learning_rate, weight_decay=weight_decay, \
                    grad_clip=gradient_clipping, log_interval=log_interval, best_part_fn=best_part_fn)
            current_iter += swap_interval
    else:
        curriculum_key_fn = curriculum_key_fn_constructor(data)
        data.sort(curriculum_key_fn)
        data_size = data.get_size()
        steps = int(config["curriculum"]["steps"])
        for i in range(steps):
            if i == steps-1:
                iterations = config["curriculum"]["final_iterations"]
            step_data = data.get_subset(0, (i+1)*data_size/steps)
            step_data.shuffle()
            current_iter = 0
            while current_iter < iterations:
                for i in range(len(models)):
                    best_part_fn = None
                    if best_part_fns is not None:
                        best_part_fn = best_part_fns[i]
                    model, best_part, best_iteration = trainers[i].train(models[i], step_data, swap_interval, \
                        batch_size=batch_size, optimizer_type=optimizer_type, lr=learning_rate, weight_decay=weight_decay, \
                        grad_clip=gradient_clipping, log_interval=log_interval, best_part_fn=best_part_fn)
                current_iter += swap_interval

    return model, best_part, best_iteration
