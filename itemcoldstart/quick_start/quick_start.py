"""
itemcoldstart.quick_start
########################
"""
import logging
from logging import getLogger
import sys
from ray import tune
from itemcoldstart.config import Config
from itemcoldstart.data import (
    create_dataset,
    data_preparation,
)
from itemcoldstart.data.transform import construct_transform
from itemcoldstart.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)


def run_itemcoldstart(
    model=None, dataset=None, config_file_list=None, config_dict=None, saved=True
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        saved_ui_emb (bool, optional): Whether to save the model. Defaults to ``False``
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    split_args = config["eval_args"]["split"]
    split_mode = list(split_args.keys())[0]
    cold_enable = split_mode.lower() == 'cold'

    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    normal_cold = config['normal_cold']
    if cold_enable:
        if normal_cold:
            train_data, valid_data, valid_hot_data, valid_cold_data, test_data, test_hot_data, test_cold_data, hot_items, cold_items, valid_cold_known_dataset, test_cold_known_dataset, valid_known_dataset, test_known_dataset = data_preparation(
                config, dataset, normal_cold=True)
        else:
            train_data, valid_data, valid_hot_data, valid_cold_data, test_data, test_hot_data, test_cold_data, hot_items, cold_items = data_preparation(
                config, dataset, normal_cold=False)
            valid_cold_known_dataset, test_cold_known_dataset, valid_known_dataset, test_known_dataset = None, None, None, None
    else:
        train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    model = get_model(config["model"])(
        config, train_data._dataset, hot_items, cold_items).to(config["device"])
    # logger.info(model)

    transform = construct_transform(config)
    if config['cal_flop']:
        flops = get_flops(model, dataset, config["device"], logger, transform)
        logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    saved_ui_emb = config["saved_ui_emb"]

    saved_entity_emb = config["saved_entity_emb"]
    if cold_enable:
        best_valid_score, best_valid_result, training_time, avg_training_time, epochs = trainer.fit(
            train_data, (valid_data, valid_hot_data, valid_cold_data), saved=saved, saved_ui_emb=saved_ui_emb, saved_entity_emb=saved_entity_emb, show_progress=config[
                "show_progress"], cold_items=cold_items, hot_items=hot_items, eval_handc=config["eval_handc"],ui_graph_known=valid_known_dataset, ui_graph_cold_known=valid_cold_known_dataset
        )

        # model evaluation
        test_result = trainer.evaluate(
            test_data, load_best_model=saved, show_progress=config["show_progress"], extra_mask_items=None, cold_items=cold_items, ui_graph_known=test_known_dataset
        )
        test_hot_result = trainer.evaluate(
            test_hot_data, load_best_model=saved, show_progress=config[
                "show_progress"], extra_mask_items=cold_items, cold_items=cold_items, 
            ui_graph_known=None
        )
        test_cold_result = trainer.evaluate(
            test_cold_data, load_best_model=saved, show_progress=config[
                "show_progress"], extra_mask_items=hot_items, cold_items=cold_items,
            ui_graph_known=test_cold_known_dataset
        )

    else:
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config["show_progress"]
        )

        # model evaluation
        test_result = trainer.evaluate(
            test_data, load_best_model=saved, show_progress=config["show_progress"], extra_mask_items=None, cold_items=cold_items
        )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    if cold_enable:
        logger.info(set_color("test result (hot)", "red") +
                    f": {test_hot_result}")
        logger.info(set_color("test result (cold)", "blue") +
                    f": {test_cold_result}")
    
    if config['output_time']:
        with open(config['time_output_path'], 'a', encoding='utf-8') as fout:
            fout.write(config['model'] + ' ' + str(training_time) + ' ' + str(avg_training_time) + ' ' + str(epochs) +'\n')
        
    return {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def run_itemcoldstarts(rank, *args):
    ip, port, world_size, nproc, offset = args[3:]
    args = args[:3]
    run_itemcoldstart(
        *args,
        config_dict={
            "local_rank": rank,
            "world_size": world_size,
            "ip": ip,
            "port": port,
            "nproc": nproc,
            "offset": offset,
        },
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(
        config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(
        config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data


def load_data_and_model_cold(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, valid_hot_data, valid_cold_data, test_data, test_hot_data, test_cold_data, hot_items, cold_items = data_preparation(
        config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(
        config, train_data._dataset, hot_items, cold_items).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, valid_hot_data, valid_cold_data, test_data, test_hot_data, test_cold_data, hot_items, cold_items
