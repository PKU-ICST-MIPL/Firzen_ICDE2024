"""
itemcoldstart.data.utils
########################
"""

import importlib
import os
import pickle
import warnings
from typing import Literal

from itemcoldstart.data.dataloader import *
from itemcoldstart.sampler import KGSampler, Sampler
from itemcoldstart.utils import ModelType, ensure_dir, get_local_time, set_color
from itemcoldstart.utils.argument_list import dataset_arguments

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module("itemcoldstart.data.dataset")
    if hasattr(dataset_module, config["model"] + "Dataset"):
        dataset_class = getattr(dataset_module, config["model"] + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(
        config["checkpoint_dir"], f'{config["dataset"]}-{dataset_class.__name__}.pth'
    )
    file = config["dataset_save_path"] or default_file
    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ["seed", "repeatable"]:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from",
                        "pink") + f": [{file}]")
            return dataset

    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config["checkpoint_dir"])
    save_path = config["checkpoint_dir"]
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color("Saving split dataloaders into",
                "pink") + f": [{file_path}]")
    Serialization_dataloaders = []
    for dataloader in dataloaders:
        generator_state = dataloader.generator.get_state()
        dataloader.generator = None
        dataloader.sampler.generator = None
        Serialization_dataloaders += [(dataloader, generator_state)]

    with open(file_path, "wb") as f:
        pickle.dump(Serialization_dataloaders, f)


def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """
    split_args = config["eval_args"]["split"]
    split_mode = list(split_args.keys())[0]

    default_file = os.path.join(
        config["checkpoint_dir"],
        f'{config["dataset"]}-for-{config["model"]}-dataloader.pth',
    )
    dataloaders_save_path = config["dataloaders_save_path"] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, "rb") as f:
        dataloaders = []
        for data_loader, generator_state in pickle.load(f):
            generator = torch.Generator()
            generator.set_state(generator_state)
            data_loader.generator = generator
            data_loader.sampler.generator = generator
            dataloaders.append(data_loader)

        if split_mode == 'COLD':
            train_data, valid_data, valid_hot_data, valid_cold_data, test_data, test_hot_data, test_cold_data = dataloaders
        else:
            train_data, valid_data, test_data = dataloaders

    for arg in dataset_arguments + ["seed", "repeatable", "eval_args"]:
        if config[arg] != train_data.config[arg]:
            return None

    train_data.update_config(config)

    if split_mode == 'cold':
        valid_data.update_config(config)
        valid_hot_data.update_config(config)
        valid_cold_data.update_config(config)
        test_data.update_config(config)
        test_hot_data.update_config(config)
        test_cold_data.update_config(config)

    else:
        valid_data.update_config(config)
        test_data.update_config(config)

    logger = getLogger()
    logger.info(
        set_color("Load split dataloaders from", "pink")
        + f": [{dataloaders_save_path}]"
    )

    if split_mode == 'COLD':
        return train_data, valid_data, valid_hot_data, valid_cold_data, test_data, test_hot_data, test_cold_data

    return train_data, valid_data, test_data


def data_preparation(config, dataset, normal_cold=False):
    """Split the dataset by :attr:`config['[valid|test]_eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    split_args = config["eval_args"]["split"]
    split_mode = list(split_args.keys())[0]

    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
        dataset._change_feat_format()
    else:
        model_type = config["MODEL_TYPE"]
        built_datasets = dataset.build(normal_cold=normal_cold)

        if split_mode == 'COLD':
            # cold datasets
            if normal_cold:
                (train_dataset, valid_hot_dataset, test_hot_dataset, valid_cold_known_dataset, valid_cold_dataset,
             test_cold_known_dataset, test_cold_dataset, valid_known_dataset, valid_dataset, test_known_dataset, test_dataset), hot_items, cold_items = built_datasets

            else:
                (train_dataset, valid_hot_dataset, test_hot_dataset, valid_cold_dataset,
             test_cold_dataset, valid_dataset, test_dataset), hot_items, cold_items = built_datasets


            train_sampler, valid_sampler, test_sampler = create_samplers(
                config, dataset, [train_dataset, valid_dataset, test_dataset]
            )
            _, valid_hot_sampler, test_hot_sampler = create_samplers(
                config, dataset, [train_dataset,
                                  valid_hot_dataset, test_hot_dataset]
            )
            _, valid_cold_sampler, test_cold_sampler = create_samplers(
                config, dataset, [train_dataset,
                                  valid_cold_dataset, test_cold_dataset]
            )


            if model_type != ModelType.KNOWLEDGE:
                train_data = get_dataloader(config, "train")(
                    config, train_dataset, train_sampler, shuffle=config["shuffle"]
                )
            else:
                kg_sampler = KGSampler(
                    dataset,
                    config["train_neg_sample_args"]["distribution"],
                    config["train_neg_sample_args"]["alpha"],
                )
                train_data = get_dataloader(config, "train")(
                    config, train_dataset, train_sampler, kg_sampler, shuffle=True
                )

            valid_data = get_dataloader(config, "valid")(
                config, valid_dataset, valid_sampler, shuffle=False
            )
            valid_hot_data = get_dataloader(config, "valid")(
                config, valid_hot_dataset, valid_sampler, shuffle=False
            )
            valid_cold_data = get_dataloader(config, "valid")(
                config, valid_cold_dataset, valid_sampler, shuffle=False
            )
            test_data = get_dataloader(config, "test")(
                config, test_dataset, test_sampler, shuffle=False
            )
            test_hot_data = get_dataloader(config, "test")(
                config, test_hot_dataset, test_hot_sampler, shuffle=False
            )
            test_cold_data = get_dataloader(config, "test")(
                config, test_cold_dataset, test_cold_sampler, shuffle=False
            )

            dataloaders_all = (train_data, valid_data, valid_hot_data,
                               valid_cold_data, test_data, test_hot_data, test_cold_data)

        else:
            train_dataset, valid_dataset, test_dataset = built_datasets
            train_sampler, valid_sampler, test_sampler = create_samplers(
                config, dataset, built_datasets
            )

            valid_data = get_dataloader(config, "valid")(
                config, valid_dataset, valid_sampler, shuffle=False
            )
            test_data = get_dataloader(config, "test")(
                config, test_dataset, test_sampler, shuffle=False
            )

            if model_type != ModelType.KNOWLEDGE:
                train_data = get_dataloader(config, "train")(
                    config, train_dataset, train_sampler, shuffle=config["shuffle"]
                )
            else:
                kg_sampler = KGSampler(
                    dataset,
                    config["train_neg_sample_args"]["distribution"],
                    config["train_neg_sample_args"]["alpha"],
                )
                train_data = get_dataloader(config, "train")(
                    config, train_dataset, train_sampler, kg_sampler, shuffle=True
                )

            dataloaders_all = (train_data, valid_data, test_data)

        if config["save_dataloaders"]:

            save_split_dataloaders(
                # (train_data, valid_data, test_data)
                config, dataloaders=dataloaders_all
            )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )

    if split_mode == 'COLD':
        if normal_cold:
             return train_data, valid_data, valid_hot_data, valid_cold_data, test_data, test_hot_data, test_cold_data, hot_items, cold_items, valid_cold_known_dataset, test_cold_known_dataset, valid_known_dataset, test_known_dataset           
        else:
            return train_data, valid_data, valid_hot_data, valid_cold_data, test_data, test_hot_data, test_cold_data, hot_items, cold_items
    return train_data, valid_data, test_data


def get_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.
    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )


    model_type = config["MODEL_TYPE"]
    if phase == "train":
        if model_type == ModelType.KNOWLEDGE:
            return KnowledgeBasedDataLoader
        else:
            return TrainDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader

def _create_sampler(
    dataset,
    built_datasets,
    distribution: str,
    repeatable: bool,
    alpha: float = 1.0,
    base_sampler=None,
    sample_by="user"
):
    phases = ["train", "valid", "test"]
    sampler = None
    if distribution != "none":
        if base_sampler is not None:
            base_sampler.set_distribution(distribution)
            return base_sampler
        if not repeatable:
            sampler = Sampler(
                phases,
                built_datasets,
                distribution,
                alpha,
                sample_by
            )

    return sampler


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    train_neg_sample_args = config["train_neg_sample_args"]
    valid_neg_sample_args = config["valid_neg_sample_args"]
    test_neg_sample_args = config["test_neg_sample_args"]
    repeatable = config["repeatable"]
    base_sampler = _create_sampler(
        dataset,
        built_datasets,
        train_neg_sample_args["distribution"],
        repeatable,
        train_neg_sample_args["alpha"],
        sample_by=train_neg_sample_args["sample_by"]
    )
    train_sampler = base_sampler.set_phase("train") if base_sampler else None

    valid_sampler = _create_sampler(
        dataset,
        built_datasets,
        valid_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    valid_sampler = valid_sampler.set_phase("valid") if valid_sampler else None

    test_sampler = _create_sampler(
        dataset,
        built_datasets,
        test_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    test_sampler = test_sampler.set_phase("test") if test_sampler else None
    return train_sampler, valid_sampler, test_sampler
