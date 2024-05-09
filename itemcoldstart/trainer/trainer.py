r"""
itemcoldstart.trainer.trainer
################################
"""

import os

from logging import getLogger
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import torch.cuda.amp as amp

from itemcoldstart.data.interaction import Interaction
from itemcoldstart.data.dataloader import FullSortEvalDataLoader
from itemcoldstart.evaluator import Evaluator, Collector
from itemcoldstart.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    KGDataLoaderState,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)
from torch.nn.parallel import DistributedDataParallel

import random
random.seed(2023)


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        if not config["single_spec"]:
            # change from BN to SyncBN, which should after DDP init, but before DDP model construction
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.distributed_model = DistributedDataParallel(
                self.model, device_ids=[config["local_rank"]]
            )

    def fit(self, train_data):
        r"""Train the model based on the train data."""
        raise NotImplementedError("Method [next] should be implemented.")

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data."""

        raise NotImplementedError("Method [next] should be implemented.")

    def set_reduce_hook(self):
        r"""Call the forward function of 'distributed_model' to apply grads
        reduce hook to each parameter of its module.

        """
        t = self.model.forward
        self.model.forward = lambda x: x
        self.distributed_model(torch.LongTensor([0]).to(self.device))
        self.model.forward = t

    def sync_grad_loss(self):
        r"""Ensure that each parameter appears to the loss function to
        make the grads reduce sync in each node.

        """
        sync_loss = 0
        for params in self.model.parameters():
            sync_loss += torch.sum(params) * 0
        return sync_loss


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)
        split_args = config["eval_args"]["split"]
        split_mode = list(split_args.keys())[0]
        self.cold_enable = split_mode.lower() == 'cold'

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.test_batch_size = config["eval_batch_size"]
        self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
        self.device = config["device"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.enable_amp = config["enable_amp"]
        self.enable_scaler = torch.cuda.is_available(
        ) and config["enable_scaler"]
        ensure_dir(self.checkpoint_dir)
        saved_model_file = "{}-{}.pth".format(
            self.config["model"], get_local_time())
        self.saved_model_file = os.path.join(
            self.checkpoint_dir, saved_model_file)
        # self.checkpoint_dir_uiemb = os.path.join(self.checkpoint_dir, self.config["model"])
        # ensure_dir(self.checkpoint_dir_uiemb)
        self.data_dir = config["data_path"]
        saved_useremb_file = "{}-{}-{}.useremb".format(
            self.config["dataset"], self.config["model"], self.config["exp_comment"])
        self.saved_useremb_file = os.path.join(
            self.data_dir, saved_useremb_file)
        saved_itememb_file = "{}-{}-{}.itememb".format(
            self.config["dataset"], self.config["model"], self.config["exp_comment"])
        self.saved_itememb_file = os.path.join(
            self.data_dir, saved_itememb_file)
        saved_entityemb_file = "{}-{}.ent".format(
            self.config["dataset"], self.config["model"])
        self.saved_entityemb_file = os.path.join(
            self.data_dir, saved_entityemb_file)

        self.weight_decay = config["weight_decay"]

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.eval_type = config["eval_type"]
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_tensor = None
        self.tot_item_num = None

        # used to record the training time
        self.training_time = 0.0
        
    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        # pop is used to delete and return the value of a given key. If the key does not exist, it returns 'default'
        params = kwargs.pop("params", self.model.parameters())
        learner = kwargs.pop("learner", self.learner)
        learning_rate = kwargs.pop("learning_rate", self.learning_rate)
        weight_decay = kwargs.pop("weight_decay", self.weight_decay)

        if (
            self.config["reg_weight"]
            and weight_decay
            and weight_decay * self.config["reg_weight"] > 0
        ):
            self.logger.warning(
                "The parameters [weight_decay] and [reg_weight] are specified simultaneously, "
                "which may lead to double regularization."
            )

        if learner.lower() == "adam":
            optimizer = optim.Adam(
                params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate,
                                  weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "sparse_adam":
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning(
                    "Sparse Adam cannot argument received argument [{weight_decay}]"
                )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        # In Python, the or operator returns the first operand if it is evaluated to True, otherwise it returns the second operand.
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = amp.GradScaler(enabled=self.enable_scaler)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " +
                              get_gpu_usage(self.device), "yellow")
                )

        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False, extra_mask_items=None, cold_items=None, ui_graph_known=None):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """

        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress, extra_mask_items=extra_mask_items, cold_items=cold_items,
            ui_graph_known=ui_graph_known
        )
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_ui_embedding(self, verbose=True, **kwargs):
        if not self.config["single_spec"] and self.config["local_rank"] != 0:
            return
        saved_useremb_file = kwargs.pop(
            "saved_useremb_file", self.saved_useremb_file)
        saved_itememb_file = kwargs.pop(
            "saved_itememb_file", self.saved_itememb_file)
        uid_dict = self.model.dataset.field2id_token[self.model.dataset.uid_field]
        iid_dict = self.model.dataset.field2id_token[self.model.dataset.iid_field]

        with open(saved_useremb_file, "w", encoding="utf-8") as fout:
            fout.write("uid:token" + "\t" + "user_emb:float_seq" + "\n")
            for i in range(1, self.model.user_embedding.weight.size(0)):
                # do not save [PAD]
                line = "{}".format(uid_dict[i]) + "\t" + " ".join(
                    str(x) for x in self.model.restore_user_e[i].tolist())
                if i != self.model.user_embedding.weight.size(0) - 1:
                    line += "\n"
                fout.write(line)
        with open(saved_itememb_file, "w", encoding="utf-8") as fout:
            fout.write("iid:token" + "\t" + "item_emb:float_seq" + "\n")
            if hasattr(self.model, "item_id_embedding"):
                for i in range(1, self.model.item_id_embedding.weight.size(0)):
                    line = "{}".format(iid_dict[i]) + "\t" + " ".join(
                        str(x) for x in self.model.restore_item_e[i].tolist())
                    if i != self.model.item_id_embedding.weight.size(0) - 1:
                        line += "\n"
                    fout.write(line)
            elif hasattr(self.model, "entity_embedding"):
                for i in range(1, self.model.n_items + 1):
                    line = "{}".format(iid_dict[i]) + "\t" + " ".join(
                        str(x) for x in self.model.restore_entity_e[i].tolist())
                    if i != self.model.n_items:
                        line += "\n"
                    fout.write(line)
            else:
                raise NotImplementedError
            if verbose:
                self.logger.info(
                    set_color("Saving useremb", "blue") +
                    f": {saved_useremb_file}"
                )
                self.logger.info(
                    set_color("Saving itememb", "blue") +
                    f": {saved_itememb_file}"
                )

    def _save_entity_embedding(self, verbose=True, **kwargs):
        """
        save pretrained entity embedding in KG
        """
        if not self.config["single_spec"] and self.config["local_rank"] != 0:
            return
        saved_entityemb_file = kwargs.pop(
            "saved_entityemb_file", self.saved_entityemb_file)

        eid_dict = self.model.dataset.field2id_token[self.model.dataset.config["ENTITY_ID_FIELD"]]
        with open(saved_entityemb_file, "w", encoding="utf-8") as fout:
            fout.write("ent_id:token" + "\t" + "ent_emb:float_seq" + "\n")
            for i in range(1, self.model.entity_embedding.weight.size(0)):
                # do not save [PAD]
                line = "{}".format(eid_dict[i]) + "\t" + " ".join(
                    str(x) for x in self.model.entity_embedding.weight.data[i].tolist())
                if i != self.model.entity_embedding.weight.size(0) - 1:
                    line += "\n"
                fout.write(line)

            if verbose:
                self.logger.info(
                    set_color("Saving entityemb", "blue") +
                    f": {saved_entityemb_file}"
                )

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        if not self.config["single_spec"] and self.config["local_rank"] != 0:
            return
        saved_model_file = kwargs.pop(
            "saved_model_file", self.saved_model_file)
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(
                set_color("Saving current", "blue") + f": {saved_model_file}"
            )

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.cur_step = checkpoint["cur_step"]
        self.best_valid_score = checkpoint["best_valid_score"]

        # load architecture params from checkpoint
        if checkpoint["config"]["model"].lower() != self.config["model"].lower():
            self.logger.warning(
                "Architecture configuration given in config file is different from that of checkpoint. "
                "This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        message_output = "Checkpoint loaded. Resume training from epoch {}".format(
            self.start_epoch
        )
        self.logger.info(message_output)

    def resume_model_only(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file, map_location=self.device)

        # load architecture params from checkpoint
        if checkpoint["config"]["model"].lower() != self.config["model"].lower():
            self.logger.warning(
                "Architecture configuration given in config file is different from that of checkpoint. "
                "This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.load_other_parameter(checkpoint.get("other_parameter"))

        message_output = "Checkpoint loaded. Resume training from epoch {}".format(
            self.start_epoch
        )
        self.model.user_embedding.weight.data = self.model.restore_user_e.clone()
        self.model.item_embedding.weight.data = self.model.restore_item_e.clone()
        self.model.restore_user_e = None
        self.model.restore_item_e = None
        self.model.user_embedding.weight.requires_grad_(True)
        self.model.item_embedding.weight.requires_grad_(True)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config["loss_decimal_place"] or 4
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
            + set_color("total_time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time, self.training_time)
        if isinstance(losses, tuple):
            des = set_color("train_loss%d", "blue") + ": %." + str(des) + "f"
            train_loss_output += ", ".join(
                des % (idx + 1, loss) for idx, loss in enumerate(losses)
            )
        else:
            des = "%." + str(des) + "f"
            train_loss_output += set_color("train loss",
                                           "blue") + ": " + des % losses
        return train_loss_output + "]"

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag="Loss/Train"):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            "learner": self.config["learner"],
            "learning_rate": self.config["learning_rate"],
            "train_batch_size": self.config["train_batch_size"],
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values()
            for parameter in parameters
        }.union({"model", "dataset", "config_files", "device"})
        # other model-specific hparam
        hparam_dict.update(
            {
                para: val
                for para, val in self.config.final_config_dict.items()
                if para not in unrecorded_parameter
            }
        )
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(
                hparam_dict[k], (bool, str, float, int)
            ):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(
            hparam_dict, {"hparam/best_valid_result": best_valid_result}
        )

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        saved_ui_emb=False,
        saved_entity_emb=False,
        show_progress=False,
        callback_fn=None,
        hot_items=None,
        cold_items=None,
        eval_handc=False,
        ui_graph_known=None,
        ui_graph_cold_known=None
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            saved_ui_emb (bool, optional): whether to save the user and item embedding, default: False
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """

        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)
        if saved_ui_emb and self.start_epoch >= self.epochs:
            self._save_ui_embedding(verbose=verbose)
        if saved_entity_emb and self.start_epoch >= self.epochs:
            self._save_entity_embedding(verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            # 
            # for DRAGON model
            if hasattr(self.model, 'pre_epoch_processing'):
                self.model.pre_epoch_processing()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(
                    train_loss, tuple) else train_loss
            )
            training_end_time = time()
            self.training_time += (training_end_time - training_start_time)
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss,
                    "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if epoch_idx == self.epochs - 1:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    if saved_ui_emb:
                        self._save_ui_embedding(verbose=verbose)
                    if saved_entity_emb:
                        self._save_entity_embedding(verbose=verbose)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                if not self.cold_enable:
                    valid_data = [valid_data]
                else:
                    valid_data = list(valid_data)
                # all -> hot -> cold
                valid_scores = []
                valid_results = []

                for i, v_data in enumerate(valid_data):
                    # all set
                    if i == 0:
                        extra_mask_items = None
                        ui_graph = ui_graph_known
                    # hot set
                    elif i == 1:
                        extra_mask_items = cold_items
                        ui_graph = None
                    # cold set
                    elif i == 2:
                        extra_mask_items = hot_items
                        ui_graph = ui_graph_cold_known
                    valid_score, valid_result = self._valid_epoch(
                        v_data, show_progress=show_progress,
                        extra_mask_items=extra_mask_items,
                        cold_items=cold_items,
                        ui_graph_known=ui_graph
                    )

                    valid_scores.append(valid_score)
                    valid_results.append(valid_result)

                    if not eval_handc:
                        break

                # use the result of "all" to judge early stopping
                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_scores[0],
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "yellow")
                    + ": %.2fs, "
                    + set_color("valid_score", "yellow")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_scores[0])
                if self.cold_enable and eval_handc:
                    valid_score_output.strip("]") + (set_color("valid_score (hot)", "red")
                                                     + ": %f, " +
                                                     set_color(
                                                         "valid_score (cold)", "blue")
                                                     + ": %f]") % (valid_scores[1], valid_scores[2])
                valid_result_output = (
                    set_color("valid result", "yellow") +
                    ": \n" + dict2str(valid_results[0])
                )
                if self.cold_enable and eval_handc:
                    valid_result_output += (
                        "\n" + set_color("valid result (hot)", "red") +
                        ": \n" + dict2str(valid_results[1])
                    )
                    valid_result_output += (
                        "\n" + set_color("valid result (cold)", "blue") +
                        ": \n" + dict2str(valid_results[2])
                    )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar(
                    "Valid_score", valid_scores[0], epoch_idx)

                self.wandblogger.log_metrics(
                    {**valid_results[0], "valid_step": valid_step}, head="valid"
                )
                if self.cold_enable and eval_handc:
                    self.tensorboard.add_scalar(
                        "Valid_score (hot)", valid_scores[1], epoch_idx)
                    self.tensorboard.add_scalar(
                        "Valid_score (cold)", valid_scores[2], epoch_idx)

                    self.wandblogger.log_metrics(
                        {**valid_results[1], "valid_step": valid_step}, head="valid (hot)"
                    )
                    self.wandblogger.log_metrics(
                        {**valid_results[2], "valid_step": valid_step}, head="valid (cold)"
                    )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    if saved_ui_emb:
                        self._save_ui_embedding(verbose=verbose)
                    self.best_valid_result = valid_results[0]

                if callback_fn:
                    callback_fn(epoch_idx, valid_scores[0])

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)

        return self.best_valid_score, self.best_valid_result, self.training_time, self.training_time / (epoch_idx + 1), epoch_idx + 1

    def _full_sort_batch_eval(self, batched_data, extra_mask_items, cold_items=None, ui_graph_known=None, calc_time=False):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            if calc_time:
                
                if cold_items is not None:
                    if ui_graph_known is None:
                        scores, time = self.model.full_sort_predict(
                            interaction.to(self.device), cold_items=cold_items, calc_time=calc_time)
                    else:
                        scores, time = self.model.full_sort_predict(
                            interaction.to(self.device), cold_items=cold_items, ui_graph_known=ui_graph_known, calc_time=calc_time)
                else:
                    scores, time = self.model.full_sort_predict(
                        interaction.to(self.device), calc_time=calc_time)
            else:
                if cold_items is not None:
                    if ui_graph_known is None:
                        scores = self.model.full_sort_predict(
                            interaction.to(self.device), cold_items=cold_items)
                    else:
                        scores = self.model.full_sort_predict(
                            interaction.to(self.device), cold_items=cold_items, ui_graph_known=ui_graph_known)
                else:
                    scores = self.model.full_sort_predict(
                        interaction.to(self.device))

        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(
                self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        # mask the first column of each row
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        # mask the cold/hot items
        if extra_mask_items is not None:
            extra_mask_items = list(extra_mask_items)
            scores[:, extra_mask_items] = -np.inf

        if calc_time:
            return interaction, scores, positive_u, positive_i, time
        else:
            return interaction, scores, positive_u, positive_i

    def _neg_sample_batch_eval(self, batched_data, extra_mask_items=None):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config["eval_type"] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config["eval_type"] == EvaluatorType.RANKING:
            col_idx = interaction[self.config["ITEM_ID_FIELD"]]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full(
                (batch_user_num, self.tot_item_num), -np.inf, device=self.device
            )
            scores[row_idx, col_idx] = origin_scores

            if extra_mask_items is not None:
                extra_mask_items = list(extra_mask_items)
                scores[:, extra_mask_items] = -np.inf

            return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False, extra_mask_items=None, cold_items=None, ui_graph_known=None, saved_ui_emb=False, calc_time=False
    ):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.
            extra_mask_items (set): Hot items are masked during evaluation on eval_cold_data and test_cold_data.
                                    Cold items are masked during evaluation on eval_hot_data and test_hot_data.
        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()

        # Evaluation by full sorting the whole candidate items
        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        # Evaluation by negative sampling
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        time_records = []
        num_sample = 0
        
        # debug
        self.model.restore_user_e, self.model.restore_item_e = None, None
        
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            if calc_time:
                if cold_items is not None:     
                    interaction, scores, positive_u, positive_i, time = eval_func(
                            batched_data, extra_mask_items, cold_items=cold_items, ui_graph_known=ui_graph_known, calc_time=True)
                else:
                    interaction, scores, positive_u, positive_i, time = eval_func(
                        batched_data, extra_mask_items, calc_time=True)
                if time is not None:
                    time_records.append(time)
            else:
                if cold_items is not None:     
                    interaction, scores, positive_u, positive_i = eval_func(
                            batched_data, extra_mask_items, cold_items=cold_items, ui_graph_known=ui_graph_known, calc_time=False)
                else:
                    interaction, scores, positive_u, positive_i = eval_func(
                        batched_data, extra_mask_items, calc_time=False)                
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " +
                              get_gpu_usage(self.device), "yellow")
                )
            self.eval_collector.eval_batch_collect(

                scores, interaction, positive_u, positive_i
            )

        
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        
        # save_ui_emb
        if saved_ui_emb:
            self._save_ui_embedding()
            
            print("INFO: save ui embeddings!")
        
        if calc_time:
            inference_time = sum(time_records) / len(time_records)
            
            print("INFO: cal inference time!")

            return result, inference_time
        else:
            return result

    def _map_reduce(self, result, num_sample):
        gather_result = {}
        total_sample = [
            torch.zeros(1).to(self.device) for _ in range(self.config["world_size"])
        ]
        torch.distributed.all_gather(
            total_sample, torch.Tensor([num_sample]).to(self.device)
        )
        total_sample = torch.cat(total_sample, 0)
        total_sample = torch.sum(total_sample).item()
        for key, value in result.items():
            result[key] = torch.Tensor([value * num_sample]).to(self.device)
            gather_result[key] = [
                torch.zeros_like(result[key]).to(self.device)
                for _ in range(self.config["world_size"])
            ]
            torch.distributed.all_gather(gather_result[key], result[key])
            gather_result[key] = torch.cat(gather_result[key], dim=0)
            gather_result[key] = round(
                torch.sum(gather_result[key]).item() / total_sample,
                self.config["metric_decimal_place"],
            )
        return gather_result

    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size -
                     1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(
                Interaction(current_interaction).to(self.device)
            )
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)



class FirzenTrainer(Trainer):
    r"""trainer for Firzen
    """
    def __init__(self, config, model):

        self.model = model
        self.D = self.model.D
        self.D_lr = config['D_lr']
        self.lr = config['learning_rate']
        
        self.optim_D = optim.Adam(
            self.D.parameters(), lr=self.D_lr, betas=(0.5, 0.9))
        self.optimizer_D = optim.AdamW(
            [
                {'params': self.model.parameters()},
            ], lr=self.lr
        )
        self.schduler_D = self.set_lr_scheduler

        super(FirzenTrainer, self).__init__(config, model)

    def set_lr_scheduler(self):
        def fac(epoch): return 0.96 ** (epoch / 50)
        scheduler_D = optim.lr_scheduler.LambdaLR(
            self.optimizer_D, lr_lambda=fac)
        return scheduler_D

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.

        """

        if not self.config["single_spec"]:
            train_data.knowledge_shuffle(epoch_idx)
        # In Python, the or operator returns the first operand if it is evaluated to True, otherwise it returns the second operand.
        # loss_func = loss_func or self.model.calculate_loss
        train_data.set_mode(KGDataLoaderState.RS)

        loss_func_D = self.model.calculate_loss_D
        loss_func_batch = self.model.calculate_loss_batch
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = amp.GradScaler(enabled=self.enable_scaler)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer_D.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                # losses = loss_func(interaction)
                loss_D = loss_func_D(
                    interaction, idx=batch_idx, total_idx=len(train_data))

            self.optim_D.zero_grad()
            loss_D.backward()
            self.optim_D.step()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                # losses = loss_func(interaction)
                losses = loss_func_batch(
                    interaction, idx=batch_idx, total_idx=len(train_data))

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)

            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer_D)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " +
                              get_gpu_usage(self.device), "yellow")
                )


        # train kg

        train_data.set_mode(KGDataLoaderState.KG)
        # kg_total_loss = total_loss

        kg_total_loss = super()._train_epoch(
            train_data,
            epoch_idx,
            loss_func=self.model.calculate_kg_loss,
            show_progress=show_progress,
        )

        # update A

        self.model.eval()
        with torch.no_grad():
            self.model.update_attentive_A()
            # update embeddings every epoch
            mask = torch.ones(
                [self.model.num_item, self.model.num_item], device=self.device)
            mask[list(self.model.hot_items), :] = 0
            mask[:, list(self.model.cold_items)] = 0
            mask[:, list(self.model.hot_items)] = 1
            mask[list(self.model.cold_items), :] = 1
            with torch.no_grad():
                _, _ = self.model.forward_predict(
                    self.model.matrix_to_tensor(self.model.csr_norm(
                        self.model.ui_graph, mean_flag=True)),
                    self.model.matrix_to_tensor(self.model.csr_norm(
                        self.model.iu_graph, mean_flag=True)),
                    self.model.image_adj * mask if self.model.v_feat is not None else None,
                    self.model.text_adj * mask if self.model.t_feat is not None else None,
                    self.model.mm_adj * mask if self.model.m_feat is not None else None,
                    self.model.kg_adj * mask if self.model.enable_kg else None)

        torch.cuda.empty_cache()

        return total_loss, kg_total_loss
