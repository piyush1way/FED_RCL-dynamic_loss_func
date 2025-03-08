import copy
import time
import gc

import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from omegaconf import OmegaConf

from utils import *
from utils.loss import ContrastiveLoss
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List
from utils.logging_utils import AverageMeter

import logging
logger = logging.getLogger(__name__)

from clients.build import CLIENT_REGISTRY
from clients import Client


@CLIENT_REGISTRY.register()
class RCLClient(Client):
    def __init__(self, args, client_index, model):
        self.args = args
        self.client_index = client_index
        self.loader = None

        self.model = model
        self.global_model = copy.deepcopy(model)

        self.rcl_criterions = {"scl": None, "penalty": None}
        args_rcl = args.client.rcl_loss
        self.global_epoch = 0

        self.pairs = {}
        for pair in args_rcl.pairs:
            self.pairs[pair.name] = pair

            # Convert OmegaConf DictConfig to a mutable dictionary
            args_rcl_dict = OmegaConf.to_container(args_rcl, resolve=True)

            # Remove unwanted keys (e.g., "weight") and keep only valid ContrastiveLoss arguments
            allowed_keys = {"temp", "margin", "lr", "dynamic_beta", "beta_min", "beta_max", "beta_decay", "class_aware_beta"}
            args_contrastive = {k: v for k, v in args_rcl_dict.items() if k in allowed_keys}

            # Rename "temp" to "temperature" if it exists, since some implementations use this
            if "temp" in args_contrastive:
                args_contrastive["temperature"] = args_contrastive.pop("temp")

            # Initialize the ContrastiveLoss with the filtered arguments
            loss_obj = ContrastiveLoss(**args_contrastive)

            # Attach the pair object to the loss instance to avoid attribute errors later
            loss_obj.pair = pair

            self.rcl_criterions[pair.name] = loss_obj

        self.criterion = nn.CrossEntropyLoss()

    def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        self._update_model(state_dict)
        self._update_global_model(state_dict)

        for fixed_model in [self.global_model]:
            for n, p in fixed_model.named_parameters():
                p.requires_grad = False

        self.device = device
        self.num_layers = self.model.num_layers

        # Initialize DataLoader
        train_sampler = None
        if self.args.dataset.num_instances > 0:
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances)
        self.loader = DataLoader(
            local_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=local_lr,
            momentum=self.args.optimizer.momentum,
            weight_decay=self.args.optimizer.wd,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.args.trainer.local_lr_decay**epoch,
        )

        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
        self.num_classes = len(self.loader.dataset.dataset.classes)

        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {}
        for key in sorted_key:
            sorted_class_dict[key] = local_dataset.class_dict[key]

        if self.args.client.get("LC"):
            self.label_dist = torch.zeros(len(local_dataset.dataset.classes), device=self.device)
            for key in sorted_class_dict:
                self.label_dist[int(key)] = sorted_class_dict[key]

        if global_epoch == 0:
            logger.warning(f"Class counts : {self.class_counts}")

        self.sorted_class_dict = sorted_class_dict
        self.trainer = trainer

    def _algorithm_rcl(self, local_results, global_results, labels):
        """
        Computes the RCL (Relaxed Contrastive Learning) loss for each layer using ContrastiveLoss.
        For each layer, we use:
        - z_prev: global feature (from the server/global model)
        - z_present: local feature (from the client model)
        - z_serv: global feature (used for regularization)
        """
        losses = {"cossim": []}
        rcl_args = self.args.client.rcl_loss

        for l in range(self.num_layers):
            # Decide whether to train this layer based on branch_level.
            train_layer = False
            if rcl_args.branch_level is False or l in rcl_args.branch_level:
                train_layer = True

            local_feature_l = local_results[f"layer{l}"]
            global_feature_l = global_results[f"layer{l}"]

            if len(local_feature_l.shape) == 4:
                local_feature_l = F.adaptive_avg_pool2d(local_feature_l, 1)
                global_feature_l = F.adaptive_avg_pool2d(global_feature_l, 1)

            # Compute feature cosine similarity loss for alignment.
            if self.args.client.feature_align_loss.align_type == "l2":
                loss_cossim = F.mse_loss(
                    local_feature_l.squeeze(-1).squeeze(-1),
                    global_feature_l.squeeze(-1).squeeze(-1),
                )
            else:
                loss_cossim = F.cosine_embedding_loss(
                    local_feature_l.squeeze(-1).squeeze(-1),
                    global_feature_l.squeeze(-1).squeeze(-1),
                    torch.ones_like(labels),
                )
            losses["cossim"].append(loss_cossim)

            # Compute RCL loss using ContrastiveLoss.
            if train_layer:
                for sub_loss_name in self.rcl_criterions:
                    rcl_criterion = self.rcl_criterions[sub_loss_name]
                    if rcl_criterion is not None:
                        # If the pair config specifies branch_level, verify this layer should be trained.
                        if rcl_criterion.pair.get("branch_level"):
                            train_layer = l in rcl_criterion.pair.branch_level
                        print(f"Global Epoch: {self.global_epoch}")  # Debugging
                        assert self.global_epoch is not None, "Error: global_epoch is None"
                        if train_layer:
                            # Call ContrastiveLoss using the new parameter names.
                            # Use global_feature_l as both z_prev and z_serv, and local_feature_l as z_present.
                            loss_rcl = rcl_criterion(
                                z_prev=global_feature_l,
                                z_present=local_feature_l,
                                z_serv=global_feature_l,
                                labels=labels,  # Pass labels for class-aware divergence penalty
                                epoch=self.global_epoch,  # Pass the current epoch
                            )
                            if sub_loss_name not in losses:
                                losses[sub_loss_name] = []
                            losses[sub_loss_name].append(loss_rcl)

        # Aggregate losses over layers.
        for loss_name in losses:
            try:
                if len(losses[loss_name]) > 0:
                    losses[loss_name] = torch.mean(torch.stack(losses[loss_name]))
                else:
                    losses[loss_name] = 0
            except Exception as e:
                print("Error stacking losses for", loss_name, ":", e)
                breakpoint()

        return losses

    def _algorithm(self, images, labels) -> Dict:
        losses = defaultdict(float)
        no_relu = not self.args.client.rcl_loss.feature_relu
        results = self.model(images, no_relu=no_relu)
        with torch.no_grad():
            global_results = self.global_model(images, no_relu=no_relu)

        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        ## Prox Loss
        prox_loss = 0
        fixed_params = {n: p for n, p in self.global_model.named_parameters()}
        for n, p in self.model.named_parameters():
            prox_loss += ((p - fixed_params[n].detach()) ** 2).sum()
        losses["prox"] = prox_loss

        losses.update(self._algorithm_rcl(local_results=results, global_results=global_results, labels=labels))

        features = {
            "local": results,
            "global": global_results,
        }

        return losses, features

    def get_weights(self, epoch=None):
        weights = {
            "cls": self.args.client.ce_loss.weight,
            "cossim": self.args.client.feature_align_loss.weight,
        }

        if self.args.client.get("prox_loss"):
            weights["prox"] = self.args.client.prox_loss.weight

        for pair in self.args.client.rcl_loss.pairs:
            weights[pair.name] = pair.weight
        return weights

    @property
    def current_progress(self):
        return self.global_epoch / self.args.trainer.global_rounds

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch

        # Move models to the GPU
        self.model.to(self.device)
        self.global_model.to(self.device)

        scaler = GradScaler()
        start = time.time()
        loss_meter = AverageMeter("Loss", ":.2f")
        time_meter = AverageMeter("BatchTime", ":3.1f")

        self.weights = self.get_weights(epoch=global_epoch)

        if global_epoch % 50 == 0:
            print(self.weights)

        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()
            for i, (images, labels) in enumerate(self.loader):
                # Move data to the GPU
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):
                    losses, features = self._algorithm(images, labels)

                    for loss_key in losses:
                        if loss_key not in self.weights.keys():
                            self.weights[loss_key] = 0

                    loss = sum([self.weights[loss_key] * losses[loss_key] for loss_key in losses])

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                scaler.step(self.optimizer)
                scaler.update()

                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)

                end = time.time()

            self.scheduler.step()

        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f},")

        # Move models back to CPU (optional, depending on your use case)
        self.model.to("cpu")
        self.global_model.to("cpu")

        loss_dict = {f"loss/{self.args.dataset.name}/{loss_key}": float(losses[loss_key]) for loss_key in losses}

        gc.collect()

        return self.model.state_dict(), loss_dict
