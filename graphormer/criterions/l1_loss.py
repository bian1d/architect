# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass

import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("l1_loss", dataclass=FairseqDataclass)
class GraphPredictionL1Loss(FairseqCriterion):
    """
    Implementation for the L1 loss (MAE loss) used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["net_input"]["atoms"].shape[0]
        # 就是说这里的atoms已经出问题了，不是全都是负数。

        with torch.no_grad():
            natoms = sample["net_input"]["atoms"].shape[1]

        ## mogai by zeh ##
        atoms = sample["net_input"]["atoms"].squeeze()
        poses = sample["net_input"]["pos"]

        poses.requires_grad = True

        tags = sample["net_input"]["tags"]
        real_mask = sample["net_input"]["real_mask"] 

        logits = model(atoms=atoms, tags=tags, pos=poses, real_mask=real_mask)
        energy, force, bsz = logits
        # [bsz], [bsz, natoms, 3]
        true_energy, true_force = sample["targets"]["energy"], sample["targets"]["forces"]
        true_energy = true_energy[:, 0]
        force = force[real_mask].reshape(sample_size, -1, 3)
        true_force = true_force[real_mask].reshape(sample_size, -1, 3)
        # [bsz], [bsz, natoms, 3]
        energy_loss = nn.L1Loss(reduction="mean")(energy, true_energy)
        force_loss = nn.L1Loss(reduction="mean")(force, true_force)

        # loss = energy_loss
        loss = force_loss
        import pdb; pdb.set_trace()
        # import pudb; pudb.set_trace() 

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("l1_loss_with_flag", dataclass=FairseqDataclass)
class GraphPredictionL1LossWithFlag(GraphPredictionL1Loss):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def perturb_forward(self, model, sample, perturb, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        batch_data = sample["net_input"]["batched_data"]["x"]
        with torch.no_grad():
            natoms = batch_data.shape[1]
        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        targets = model.get_targets(sample, [logits])
        loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output
