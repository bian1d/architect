# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass

import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
import numpy as np

def generate_online(n_electrons, batch_size):
    """
    生成电荷和位置数据，并计算力和势能
    """
    # 初始化数据张量
    charges = np.random.randint(1, 10, size=(batch_size, n_electrons))  # 随机生成电荷
    positions = np.random.uniform(-10, 10, size=(batch_size, n_electrons, 3))  # 位置在-10到10之间
    tags = np.ones((batch_size, n_electrons), dtype=np.long)  # 标签全1
    real_mask = np.ones((batch_size, n_electrons), dtype=bool)  # 掩码全True

    forces = np.zeros_like(positions)
    potential_energy = np.zeros((batch_size, 1))

    # 计算力和势能
    for b in range(batch_size):
        batch_forces, batch_potential_energy = calculate_forces_and_potential_iter_1(
            positions[b], charges[b]
        )
        forces[b] = batch_forces
        potential_energy[b] = batch_potential_energy

    return charges, positions, tags, real_mask, forces, potential_energy


def calculate_forces_and_potential_iter_1(positions, charges):
    k_e = 1  # 库伦常数，用了原子单位制
    n_electrons, _ = positions.shape
    
    # 计算位置差
    r_vec = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_mag = np.linalg.norm(r_vec, axis=-1)
    r_mag[r_mag == 0] = np.inf
    # 单位向量
    r_hat = r_vec / r_mag[..., np.newaxis]
    
    # 计算力的大小
    force_magnitude = k_e * charges[:, np.newaxis] * charges[np.newaxis, :] / r_mag ** 2
    
    # 计算力
    forces = np.sum(force_magnitude[..., np.newaxis] * r_hat, axis=1)
    
    # 计算势能
    potential_energy = np.sum(k_e * charges[:, np.newaxis] * charges[np.newaxis, :] / r_mag) / 2  

    
    return forces, potential_energy


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
        ## read from ##
        torch.autograd.set_detect_anomaly(True)
        # with torch.no_grad():
        #     sample_size = sample["net_input"]["atoms"].shape[0]
        #     natoms = sample["net_input"]["atoms"].shape[1]

        # ## mogai by zeh ##
        # atoms = sample["net_input"]["atoms"].squeeze()
        # poses = sample["net_input"]["pos"]

        # poses.requires_grad = True

        # tags = sample["net_input"]["tags"]
        # real_mask = sample["net_input"]["real_mask"] 


        # logits = model(atoms=atoms, tags=tags, pos=poses, real_mask=real_mask)
        # energy, force, bsz = logits
        # # [bsz], [bsz, natoms, 3]
        # true_energy, true_force = sample["targets"]["energy"], sample["targets"]["forces"]
        # true_energy = true_energy[:, 0]
        # force = force[real_mask].reshape(sample_size, -1, 3)
        # true_force = true_force[real_mask].reshape(sample_size, -1, 3)
        # [bsz], [bsz, natoms, 3]

        # online version #
        with torch.enable_grad():
            sample_size = 64
            natoms = 16
            charges, positions, tags, real_mask, true_force, true_energy = generate_online(n_electrons=natoms, batch_size=sample_size)
            charges = torch.tensor(charges, dtype=torch.long).to(device="cuda")
            positions = torch.tensor(positions, dtype=torch.float, requires_grad=True).to(device="cuda")
            tags = torch.tensor(tags, dtype=torch.long).to(device="cuda")
            real_mask = torch.tensor(real_mask, dtype=torch.bool).to(device="cuda")
            true_force= torch.tensor(true_force, dtype=torch.float).to(device="cuda")
            true_energy = torch.tensor(true_energy, dtype=torch.float).to(device="cuda")
            true_energy = true_energy[:, 0]
            logits = model(atoms=charges, tags=tags, pos=positions, real_mask=real_mask)
            energy, force, bsz = logits
            force = force[real_mask].reshape(sample_size, -1, 3)
            true_force = true_force[real_mask].reshape(sample_size, -1, 3)
        

        energy_loss = nn.L1Loss(reduction="sum")(energy, true_energy)
        force_loss = nn.L1Loss(reduction="mean")(force, true_force) * sample_size

        # loss = energy_loss
        loss = force_loss

        logging_output = {
            "loss": loss.data, 
            "force_loss": force_loss.data,
            "enegy_loss": energy_loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        # import pdb; pdb.set_trace()
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
