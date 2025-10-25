import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, num_risk_groups=4):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_risk_groups = num_risk_groups

    def forward(self, embeddings, survival_times, censor):

        device = embeddings.device

        uncensored_idx = (censor == 1).nonzero(as_tuple=True)[0]

        if len(uncensored_idx) < self.num_risk_groups:
            return torch.tensor(0.0, requires_grad=True, device=device)

        embeddings_u = embeddings[uncensored_idx]
        times_u = survival_times[uncensored_idx]

        quantiles = torch.quantile(
            times_u.float(),
            torch.linspace(0, 1, self.num_risk_groups + 1, device=device),
        )

        # assign patients to risk group
        risk_groups = torch.zeros(len(times_u), dtype=torch.long, device=device)

        for i in range(self.num_risk_groups):
            mask = (times_u >= quantiles[i]) & (times_u < quantiles[i + 1])
            risk_groups[mask] = i

        risk_groups[times_u >= quantiles[-1]] = self.num_risk_groups - 1

        z_norm = F.normalize(embeddings_u, dim=1)

        sim_matrix = z_norm @ z_norm.T / self.temperature

        # diagonal mask

        mask_diag = torch.eye(len(times_u), dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask_diag, -float("inf"))

        losses = []

        for i in range(len(times_u)):

            anchor_group = risk_groups[i]

            # postive for same risk group
            positive_mask = (risk_groups == anchor_group) & ~mask_diag[i]

            # negative for different risk group
            group_diff = torch.abs(risk_groups - anchor_group)

            # hard negative if at least one group away
            negative_mask = (group_diff >= 1) & ~mask_diag[i]

            if not positive_mask.any() or not negative_mask.any():
                continue

            pos_sim = sim_matrix[i][positive_mask]
            neg_sim = sim_matrix[i][negative_mask]

            # standard InfoNCE
            numerator = torch.logsumexp(pos_sim, dim=0)
            denominator = torch.logsumexp(torch.cat([pos_sim, neg_sim]), dim=0)

            loss = -numerator + denominator
            losses.append(loss)

        if not losses:
            print(f"If not losses trigged because losses is: {losses}")
            return torch.tensor(0.0, requires_grad=True, device=device)

        return torch.mean(torch.stack(losses))


class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, wsi_embeddings, omic_embeddings):
        """
        want to force embeddings to be similar for the same patient
        """
        # normalize vectors
        cosine_sim = F.cosine_similarity(wsi_embeddings, omic_embeddings)
        cosine_sim = cosine_sim.clamp(min=-1, max=1)

        # need to define for intrapatient

        # need to define for interpatient

        # need to to a max between 0 and M - interpatient + intrapatient

        loss = torch.mean(1 - cosine_sim)

        return loss


class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, log_risks, times, censor):
        """
        :param log_risks: predictions from the NN
        :param times: observed survival times (i.e. times to death) for the batch
        :param censor: censor data, event (death) indicators (1/0)
        :return: Cox loss (scalar)
        """
        device = log_risks.device
        # numerical stability
        log_risks = torch.clamp(log_risks, min=-10, max=10)
        sorted_times, sorted_indices = torch.sort(times, descending=True)
        sorted_log_risks = log_risks[sorted_indices]
        sorted_censor = censor[sorted_indices]

        # precompute for using within the inner sum of term 2 in Cox loss
        exp_sorted_log_risks = torch.exp(sorted_log_risks)

        # initialize all samples to be at-risk (will update it below)
        at_risk_mask = torch.ones_like(sorted_times, dtype=torch.bool)

        losses = []
        for time_index in range(len(sorted_times)):
            # include only the uncensored samples (i.e., for whom the event has happened)
            if sorted_censor[time_index] == 1:
                at_risk_mask = (
                    torch.arange(len(sorted_times)) <= time_index
                )  # less than, as sorted_times is in descending order
                at_risk_mask = at_risk_mask.to(device)
                at_risk_sum = torch.sum(
                    exp_sorted_log_risks[at_risk_mask]  # 2nd term on the RHS
                )  # all are at-risk for the first sample (after arranged in descending order)
                loss = sorted_log_risks[time_index] - torch.log(at_risk_sum + 1e-15)
                losses.append(loss)

            # at_risk_mask[time_index] = False # the i'th sample is no more in the risk-set as the event has already occurred for it

        # if no uncensored samples are in the mini-batch return 0
        if not losses:
            return torch.tensor(0.0, requires_grad=True)

        cox_loss = -torch.mean(torch.stack(losses))
        return cox_loss


class JointLoss(nn.Module):
    def __init__(
        self, sim_weight=1.0, contrast_weight=0.1, temperature=0.1, num_risk_groups=4
    ):
        super(JointLoss, self).__init__()

        self.cox_loss_fn = CoxLoss()
        self.sim_loss_fn = SimilarityLoss()
        self.contrast_loss_fn = ContrastiveLoss(
            temperature=temperature, num_risk_groups=num_risk_groups
        )

        self.sim_weight = sim_weight
        self.contrast_weight = contrast_weight

    def forward(self, log_risks, times, censor, wsi_embeddings, omic_embeddings):
        cox_loss = self.cox_loss_fn(log_risks, times, censor)
        sim_loss = self.sim_loss_fn(wsi_embeddings, omic_embeddings)
        wsi_contrastive_loss = self.contrast_loss_fn(wsi_embeddings, times, censor)
        omic_contrastive_loss = self.contrast_loss_fn(omic_embeddings, times, censor)
        contrastive_loss = (wsi_contrastive_loss + omic_contrastive_loss) / 2

        return (
            cox_loss
            + self.sim_weight * sim_loss
            + self.contrast_weight * contrastive_loss
        )
