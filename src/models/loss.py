import torch.nn.functional as F
import torch


# def kl_divergence(mean, logvar):
#     return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1).mean()


def kl_divergence(mean, logvar):
    # mean, logvar shape: [batch_size, latent_dim]
    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    # sum over latent dimension, mean over batch
    return kl.sum(dim=1).mean()


def kl_anneal_function(epoch, max_beta=0.01, start_epoch=0, end_epoch=50):
    """
    Linearly increase beta from 0 to max_beta between start_epoch and end_epoch.
    """
    if epoch < start_epoch:
        return 0.0
    elif epoch > end_epoch:
        return max_beta
    else:
        return max_beta * (epoch - start_epoch) / (end_epoch - start_epoch)


def latent_alignment_loss(motion_latent, waveform_latent):
    """
    Aligns motion & waveform latents so similar motion and waveforms are close together in the same space.
    """
    cos_sim = F.cosine_similarity(motion_latent, waveform_latent, dim=-1)
    penalty = (1 - cos_sim).mean()
    return penalty


def total_loss_fn(
    xyz_true,
    xyz_reconstructed,
    ecog_true,
    ecog_synth,
    motion_latent,
    waveform_latent,
    motion_mean,
    motion_logvar,
    waveform_mean,
    waveform_logvar,
    beta,  # KL Annealing will gradually increment the KL Divergence penalty
):

    # Hyperparameters
    alpha = 0.5  # cos loss (latent alignment)
    # beta = 0.001  # KL Divergence (regularization to smooth and avoid overfitting latent loss)

    beta = 0

    loss_motion_reconstructed = F.mse_loss(xyz_reconstructed, xyz_true)
    loss_waveform_reconstructed = F.mse_loss(ecog_synth, ecog_true)
    loss_latent_align = latent_alignment_loss(motion_latent, waveform_latent)
    kl_loss_motion = kl_divergence(motion_mean, motion_logvar)
    kl_loss_waveform = kl_divergence(waveform_mean, waveform_logvar)

    total_loss = (
        loss_motion_reconstructed
        + loss_waveform_reconstructed
        + alpha * loss_latent_align
        + beta * (kl_loss_motion + kl_loss_waveform)
    )

    return {
        "total_loss": total_loss,
        "motion_recon": loss_motion_reconstructed,
        "waveform_recon": loss_waveform_reconstructed,
        "latent_align": loss_latent_align,
        "kl_motion": kl_loss_motion,
        "kl_waveform": kl_loss_waveform,
    }
