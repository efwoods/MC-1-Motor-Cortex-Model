import torch.nn.functional as F


def latent_alignment_loss(motion_latent, waveform_latent):
    """
    Aligns motion & waveform latents so similar motion and waveforms are close together in the same space.
    """
    mse = F.mse_loss(motion_latent, waveform_latent)
    cos_sim = F.cosine_similarity(motion_latent, waveform_latent, dim=-1)
    penalty = 0.5 * (1 - cos_sim).mean()
    return mse + penalty


def total_loss_fn(
    xyz_true, xyz_recon, ecog_true, ecog_synth, motion_latent, waveform_latent
):
    loss_motion_recon = F.mse_loss(xyz_recon, xyz_true)
    loss_waveform_recon = F.mse_loss(ecog_synth, ecog_true)
    loss_latent_align = latent_alignment_loss(motion_latent, waveform_latent)

    return loss_motion_recon + loss_waveform_recon + loss_latent_align
