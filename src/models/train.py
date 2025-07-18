import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import MotionECoGDataset
from motion_encoder import MotionEncoder
from variational_motion_encoder import VariationalMotionEncoder
from waveform_decoder import WaveformDecoder
from waveform_encoder import WaveformEncoder
from variational_waveform_encoder import VariationalWaveformEncoder
from motion_decoder import MotionDecoder
from loss import total_loss_fn, kl_anneal_function

import os
import time


def evaluate(models, dataloader, device, beta):
    motion_encoder, waveform_decoder, waveform_encoder, motion_decoder = models
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            motion = batch["motion"].to(device)
            ecog = batch["ecog"].to(device)

            motion_latent, motion_mean, motion_logvar = motion_encoder(motion)
            ecog_synth = waveform_decoder(motion_latent)
            waveform_latent, waveform_mean, waveform_logvar = waveform_encoder(
                ecog_synth
            )
            motion_reconstructed = motion_decoder(waveform_latent)

            val_losses = total_loss_fn(
                xyz_true=motion,
                xyz_reconstructed=motion_reconstructed,
                ecog_true=ecog,
                ecog_synth=ecog_synth,
                motion_latent=motion_latent,
                waveform_latent=waveform_latent,
                motion_mean=motion_mean,
                motion_logvar=motion_logvar,
                waveform_mean=waveform_mean,
                waveform_logvar=waveform_logvar,
                beta=beta,
            )

            total_loss += val_losses["total_loss"].item()
            normalized_total_loss = total_loss / len(dataloader)

    return normalized_total_loss, val_losses


def train(
    motion_np,
    ecog_np,
    batch_size=128,
    epochs=300,
    lr=5e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_every=10,
    checkpoint_dir="checkpoints",
    log_dir="runs",
    val_split=0.1,
    early_stopping_patience=10,
):
    best_val_loss = float("inf")
    epochs_no_improve = 0
    # Prepare dataset
    dataset = MotionECoGDataset(motion_np, ecog_np)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # Initialize models
    latent_dim = 128

    motion_encoder = VariationalMotionEncoder(latent_dim=latent_dim).to(device)

    # MotionEncoder(latent_dim).to(device)

    waveform_decoder = WaveformDecoder(latent_dim=latent_dim).to(device)

    waveform_encoder = VariationalWaveformEncoder(
        input_dim=64, hidden_dim=128, latent_dim=128
    ).to(device)

    # WaveformEncoder(latent_dim).to(device)
    motion_decoder = MotionDecoder(latent_dim).to(device)

    models = [motion_encoder, waveform_decoder, waveform_encoder, motion_decoder]

    model_file_names = [
        "motion_encoder",
        "waveform_decoder",
        "waveform_encoder",
        "motion_decoder",
    ]

    optimizer = Adam([p for m in models for p in m.parameters()], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Create folders
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_id))

    for epoch in range(epochs):
        beta = kl_anneal_function(epoch, max_beta=0.0001, start_epoch=0, end_epoch=50)
        total_loss = 0.0
        motion_encoder.train()
        waveform_decoder.train()
        waveform_encoder.train()
        motion_decoder.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            motion = batch["motion"].to(device)
            ecog = batch["ecog"].to(device)

            optimizer.zero_grad()

            motion_latent, motion_mean, motion_logvar = motion_encoder(motion)

            ecog_synth = waveform_decoder(motion_latent)
            waveform_latent, waveform_mean, waveform_logvar = waveform_encoder(
                ecog_synth
            )
            motion_reconstructed = motion_decoder(waveform_latent)

            losses = total_loss_fn(
                xyz_true=motion,
                xyz_reconstructed=motion_reconstructed,
                ecog_true=ecog,
                ecog_synth=ecog_synth,
                motion_latent=motion_latent,
                waveform_latent=waveform_latent,
                motion_mean=motion_mean,
                motion_logvar=motion_logvar,
                waveform_mean=waveform_mean,
                waveform_logvar=waveform_logvar,
                beta=beta,
            )

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for m in models for p in m.parameters()], 1.0
            )
            optimizer.step()

            total_loss += losses["total_loss"].item()

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on validation
        motion_encoder.eval()
        waveform_decoder.eval()
        waveform_encoder.eval()
        motion_decoder.eval()

        avg_val_loss, val_losses = evaluate(models, val_loader, device, beta)
        scheduler.step(avg_val_loss)
        print(f"Current learning rate: {scheduler.get_last_lr()[0]}")
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch + 1)

        # Logging to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch + 1)
        # Optional: log individual components if desired
        writer.add_scalar(
            "Loss/MotionReconstruction", val_losses["motion_recon"].item(), epoch + 1
        )
        writer.add_scalar(
            "Loss/WaveformReconstruction",
            val_losses["waveform_recon"].item(),
            epoch + 1,
        )
        writer.add_scalar(
            "Loss/LatentAlignment", val_losses["latent_align"].item(), epoch + 1
        )
        # writer.add_scalar("Loss/KL_Motion", val_losses["kl_motion"].item(), epoch + 1)
        # writer.add_scalar(
        #     "Loss/KL_Waveform", val_losses["kl_waveform"].item(), epoch + 1
        # )

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Save checkpoints
        if (epoch + 1) % save_every == 0:
            for i, model in enumerate(models):
                torch.save(
                    model.state_dict(),
                    f"{checkpoint_dir}/{model_file_names[i]}_model_epoch_{epoch+1}.pt",
                )

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            # Optional: Save the best model
            for i, model in enumerate(models):
                torch.save(
                    model.state_dict(),
                    f"{checkpoint_dir}/{model_file_names[i]}_best.pt",
                )
        else:
            epochs_no_improve += 1
            print(f"No improvement in {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
    writer.close()


if __name__ == "__main__":
    motion_np = "../data/motion_values_normalized.npy"
    ecog_np = "../data/ecog_values_normalized.npy"
    train(motion_np, ecog_np)
