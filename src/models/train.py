import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from dataset import MotionECoGDataset
from models import MotionEncoder, WaveformDecoder, WaveformEncoder, MotionDecoder
from loss import total_loss_fn

import os


def train(
    motion_csv,
    ecog_csv,
    batch_size=128,
    epochs=50,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_every=10,
    checkpoint_dir="checkpoints",
):
    # Dataset and DataLoader
    dataset = MotionECoGDataset(motion_csv, ecog_csv)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Models
    latent_dim = 128
    motion_encoder = MotionEncoder(latent_dim).to(device)
    waveform_decoder = WaveformDecoder(latent_dim).to(device)
    waveform_encoder = WaveformEncoder(latent_dim).to(device)
    motion_decoder = MotionDecoder(latent_dim).to(device)

    models = [motion_encoder, waveform_decoder, waveform_encoder, motion_decoder]
    optimizer = Adam([p for m in models for p in m.parameters()], lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            motion = batch["motion"].to(device)  # (B, 3)
            ecog = batch["ecog"].to(device)  # (B, 20, 64)

            optimizer.zero_grad()

            # Forward pass
            motion_latent = motion_encoder(motion)  # (B, latent)
            ecog_synth = waveform_decoder(motion_latent)  # (B, 20, 64)
            waveform_latent = waveform_encoder(ecog_synth)  # (B, latent)
            motion_recon = motion_decoder(waveform_latent)  # (B, 3)

            # Loss
            loss = total_loss_fn(
                xyz_true=motion,
                xyz_recon=motion_recon,
                ecog_true=ecog,
                ecog_synth=ecog_synth,
                motion_latent=motion_latent,
                waveform_latent=waveform_latent,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for m in models for p in m.parameters()], 1.0
            )
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            for i, model in enumerate(models):
                torch.save(
                    model.state_dict(), f"{checkpoint_dir}/model_{i}_epoch_{epoch+1}.pt"
                )


if __name__ == "__main__":
    motion_csv = "motion_data.csv"
    ecog_csv = "ecog_data.csv"
    train(motion_csv, ecog_csv)
