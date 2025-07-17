import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import MotionECoGDataset
from motion_encoder import MotionEncoder
from waveform_decoder import WaveformDecoder
from waveform_encoder import WaveformEncoder
from motion_decoder import MotionDecoder
from loss import total_loss_fn

import os


def evaluate(models, dataloader, device):
    motion_encoder, waveform_decoder, waveform_encoder, motion_decoder = models
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            motion = batch["motion"].to(device)
            ecog = batch["ecog"].to(device)

            motion_latent = motion_encoder(motion)
            ecog_synth = waveform_decoder(motion_latent)
            waveform_latent = waveform_encoder(ecog_synth)
            motion_recon = motion_decoder(waveform_latent)

            loss = total_loss_fn(
                xyz_true=motion,
                xyz_recon=motion_recon,
                ecog_true=ecog,
                ecog_synth=ecog_synth,
                motion_latent=motion_latent,
                waveform_latent=waveform_latent,
            )

            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(
    motion_csv,
    ecog_csv,
    batch_size=128,
    epochs=50,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_every=10,
    checkpoint_dir="checkpoints",
    log_dir="runs",
    val_split=0.1,
):
    # Prepare dataset
    dataset = MotionECoGDataset(motion_csv, ecog_csv)
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
    motion_encoder = MotionEncoder(latent_dim).to(device)
    waveform_decoder = WaveformDecoder(latent_dim).to(device)
    waveform_encoder = WaveformEncoder(latent_dim).to(device)
    motion_decoder = MotionDecoder(latent_dim).to(device)

    models = [motion_encoder, waveform_decoder, waveform_encoder, motion_decoder]
    optimizer = Adam([p for m in models for p in m.parameters()], lr=lr)

    # Create folders
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        total_loss = 0.0
        motion_encoder.train()
        waveform_decoder.train()
        waveform_encoder.train()
        motion_decoder.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            motion = batch["motion"].to(device)
            ecog = batch["ecog"].to(device)

            optimizer.zero_grad()

            motion_latent = motion_encoder(motion)
            ecog_synth = waveform_decoder(motion_latent)
            waveform_latent = waveform_encoder(ecog_synth)
            motion_recon = motion_decoder(waveform_latent)

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

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on validation
        motion_encoder.eval()
        waveform_decoder.eval()
        waveform_encoder.eval()
        motion_decoder.eval()
        avg_val_loss = evaluate(models, val_loader, device)

        # Logging to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch + 1)

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Save checkpoints
        if (epoch + 1) % save_every == 0:
            for i, model in enumerate(models):
                torch.save(
                    model.state_dict(),
                    f"{checkpoint_dir}/model_{i}_epoch_{epoch+1}.pt",
                )

    writer.close()


if __name__ == "__main__":
    motion_csv = "../data/Contralateral_2018-04-12_(S4)_cleaned_aligned_motion_data_DATA_ONLY.csv"
    ecog_csv = (
        "../data/Contralateral_2018-04-12_(S4)_cleaned_aligned_ecog_data_DATA_ONLY.csv"
    )
    train(motion_csv, ecog_csv)
