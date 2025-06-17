import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    in_channels: int = 1
    out_channels: int = 1
    num_layers: int = 80
    growth_rate: int = 1
    dilation_rates: List[int] = None

    batch_size: int = 1
    learning_rate: float = 0.001
    num_epochs: int = 50
    weight_decay: float = 1e-5

    dataset_path: str = "sr4zct_exp_dataset"
    output_dir: str = "msdnet_training_results"

    device: str = "cuda"
    num_workers: int = 8

    save_interval: int = 10
    log_interval: int = 50

    def __post_init__(self):
        if self.dilation_rates is None:
            self.dilation_rates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class MixedScaleDenseBlock(nn.Module):

    def __init__(self, in_channels: int, growth_rate: int, dilation_rates: List[int]):
        super().__init__()

        self.growth_rate = growth_rate
        self.dilation_rates = dilation_rates

        self.convs = nn.ModuleList()
        for i in range(growth_rate):
            dilation = dilation_rates[i % len(dilation_rates)]
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            )
            self.convs.append(conv)

        self.bn = nn.BatchNorm2d(growth_rate)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = []
        for conv in self.convs:
            new_feature = conv(x)
            new_features.append(new_feature)

        new_features = torch.cat(new_features, dim=1)

        new_features = self.bn(new_features)
        new_features = self.relu(new_features)

        return torch.cat([x, new_features], dim=1)


class MSDNet(nn.Module):

    def __init__(self, config: TrainingConfig):
        super().__init__()

        self.config = config

        current_channels = config.in_channels

        self.layers = nn.ModuleList()

        for i in range(config.num_layers):
            layer = MixedScaleDenseBlock(
                in_channels=current_channels,
                growth_rate=config.growth_rate,
                dilation_rates=config.dilation_rates
            )
            self.layers.append(layer)
            current_channels += config.growth_rate

        self.final_conv = nn.Conv2d(
            current_channels,
            config.out_channels,
            kernel_size=1,
            bias=True
        )

        self._initialize_weights()

        logger.info(f"MS-DNet initialized with {self._count_parameters():,} parameters")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * 9
                fan_out = module.out_channels * 9
                nc = fan_in + fan_out
                std = (2.0 / nc) ** 0.5
                nn.init.normal_(module.weight, mean=0, std=std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        current_features = x

        for layer in self.layers:
            current_features = layer(current_features)

        output = self.final_conv(current_features)

        if output.shape[1] == identity.shape[1]:
            output = output + identity

        return output


class SR4ZCTDataset(Dataset):

    def __init__(self, dataset_path: str, augmentation: bool = False):
        self.dataset_path = Path(dataset_path)
        self.augmentation = augmentation

        self._load_data()

        logger.info(f"Dataset loaded: {len(self)} samples")

    def _load_data(self):
        try:
            self.vertical_degraded = np.load(self.dataset_path / 'recon_low_vertical.npy')
            self.horizontal_degraded = np.load(self.dataset_path / 'recon_low_horizontal.npy')
            self.reference = np.load(self.dataset_path / 'recon_low.npy')

            assert self.vertical_degraded.shape == self.reference.shape
            assert self.horizontal_degraded.shape == self.reference.shape

            self.num_slices = self.reference.shape[0]

            logger.info(f"Data shapes: {self.reference.shape}")

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {self.dataset_path}: {e}")

    def __len__(self) -> int:
        return self.num_slices * 2

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < self.num_slices:
            input_img = self.vertical_degraded[idx]
            target_img = self.reference[idx]
        else:
            slice_idx = idx - self.num_slices
            input_img = np.rot90(self.horizontal_degraded[slice_idx]).copy()
            target_img = np.rot90(self.reference[slice_idx]).copy()

        input_tensor = torch.from_numpy(input_img[None, ...]).float()
        target_tensor = torch.from_numpy(target_img[None, ...]).float()

        if self.augmentation:
            input_tensor, target_tensor = self._apply_augmentation(input_tensor, target_tensor)

        return input_tensor, target_tensor

    def _apply_augmentation(self, input_img: torch.Tensor, target_img: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        if torch.rand(1) < 0.5:
            input_img = torch.flip(input_img, dims=[2])
            target_img = torch.flip(target_img, dims=[2])

        if torch.rand(1) < 0.5:
            input_img = torch.flip(input_img, dims=[1])
            target_img = torch.flip(target_img, dims=[1])

        return input_img, target_img


class TestCoronalDataset(Dataset):

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.test_volume = np.load(self.dataset_path / 'recon_low_test.npy')
        self.num_slices = self.test_volume.shape[1]

    def __len__(self) -> int:
        return self.num_slices

    def __getitem__(self, idx: int) -> torch.Tensor:
        slice_data = self.test_volume[:, idx, :]
        return torch.from_numpy(slice_data[None, ...]).float()


class TestSagittalDataset(Dataset):

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.test_volume = np.load(self.dataset_path / 'recon_low_test.npy')
        self.num_slices = self.test_volume.shape[2]

    def __len__(self) -> int:
        return self.num_slices

    def __getitem__(self, idx: int) -> torch.Tensor:
        slice_data = self.test_volume[:, :, idx]
        return torch.from_numpy(slice_data[None, ...]).float()


class MetricsCalculator:

    @staticmethod
    def calculate_psnr(pred: np.ndarray, target: np.ndarray, data_range: Optional[float] = None) -> float:
        if data_range is None:
            data_range = target.max() - target.min()

        if data_range == 0:
            data_range = 1.0

        return psnr(target, pred, data_range=data_range)

    @staticmethod
    def calculate_ssim(pred: np.ndarray, target: np.ndarray, data_range: Optional[float] = None) -> float:
        if data_range is None:
            data_range = target.max() - target.min()

        if data_range == 0:
            data_range = 1.0

        return ssim(target, pred, data_range=data_range)

    @staticmethod
    def batch_metrics(pred_batch: torch.Tensor, target_batch: torch.Tensor,
                      fixed_data_range: Optional[float] = None) -> Dict[str, float]:
        pred_np = pred_batch.detach().cpu().numpy()
        target_np = target_batch.detach().cpu().numpy()

        psnr_values = []
        ssim_values = []

        for i in range(pred_batch.shape[0]):
            pred_img = pred_np[i, 0]
            target_img = target_np[i, 0]

            if fixed_data_range is not None:
                data_range = fixed_data_range
            else:
                data_range = 1.0

            psnr_val = MetricsCalculator.calculate_psnr(pred_img, target_img, data_range)
            ssim_val = MetricsCalculator.calculate_ssim(pred_img, target_img, data_range)

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)

        return {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values)
        }


class MSDNetTrainer:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self._setup_model()
        self._setup_data()
        self._setup_training()

        self.train_history = {
            'train_losses': [],
            'val_losses': []
        }

        self._setup_intermediate_results()

        print(f'Using {self.device} device')

    def _setup_model(self):
        self.model = MSDNet(self.config).to(self.device)

    def _setup_data(self):
        train_dataset = SR4ZCTDataset(
            self.config.dataset_path,
            augmentation=True
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        val_dataset = SR4ZCTDataset(
            self.config.dataset_path,
            augmentation=False
        )

        val_size = min(100, len(val_dataset))
        val_indices = list(range(0, len(val_dataset), len(val_dataset) // val_size))[:val_size]
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)

        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers // 2,
            pin_memory=True
        )

    def _setup_training(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        self.criterion = nn.MSELoss()

    def _setup_intermediate_results(self):
        self.intermediate_dir = self.output_dir / "sr_intermediate"
        self.intermediate_dir.mkdir(exist_ok=True)

        val_dataset = SR4ZCTDataset(self.config.dataset_path, augmentation=False)

        monitor_idx = 10
        self.monitor_input, self.monitor_target = val_dataset[monitor_idx]

        self.monitor_input = self.monitor_input.unsqueeze(0).to(self.device)
        self.monitor_target = self.monitor_target.unsqueeze(0).to(self.device)

        target_img = self.monitor_target[0, 0].cpu().numpy()
        self.monitor_data_range = target_img.max() - target_img.min()
        if self.monitor_data_range == 0:
            self.monitor_data_range = 1.0

        print(f"Intermediate results will be saved to: {self.intermediate_dir}")

    def _save_monitoring_images(self, epoch: int):
        self.model.eval()

        with torch.no_grad():
            prediction = self.model(self.monitor_input)

            input_img = self.monitor_input[0, 0].cpu().numpy()
            target_img = self.monitor_target[0, 0].cpu().numpy()
            pred_img = prediction[0, 0].cpu().numpy()

            vmin, vmax = target_img.min(), target_img.max()

            psnr_input = MetricsCalculator.calculate_psnr(
                input_img, target_img, self.monitor_data_range)
            psnr_pred = MetricsCalculator.calculate_psnr(
                pred_img, target_img, self.monitor_data_range)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(input_img, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0].set_title(f'original')
            axes[0].text(0.02, 0.02, f'psnr:{psnr_input:.2f}dB', 
                        transform=axes[0].transAxes, color='white', fontsize=10)
            axes[0].axis('off')

            axes[1].imshow(pred_img, cmap='gray', vmin=vmin, vmax=vmax)
            axes[1].set_title(f'intermediate result')
            axes[1].text(0.02, 0.02, f'{psnr_pred:.2f}dB', 
                        transform=axes[1].transAxes, color='white', fontsize=10)
            axes[1].axis('off')

            axes[2].imshow(target_img, cmap='gray', vmin=vmin, vmax=vmax)
            axes[2].set_title('ground truth')
            axes[2].axis('off')

            plt.tight_layout()

            plt.savefig(self.intermediate_dir / f'intermediate_epoch_{epoch + 1}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

        self.model.train()

    def _save_intermediate_sample(self, epoch: int):
        try:
            test_data_path = Path(self.config.dataset_path) / 'recon_low_test.npy'
            if test_data_path.exists():
                test_volume = np.load(test_data_path)

                mid_slice_idx = test_volume.shape[1] // 2
                test_slice = test_volume[:, mid_slice_idx, :]

                test_input = torch.from_numpy(test_slice[None, None, ...]).float().to(self.device)

                self.model.eval()
                with torch.no_grad():
                    test_prediction = self.model(test_input)

                pred_slice = test_prediction[0, 0].cpu().numpy()

                try:
                    import tifffile
                    if hasattr(tifffile, 'imwrite'):
                        tifffile.imwrite(
                            self.intermediate_dir / f'test_epoch_{epoch + 1}.tif',
                            pred_slice.astype(np.float32)
                        )
                    else:
                        tifffile.imsave(
                            self.intermediate_dir / f'test_epoch_{epoch + 1}.tif',
                            pred_slice.astype(np.float32)
                        )
                except ImportError:
                    np.save(
                        self.intermediate_dir / f'test_epoch_{epoch + 1}.npy',
                        pred_slice.astype(np.float32)
                    )

                self.model.train()

        except Exception as e:
            print(f"Warning: Could not save intermediate sample: {e}")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Training", leave=False)

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}'
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        self.model.eval()

        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_history': self.train_history
        }

        torch.save(self.model.state_dict(), self.output_dir / "sr.pt")

        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.output_dir / 'best_model.pth')

    def _process_test_volume(self, dataloader, view_type: str) -> np.ndarray:
        self.model.eval()

        num_slices = len(dataloader.dataset)

        sample_input = dataloader.dataset[0]
        if len(sample_input.shape) == 3:
            height, width = sample_input.shape[1], sample_input.shape[2]
        else:
            height, width = sample_input.shape[0], sample_input.shape[1]

        if view_type == "coronal":
            output_volume = np.zeros((height, num_slices, width), dtype=np.float32)
        else:
            output_volume = np.zeros((height, width, num_slices), dtype=np.float32)

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Processing {view_type} slices")
            for i, inputs in enumerate(progress_bar):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                output_slice = outputs[0, 0].cpu().numpy()

                if view_type == "coronal":
                    output_volume[:, i, :] = output_slice
                else:
                    output_volume[:, :, i] = output_slice

        return output_volume

    def generate_final_outputs(self):
        print("Generating final outputs...")

        test_cor_dataset = TestCoronalDataset(self.config.dataset_path)
        test_sag_dataset = TestSagittalDataset(self.config.dataset_path)

        test_cor_loader = DataLoader(
            test_cor_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers // 2
        )

        test_sag_loader = DataLoader(
            test_sag_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers // 2
        )

        cor_output = self._process_test_volume(test_cor_loader, "coronal")
        cor_output_path = self.output_dir / 'output_cor.npy'
        np.save(cor_output_path, cor_output)
        print(f"Saved coronal output: {cor_output_path}")

        sag_output = self._process_test_volume(test_sag_loader, "sagittal")
        sag_output_path = self.output_dir / 'output_sag.npy'
        np.save(sag_output_path, sag_output)
        print(f"Saved sagittal output: {sag_output_path}")

        return cor_output_path, sag_output_path

    def save_training_plots(self):
        epochs = range(1, len(self.train_history['train_losses']) + 1)

        fig = plt.figure(frameon=True, figsize=(10, 6))
        plt.plot(epochs, self.train_history['train_losses'], '-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.train_history['val_losses'], '-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.output_dir / "train_loss.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig = plt.figure(frameon=True)
        plt.plot(self.train_history['train_losses'], '-')
        plt.xlabel('epoch')
        plt.ylabel('mse loss')
        plt.legend(['Train'])
        plt.title('Train loss')
        fig.savefig(self.output_dir / "train_loss_simple.png")
        plt.close()

    def train(self):
        print("Starting training...")

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            
            train_loss = self.train_epoch(epoch)

            val_loss = self.validate()

            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            self.scheduler.step(val_loss)

            self.train_history['train_losses'].append(train_loss)
            self.train_history['val_losses'].append(val_loss)

            if epoch == 0 or (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)

                self._save_monitoring_images(epoch)

                self._save_intermediate_sample(epoch)

            torch.save(self.model.state_dict(), self.output_dir / "sr.pt")

        print("Training completed.")

        self.save_training_plots()

        self.generate_final_outputs()


def main():
    if torch.cuda.is_available():
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    config = TrainingConfig(
        num_layers=100,
        growth_rate=1,
        dilation_rates=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

        batch_size=1,
        learning_rate=0.001,
        num_epochs=100,

        dataset_path="sr4zct_exp_dataset",
        output_dir="msdnet_training_results",

        num_workers=16
    )

    trainer = MSDNetTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()