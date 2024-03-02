import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from model_definition import (FCEncoderConfig, GaussianNoiseConfig, RoutedAutoEncoder,
                              RoutedAutoEncoderModule, RoutedAutoEncoderModuleConfig)


class SyntheticDataModule(LightningDataModule):
    def __init__(
            self,
            num_train_samples: int,
            num_val_samples: int,
            num_test_samples: int,
            batch_size: int,
            function: str,
        ):
        super().__init__()
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.batch_size = batch_size
        self.function = function

        self.train_ds = self.generate_dataset(num_train_samples, function, seed=42, mode="train")
        self.val_ds = self.generate_dataset(num_val_samples, function, seed=43, mode="val")
        self.test_ds = self.generate_dataset(num_test_samples, function, seed=44, mode="test")
        
    def get_input_range(self, function: str) -> tuple[float, float]:
        if function == "circle":
            return 0, 2 * math.pi
        elif function == "spiral":
            return 0, 2 * math.pi * 3
        elif function == "figure8":
            return 0, 2 * math.pi
        else:
            raise ValueError(f"Unknown function {function}")
        
    def map_input_to_output(self, x: torch.Tensor, function: str) -> tuple[torch.Tensor, torch.Tensor]:
        if function == "circle":
            return torch.cos(x), torch.sin(x)
        elif function == "spiral":
            return x * torch.cos(x), x * torch.sin(x)
        elif function == "figure8":
            return torch.sin(x), torch.sin(x) * torch.cos(x)
        else:
            raise ValueError(f"Unknown function {function}")


    def generate_dataset(
            self, 
            num_samples: int,
            function: str,
            seed: int,
            mode: str = "train",
            ) -> TensorDataset:
        range_min, range_max = self.get_input_range(function)
        torch.manual_seed(seed)
        if mode == "train":
            angle = torch.rand(num_samples) * (range_max - range_min) + range_min
        else:
            angle = torch.linspace(range_min, range_max, num_samples)
        x, y = self.map_input_to_output(angle, function)
        samples = torch.stack([x, y], dim=1)
        return TensorDataset(samples)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size)
    
    def __str__(self) -> str:
        return (f"CircleDataModule(train_ds={len(self.train_ds)}, val_ds={len(self.val_ds)}, test_ds={len(self.test_ds)}, batch_size={self.batch_size})")
    

def plot_predictions(
        ds: TensorDataset, 
        predictions_stacked: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_file: Path | None = None,
        title: str | None = None,
        ):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    true_x = ds.tensors[0]  # (num_samples, 2)
    
    pred_x = predictions_stacked[0]  # (num_samples, 2)
    router_logits = predictions_stacked[1]  # (num_samples, num_experts)
    
    pred_expert_nr = router_logits.argmax(dim=1)  # (num_samples,)

    true_x_np = true_x.cpu().numpy()
    pred_x_np = pred_x.cpu().numpy()
    pred_expert_nr_np = pred_expert_nr.cpu().numpy()

    # use expert number to color the points
    ax.scatter(true_x_np[:, 0], true_x_np[:, 1], label="True")

    num_experts = pred_expert_nr_np.max() + 1
    for i in range(num_experts):
        expert_mask = pred_expert_nr_np == i
        ax.scatter(pred_x_np[expert_mask, 0], pred_x_np[expert_mask, 1], label=f"Predicted Expert {i}")

    # draw a line from each true point to its prediction
    max_lines = 100
    step = max(len(true_x_np) // max_lines, 1)
    for i in range(0, len(true_x_np), step):
        ax.plot([true_x_np[i, 0], pred_x_np[i, 0]], [true_x_np[i, 1], pred_x_np[i, 1]], c="black", alpha=0.1)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if target_file is None:
        plt.show()
    else:
        plt.savefig(target_file)
        plt.close(fig)

class VisualizePredictionsCallback(Callback):
    def __init__(self, dm: SyntheticDataModule, target_dir: Path, title: str):
        super().__init__()
        self.dm = dm
        self.target_dir = target_dir
        self.title = title

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_nr = trainer.current_epoch
        target_file = self.target_dir /f"predictions_epoch_{epoch_nr}.png"
        # predictions = trainer.predict(pl_module, self.dm)
        predictions = []
        inputs = []
        for batch in self.dm.test_dataloader():
            batch = tuple(t.to(pl_module.device) for t in batch)
            with torch.no_grad():
                pred = pl_module(batch)
            predictions.append(pred)
            inputs.append(batch[0])

        predictions_stacked = self.stack_predictions(predictions)
        plot_predictions(self.dm.test_ds, predictions_stacked, target_file, title=f"{self.title} - Epoch {epoch_nr}")
        inputs_stacked = torch.cat(inputs, dim=0)
        self.save_predictions(predictions_stacked, inputs_stacked, self.target_dir / f"data_epoch_{epoch_nr}.npz")

    def stack_predictions(self, predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_hat = torch.cat([p[0] for p in predictions], dim=0)
        router_logits = torch.cat([p[1] for p in predictions], dim=0)
        expert_outputs = torch.cat([p[2] for p in predictions], dim=0)
        return x_hat, router_logits, expert_outputs
    
    def save_predictions(
            self, 
            predictions_stacked: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
            inputs_stacked: torch.Tensor,
            target_file: Path):
        x_hat, router_logits, expert_outputs = predictions_stacked
        x_hat_np = x_hat.cpu().numpy()
        router_logits_np = router_logits.cpu().numpy()
        expert_outputs_np = expert_outputs.cpu().numpy()
        inputs_np = inputs_stacked.cpu().numpy()
        np.savez(
            target_file, 
            inputs=inputs_np,
            x_hat=x_hat_np, 
            router_logits=router_logits_np, 
            expert_outputs=expert_outputs_np,
            )
        



def train_model(
        function: str,
        num_experts: int,    
    ):
    target_dir = Path(f"models/{function}-{num_experts}")
    if target_dir.exists():
        print(f"Skipping {target_dir} as it already exists")
        return
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = 128

    dm = SyntheticDataModule(20_000, 1_000, 1_000, batch_size, function)
    print(dm)
    
    encoder_args = FCEncoderConfig(input_dim=2, hidden_dim=100, output_dim=1, num_blocks=3, dropout=0.1, inner_activation="relu", output_activation="tanh")
    decoder_args = FCEncoderConfig(input_dim=1, hidden_dim=100, output_dim=2, num_blocks=3, dropout=0.1, inner_activation="relu", output_activation="linear")
    router_args = FCEncoderConfig(input_dim=2, hidden_dim=100, output_dim=num_experts, num_blocks=3, dropout=0.1, inner_activation="relu", output_activation="linear")
    latent_noise_args = GaussianNoiseConfig(std=0.1, mean=0)
    router_noise_args = GaussianNoiseConfig(std=0.1, mean=0)

    model = RoutedAutoEncoder(
        encoder_args=encoder_args, 
        decoder_args=decoder_args, 
        router_args=router_args, 
        num_experts=num_experts, 
        latent_noise_args=latent_noise_args,
        router_noise_args=router_noise_args,
        )
    
    print(model)
    title = f"{function} - {num_experts} experts"
    viz_clb = VisualizePredictionsCallback(dm, target_dir, title)
    raem_args = RoutedAutoEncoderModuleConfig(
        loss_fn="mse", 
        learning_rate=1e-3, 
        moe_repro_loss_weight=1,
        router_loss_weight=1, 
        top_expert_loss_weight=0, 
        sampled_expert_loss_weight=0,
        scaled_loss_weight=1,
        )
    module = RoutedAutoEncoderModule(inner=model, args=raem_args, on_val_epoch_end_clb=viz_clb.on_validation_epoch_end)
    print(module)

    callbacks=[ModelCheckpoint(dirpath=target_dir, filename="model-{epoch:02d}", save_top_k=-1)]
    trainer = Trainer(max_epochs=5, accelerator='gpu', callbacks=callbacks)
    trainer.fit(module, dm)
    test_result = trainer.test(module, dm)
    # print(test_result)
    
    # predict_result = trainer.predict(module, dm)
    # print(predict_result)
    # plot_predictions(dm.test_ds, predict_result)


if __name__ == "__main__":
    train_model("circle", 8)

    train_model("circle", 1)
    train_model("circle", 2)
    train_model("circle", 4)
    train_model("circle", 8)

    train_model("figure8", 1)
    train_model("figure8", 2)
    train_model("figure8", 4)
    train_model("figure8", 8)

    train_model("spiral", 1)
    train_model("spiral", 2)
    train_model("spiral", 4)
    train_model("spiral", 8)

    