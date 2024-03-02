from typing import Callable, NamedTuple

import torch
import torch.nn as nn
from lightning import LightningModule
from pydantic import BaseModel


class FCBlockConfig(BaseModel):
    input_dim: int
    output_dim: int
    dropout: float
    activation: str

class FCBlock(nn.Module):
    def __init__(
        self,
        args: FCBlockConfig
    ):
        super().__init__()
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.dropout = args.dropout
        self.activation = args.activation

        self.fc = nn.Linear(self.input_dim, self.output_dim)
        self.act = self.get_activation(self.activation)
        if self.dropout > 0:
            self.dropout = nn.Dropout(self.dropout)
        else:
            self.dropout = nn.Identity()

    @classmethod
    def get_activation(cls, activation: str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "linear":
            return nn.Identity()
        else:
            raise ValueError(f"Activation {activation} is not supported")

    def forward(self, x) -> torch.Tensor:
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
    
class FCEncoderConfig(BaseModel):
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_blocks: int
    dropout: float
    inner_activation: str
    output_activation: str

    def get_input_config(self) -> FCBlockConfig:
        return FCBlockConfig(
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=self.inner_activation,
        )
    
    def get_output_config(self) -> FCBlockConfig:
        return FCBlockConfig(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
            activation=self.output_activation,
        )
    
    def get_hidden_config(self) -> FCBlockConfig:
        return FCBlockConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=self.inner_activation,
        )
    
    def get_configs(self) -> list[FCBlockConfig]:
        return [
            self.get_input_config(),
            *([self.get_hidden_config()] * self.num_blocks),
            self.get_output_config(),
        ]

class FCEncoder(nn.Module):
    def __init__(
        self,
        args: FCEncoderConfig
    ):
        super().__init__()
        self.args = args

        self.blocks = nn.ModuleList()
        for block_args in args.get_configs():
            self.blocks.append(FCBlock(block_args))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        for block in self.blocks:
            x = block(x)
        return x




class Router(nn.Module):
    """
    Router is a module that takes in a sequence of input vectors and
    outputs a sequence of routing probabilities. The routing probabilities
    are used to compute a weighted sum of the input vectors. 
    """
    def __init__(
        self,
        args: FCEncoderConfig
    ):
        super().__init__()
        self.args = args

        self.blocks = nn.ModuleList()
        for block_args in args.get_configs():
            self.blocks.append(FCBlock(block_args))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, num_route)
        """
        for block in self.blocks:
            x = block(x)

        return x
    

class GaussianNoiseConfig(BaseModel):
    mean: float
    std: float
        
class GaussianNoiseLayer(nn.Module):
    """
    Adds Gaussian noise to the input during training.
    """

    def __init__(self, args: GaussianNoiseConfig):
        super().__init__()
        self.mean = args.mean
        self.std = args.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        else:
            return x
        

AutoEncoderOutput = NamedTuple("AutoEncoderOutput", [("encoder_output", torch.Tensor), ("decoder_output", torch.Tensor)])

class AutoEncoder(nn.Module):
    """
    Takes an encoder and a decoder and applies them sequentially.
    """
    def __init__(self, encoder: FCEncoder, decoder: FCEncoder, noise: GaussianNoiseLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise = noise

        # check dimension compatibility
        if encoder.args.output_dim != decoder.args.input_dim:
            raise ValueError(
                f"Encoder output dimension {encoder.args.output_dim} "
                f"does not match decoder input dimension {decoder.args.input_dim}"
            )
        if encoder.args.input_dim != decoder.args.output_dim:
            raise ValueError(
                f"Encoder input dimension {encoder.args.input_dim} "
                f"does not match decoder output dimension {decoder.args.output_dim}"
            )
        
    def forward(self, x: torch.Tensor) -> AutoEncoderOutput:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        enc_cout = self.encoder(x)
        x = self.noise(enc_cout)
        dec_out = self.decoder(x)
        return AutoEncoderOutput(enc_cout, dec_out)
    

RoutedAutoEncoderOutput = NamedTuple("RoutedAutoEncoderOutput", [
    ("reconstruction", torch.Tensor),
    ("routing_logits", torch.Tensor),
    ("expert_outputs", torch.Tensor),
])

class RoutedAutoEncoder(nn.Module):
    """
    Composes a list of autoencoders with a router.
    """
    def __init__(
            self,
            *,
            encoder_args: FCEncoderConfig,
            decoder_args: FCEncoderConfig,
            router_args: FCEncoderConfig,
            latent_noise_args: GaussianNoiseConfig,
            router_noise_args: GaussianNoiseConfig,
            num_experts: int
        ):
        super().__init__()
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.router_args = router_args
        self.latent_noise_args = latent_noise_args
        self.router_noise_args = router_noise_args
        self.num_experts = num_experts

        # check dimension compatibility
        if num_experts != router_args.output_dim:
            raise ValueError(
                f"Number of experts {num_experts} does not match router output dimension {router_args.output_dim}"
            )

        self.autoencoders: nn.ModuleList[AutoEncoder] = nn.ModuleList()
        for _ in range(num_experts):
            encoder = FCEncoder(encoder_args)
            decoder = FCEncoder(decoder_args)
            noise = GaussianNoiseLayer(latent_noise_args)
            self.autoencoders.append(AutoEncoder(encoder, decoder, noise))
        
        self.router = Router(router_args)
        self.router_noise = GaussianNoiseLayer(router_noise_args)

    def forward(self, x: torch.Tensor) -> RoutedAutoEncoderOutput:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        router_input = self.router_noise(x)
        routing_logits = self.router(router_input)
        routing_probs = torch.softmax(routing_logits, dim=-1)


        ae_outputs: list[AutoEncoderOutput] = [ae(x) for ae in self.autoencoders]
        expert_outputs = torch.stack([aeo.decoder_output for aeo in ae_outputs], dim=1)
        x_hat = (expert_outputs * routing_probs.unsqueeze(-1)).sum(dim=1)
        return RoutedAutoEncoderOutput(x_hat, routing_logits, expert_outputs)

class RouterLoss(nn.Module):
    """
    A loss that encourages the routing probabilities to have zero mean.
    """
    def __init__(self):
        super().__init__()
        self.mean_loss = nn.MSELoss()

    def forward(self, routing_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            routing_logits: (batch_size, num_experts)
        Returns:
            scalar
        """
        mean_logits = routing_logits.mean(dim=0)
        return self.mean_loss(mean_logits, torch.zeros_like(mean_logits))

class RoutedAutoEncoderModuleConfig(BaseModel):
    loss_fn: str
    moe_repro_loss_weight: float
    router_loss_weight: float
    top_expert_loss_weight: float
    sampled_expert_loss_weight: float
    scaled_loss_weight: float
    learning_rate: float

class RoutedAutoEncoderModule(LightningModule):
    def __init__(
        self,
        *,
        inner: RoutedAutoEncoder,
        args: RoutedAutoEncoderModuleConfig,
        on_val_epoch_end_clb: Callable
    ):
        super().__init__()
        self.inner = inner
        self.loss_fn = self.get_loss_fn(args.loss_fn)
        self.router_loss = RouterLoss()
        self.moe_repro_loss_weight = args.moe_repro_loss_weight
        self.router_loss_weight = args.router_loss_weight
        self.top_expert_loss_weight = args.top_expert_loss_weight
        self.sampled_expert_loss_weight = args.sampled_expert_loss_weight
        self.scaled_loss_weight = args.scaled_loss_weight
        self.learning_rate = args.learning_rate
        self.on_val_epoch_end_clb = on_val_epoch_end_clb


    @classmethod
    def get_loss_fn(cls, loss_fn: str):
        if loss_fn == "mse":
            return nn.MSELoss()
        else:
            raise ValueError(f"Loss function {loss_fn} is not supported")

    def forward(self, batch: torch.Tensor) -> RoutedAutoEncoderOutput:
        x, = batch
        rae_out = self.inner(x)
        return rae_out
    
    def get_top_expert_outputs(self, expert_outputs: torch.Tensor, routing_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_outputs: (batch_size, num_experts, output_dim)
            routing_logits: (batch_size, num_experts)
        Returns:
            (batch_size, output_dim)
        """
        top_expert_idx = routing_logits.argmax(dim=1)
        return expert_outputs[torch.arange(expert_outputs.size(0)), top_expert_idx]
    
    def sample_expert_outputs(self, expert_outputs: torch.Tensor, routing_logits: torch.Tensor) -> torch.Tensor:
        """
        Samples the expert to choose based on the router probabilities.
        Args:
            expert_outputs: (batch_size, num_experts, output_dim)
            routing_logits: (batch_size, num_experts)
        Returns:
            (batch_size, output_dim)
        """
        routing_probs = torch.softmax(routing_logits, dim=-1)
        expert_idx = torch.multinomial(routing_probs, 1).squeeze(-1)
        return expert_outputs[torch.arange(expert_outputs.size(0)), expert_idx]
        
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, = batch
        rae_out = self.inner(x)
        moe_repro_loss = self.loss_fn(x, rae_out.reconstruction)
        router_loss = self.router_loss(rae_out.routing_logits)
        
        top_expert_outputs = self.get_top_expert_outputs(rae_out.expert_outputs, rae_out.routing_logits)
        top_expert_loss = self.loss_fn(x, top_expert_outputs)

        sampled_expert_outputs = self.sample_expert_outputs(rae_out.expert_outputs, rae_out.routing_logits)
        sampled_expert_loss = self.loss_fn(x, sampled_expert_outputs)

        # scale individual expert losses by the router probabilities
        router_probs = torch.softmax(rae_out.routing_logits, dim=-1)  # (batch_size, num_experts)
        expert_diffs = x.unsqueeze(1) - rae_out.expert_outputs  # (batch_size, num_experts, output_dim)
        expert_losses = (expert_diffs ** 2).mean(dim=-1)  # (batch_size, num_experts)
        scaled_expert_losses = (router_probs * expert_losses).sum(dim=-1)  # (batch_size,)
        

        loss = moe_repro_loss * self.moe_repro_loss_weight \
            + self.router_loss_weight * router_loss \
            + self.top_expert_loss_weight * top_expert_loss \
            + self.sampled_expert_loss_weight * sampled_expert_loss \
            + self.scaled_loss_weight * scaled_expert_losses.mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("router_loss", router_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("top_expert_loss", top_expert_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, = batch
        rae_out = self.inner(x)
        moe_repro_loss = self.loss_fn(x, rae_out.reconstruction)
        router_loss = self.router_loss(rae_out.routing_logits)
        
        top_expert_outputs = self.get_top_expert_outputs(rae_out.expert_outputs, rae_out.routing_logits)
        top_expert_loss = self.loss_fn(x, top_expert_outputs)

        sampled_expert_outputs = self.sample_expert_outputs(rae_out.expert_outputs, rae_out.routing_logits)
        sampled_expert_loss = self.loss_fn(x, sampled_expert_outputs)

        # scale individual expert losses by the router probabilities
        router_probs = torch.softmax(rae_out.routing_logits, dim=-1)  # (batch_size, num_experts)
        expert_diffs = x.unsqueeze(1) - rae_out.expert_outputs  # (batch_size, num_experts, output_dim)
        expert_losses = (expert_diffs ** 2).mean(dim=-1)  # (batch_size, num_experts)
        scaled_expert_losses = (router_probs * expert_losses).sum(dim=-1)  # (batch_size,)
        

        loss = moe_repro_loss * self.moe_repro_loss_weight \
            + self.router_loss_weight * router_loss \
            + self.top_expert_loss_weight * top_expert_loss \
            + self.sampled_expert_loss_weight * sampled_expert_loss \
            + self.scaled_loss_weight * scaled_expert_losses.mean()

        self.log("val_loss", loss)
        self.log("val_repro_loss", moe_repro_loss)
        self.log("val_router_loss", router_loss)
        self.log("val_top_expert_loss", top_expert_loss)
        self.log("val_sampled_expert_loss", sampled_expert_loss)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        print()
        print("Validation epoch end")
        print(f"val_loss: {self.trainer.callback_metrics['val_loss']}")
        print(f"val_repro_loss: {self.trainer.callback_metrics['val_repro_loss']}")
        print(f"val_router_loss: {self.trainer.callback_metrics['val_router_loss']}")
        print(f"val_top_expert_loss: {self.trainer.callback_metrics['val_top_expert_loss']}")
        print(f"val_sampled_expert_loss: {self.trainer.callback_metrics['val_sampled_expert_loss']}")
        print()
        self.on_val_epoch_end_clb(self.trainer, self)

    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x, = batch
        rae_out = self.inner(x)
        moe_repro_loss = self.loss_fn(x, rae_out.reconstruction)
        router_loss = self.router_loss(rae_out.routing_logits)
        
        top_expert_outputs = self.get_top_expert_outputs(rae_out.expert_outputs, rae_out.routing_logits)
        top_expert_loss = self.loss_fn(x, top_expert_outputs)

        sampled_expert_outputs = self.sample_expert_outputs(rae_out.expert_outputs, rae_out.routing_logits)
        sampled_expert_loss = self.loss_fn(x, sampled_expert_outputs)

        # scale individual expert losses by the router probabilities
        router_probs = torch.softmax(rae_out.routing_logits, dim=-1)  # (batch_size, num_experts)
        expert_diffs = x.unsqueeze(1) - rae_out.expert_outputs  # (batch_size, num_experts, output_dim)
        expert_losses = (expert_diffs ** 2).mean(dim=-1)  # (batch_size, num_experts)
        scaled_expert_losses = (router_probs * expert_losses).sum(dim=-1)  # (batch_size,)
        

        loss = moe_repro_loss * self.moe_repro_loss_weight \
            + self.router_loss_weight * router_loss \
            + self.top_expert_loss_weight * top_expert_loss \
            + self.sampled_expert_loss_weight * sampled_expert_loss \
            + self.scaled_loss_weight * scaled_expert_losses.mean()

        self.log("test_loss", loss)
        self.log("test_repro_loss", moe_repro_loss)
        self.log("test_router_loss", router_loss)
        self.log("test_top_expert_loss", top_expert_loss)
        self.log("test_sampled_expert_loss", sampled_expert_loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

