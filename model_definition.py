from typing import Callable

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

    def forward(self, x):
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
        

class AutoEncoder(nn.Module):
    """
    Takes an encoder and a decoder and applies them sequentially.
    """
    def __init__(self, encoder: FCEncoder, decoder: FCEncoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
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
            num_experts: int
        ):
        super().__init__()
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.router_args = router_args
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
            self.autoencoders.append(AutoEncoder(encoder, decoder))
        
        self.router = Router(router_args)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        routing_logits = self.router(x)
        routing_probs = torch.softmax(routing_logits, dim=-1)

        # output = torch.zeros_like(x)
        # for i in range(self.num_experts):
        #     expert_output = self.autoencoders[i](x)
        #     output += routing_probs[:, i].unsqueeze(-1) * expert_output
        expert_outputs = [ae(x) for ae in self.autoencoders]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        x_hat = (expert_outputs * routing_probs.unsqueeze(-1)).sum(dim=1)
        return x_hat, routing_logits, expert_outputs

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



class RoutedAutoEncoderModule(LightningModule):
    def __init__(
        self,
        *,
        inner: RoutedAutoEncoder,
        loss_fn: str,
        router_loss_weight: float,
        learning_rate: float,
        on_val_epoch_end_clb: Callable
    ):
        super().__init__()
        self.inner = inner
        self.loss_fn = self.get_loss_fn(loss_fn)
        self.router_loss = RouterLoss()
        self.router_loss_weight = router_loss_weight
        self.learning_rate = learning_rate
        self.on_val_epoch_end_clb = on_val_epoch_end_clb


    @classmethod
    def get_loss_fn(cls, loss_fn: str):
        if loss_fn == "mse":
            return nn.MSELoss()
        else:
            raise ValueError(f"Loss function {loss_fn} is not supported")

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, = batch
        x_hat, routing_logits, expert_outputs = self.inner(x) # (x_hat, routing_logits, expert_outputs)
        return x_hat, routing_logits, expert_outputs
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, = batch
        x_hat, routing_logits, expert_outputs = self.inner(x)
        repro_loss = self.loss_fn(x, x_hat)
        router_loss = self.router_loss(routing_logits)
        loss = repro_loss + self.router_loss_weight * router_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("router_loss", router_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, = batch
        x_hat, routing_logits, expert_outputs = self.inner(x)
        repro_loss = self.loss_fn(x, x_hat)
        router_loss = self.router_loss(routing_logits)
        loss = repro_loss + self.router_loss_weight * router_loss
        self.log("val_loss", loss)
        self.log("val_repro_loss", repro_loss)
        self.log("val_router_loss", router_loss)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        print()
        print("Validation epoch end")
        print(f"val_loss: {self.trainer.callback_metrics['val_loss']}")
        print(f"val_repro_loss: {self.trainer.callback_metrics['val_repro_loss']}")
        print(f"val_router_loss: {self.trainer.callback_metrics['val_router_loss']}")
        print()
        self.on_val_epoch_end_clb(self.trainer, self)

    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x, = batch
        x_hat, routing_logits, expert_outputs = self.inner(x)
        repro_loss = self.loss_fn(x, x_hat)
        router_loss = self.router_loss(routing_logits)
        loss = repro_loss + self.router_loss_weight * router_loss
        self.log("test_loss", loss)
        self.log("test_repro_loss", repro_loss)
        self.log("test_router_loss", router_loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

