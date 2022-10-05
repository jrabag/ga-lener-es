import math
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch.nn.modules.loss import _Loss as Loss

# https://github.com/pytorch/examples/blob/main/word_language_model/main.py
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(pl.LightningModule):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self,
        ntoken: int,
        ninp: int,
        seq_len: int,
        nout: int,
        nhead: int,
        nhid: int,
        nlayers: int,
        loss_fn: "Loss",
        dropout=0.5,
    ):
        """_summary_

        Args:
            ntoken (_type_): vocab size
            ninp (_type_): Embedding size
            nhead (_type_): Number of heads
            nhid (_type_):  the dimension of the feedforward enconder layer
            nlayers (_type_): Number of layers
            dropout (float, optional): _description_. Defaults to 0.5.

        Raises:
            ImportError: _description_
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError(
                "TransformerEncoder module does not exist in PyTorch 1.1 or lower."
            )
        self.model_type = "Transformer"
        self.loss_fn = loss_fn
        self.src_mask = None
        self.ninp = ninp
        self.encoder = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = nn.Linear(ninp, nout)
        # self.decoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Sigmoid(),
        #     nn.Linear(ninp, nout)
        # )
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output: "torch.Tensor" = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output.mean(dim=1))
        # output = self.decoder(output.view(output.shape[1], -1))
        # output = self.decoder(output)
        return F.softmax(output, dim=-1)

    def _eval_metrics(self, output, target, mode="train"):
        loss = self.loss_fn(output, target)
        acc = (2 * (output * target) / (output + target)).sum(dim=-1).mean()
        # acc = accuracy(output, target)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True)
        self.log(f"{mode}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self._eval_metrics(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self._eval_metrics(y_hat, y, mode="test")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self._eval_metrics(y_hat, y, mode="val")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self(batch)
        # _, indices = torch.topk(y_hat, 5, -1)
        return y_hat
