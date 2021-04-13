import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
from copy import deepcopy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TransformerEncoder(nn.Module):
    def __init__(self, image_size: int = 36, hidden_size: int = 768,
                 num_head: int = 12, attention_size: int = 768,
                 num_mlp_layers: int = 2, num_encoder_layers: int = 12,
                 dropout: float = .2, num_classes: int = 10, div_term: int = 3):
        super(TransformerEncoder, self).__init__()
        self.model_name = "ViT AKA image transformer"

        self.image_size = image_size
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.attention_size = attention_size
        self.dropout = dropout
        self.num_mlp_layers = num_mlp_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes

        # check to see if we can break image to N tiles
        self.N = int(math.pow(div_term, 2))  # N aka number of tiles
        assert image_size % self.N == 0
        self.num_input_features = int(math.pow(image_size / div_term, 2)) * 3

        # Define embedding linear projection
        self.embedding = nn.Linear(self.num_input_features, hidden_size)
        # Define learnable positional embeddings of size (N+1), hidden_size
        self.pos_embedding = nn.Parameter(torch.zeros((1, self.N + 1, hidden_size)))
        # define class learnable class token
        self.cls_token = nn.Parameter(torch.zeros((1, 1, hidden_size)))

        # define encoder layers
        layer = TransformerEncoderLayer(input_size=self.num_input_features, hidden_size=hidden_size,
                                        num_head=num_head, attention_size=attention_size,
                                        num_mlp_layers=num_mlp_layers, N=self.N, dropout=dropout).to(device)

        encoder_layers = []
        for _ in range(num_encoder_layers):
            encoder_layers.append(deepcopy(layer))

        self.encoder_layers = nn.Sequential(*encoder_layers)

        # define final MLP head for classification
        self.mlp = MLP(hidden_size=hidden_size, num_layers=num_mlp_layers, out_size=num_classes, dropout=dropout).to(
            device)

    def forward(self, x):
        """

        :param x: image tensor of size (B, C, H, W)
        :return: prediction for each class of size (B, num_classes)
        """

        B = x.size(0)
        x = x.view(B, self.N, -1)
        cls_token = self.cls_token.expand(B, 1, self.hidden_size)

        # apply embedding to image
        x = self.embedding(x)
        # cat class token
        x = torch.cat([cls_token, x], dim=1)
        # apply positional embedding
        x = x + self.pos_embedding

        x = self.encoder_layers(x)
        x = self.mlp(x)
        # we only going to need class token output
        x = x[:, 1, :]  # x is of size (B, 1, num_classes)

        return x.view(B, -1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_size: int = 363, hidden_size: int = 768,
                 num_head: int = 12, attention_size: int = 768,
                 num_mlp_layers: int = 2, N: int = 9, dropout: float = .2):
        super(TransformerEncoderLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.attention_size = attention_size
        self.dropout = dropout
        self.num_mlp_layers = num_mlp_layers
        self.N = N

        self.multi_head_attention = Multi_Head_Attention(hidden_size=hidden_size, num_head=num_head,
                                                         attention_size=hidden_size, dropout=dropout).to(device)
        self.mlp = MLP(hidden_size=hidden_size, num_layers=num_mlp_layers, out_size=hidden_size, dropout=dropout).to(
            device)

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        """

        :param x: encoder layer input of size (B, N, hidden_size)
        :return:
        """

        # attention part
        x1 = x
        x1 = self.norm1(x1)
        x1, _ = self.multi_head_attention(x1)
        x = x + x1

        # mlp part
        x1 = x
        x1 = self.norm2(x1)
        x1 = self.mlp(x1)
        x = x + x1

        return x


class Multi_Head_Attention(nn.Module):
    def __init__(self, hidden_size: int, num_head: int, attention_size: int, dropout: float):
        super(Multi_Head_Attention, self).__init__()
        # check if number if hidden size can be divided number of heads
        assert attention_size % num_head == 0

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.num_head = num_head

        # define Q K V linear
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)

        self.linear_projection = nn.Linear(hidden_size, hidden_size)

        self.attention_dropout = nn.Dropout(p=dropout)
        self.projection_dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        """

        :param x: image batch of size (B, N, hidden_size)
        :return: output of multi-head attention and attention weights
        """
        B, N = x.size()[:2]

        # covert inputs to desired shape
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # reshape q, k and v into multi head shape (B, N, num_heads, rest) then permute dim 1&2

        q = q.view(B, N, self.num_head, -1).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_head, -1).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_head, -1).permute(0, 2, 1, 3)

        weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.attention_size)
        weights = F.softmax(weights, dim=-1)
        weights = self.attention_dropout(weights)

        # calculate new outputs
        new_values = torch.matmul(weights, v)

        # reshape to original size
        new_values = new_values.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        weights = weights.view(B, N, -1)

        output = self.projection_dropout(self.linear_projection(new_values))

        return output, weights


class MLP(nn.Module):
    def __init__(self, hidden_size: int, out_size: int, num_layers: int, dropout: float):
        super(MLP, self).__init__()

        fc_layers = []
        self.num_layers = num_layers

        for _ in range(num_layers - 1):
            fc_layers.append(nn.Linear(hidden_size, hidden_size).to(device))
            fc_layers.append(nn.GELU())
            fc_layers.append(nn.Dropout(p=dropout))

        self.fc_layers = nn.Sequential(*fc_layers)

        self.fc_last = nn.Linear(hidden_size, out_size)
        self.drop_last = nn.Dropout(p=dropout)

        self.init_weights()

    def init_weights(self):
        for module in self.fc_layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.normal_(module.bias, std=1e-6)

        nn.init.xavier_uniform_(self.fc_last.weight)
        nn.init.normal_(self.fc_last.bias, std=1e-6)

    def forward(self, x: Tensor):
        """

        :param x: output of multi-head attention of size (B, N, hidden_size)
        :return:
        """

        x = self.fc_layers(x)
        x = self.fc_last(x)
        x = self.drop_last(x)
        return x
