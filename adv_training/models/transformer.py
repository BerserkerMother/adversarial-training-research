import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet

import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TransformerEncoder(nn.Module):
    def __init__(self, image_size: int = 36, hidden_size: int = 516,
                 num_head: int = 6, attention_size: int = 516,
                 num_encoder_layers: int = 6, dropout: float = .4,
                 num_classes: int = 10, patch_size: int = 16):
        super(TransformerEncoder, self).__init__()
        self.model_name = "ViT AKA image transformer"

        self.image_size = image_size
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.attention_size = attention_size
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes
        self.div_term = int(image_size / patch_size)

        # check to see if we can break image to N tiles
        self.N = int(math.pow(self.div_term, 2))  # N aka number of tiles
        assert image_size % patch_size == 0
        self.num_input_features = int(math.pow(image_size / self.div_term, 2)) * 3

        self.embedding = Embedding(image_size=image_size, hidden_size=hidden_size,
                                   div_term=self.div_term, dropout=dropout)

        Blocks = []
        for _ in range(num_encoder_layers):
            # define encoder layers
            Block = TransformerEncoderLayer(input_size=self.num_input_features, hidden_size=hidden_size,
                                            num_head=num_head, attention_size=attention_size,
                                            N=self.N, dropout=dropout).to(device)
            Blocks.append(Block)

        self.encoder_layers = nn.ModuleList(Blocks)
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        # define final MLP head for classification
        self.mlp = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """

        :param x: image tensor of size (B, C, H, W)
        :return: prediction for each class of size (B, num_classes) & attention weights of all layers
        """

        # apply embedding to image
        x = self.embedding(x)

        attn_weights = []
        for block in self.encoder_layers:
            x, attn_weight = block(x)
            attn_weights.append(attn_weight)

        x = self.encoder_norm(x)
        # we only going to need class token output
        x = self.mlp(x[:, 0])

        return x, attn_weights


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
        self.mlp = MLP(hidden_size=hidden_size, out_size=hidden_size, dropout=dropout).to(
            device)

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        """

        :param x: tuple of encoder layer input of size (B, N, hidden_size) & list of attention weights
        :return:
        """

        # attention part
        x1 = x
        x1 = self.norm1(x1)
        x1, weights = self.multi_head_attention(x1)
        x = x + x1

        # mlp part
        x1 = x
        x1 = self.norm2(x1)
        x1 = self.mlp(x1)
        x = x + x1

        return x, weights


class Embedding(nn.Module):
    def __init__(self, image_size: int = 36, hidden_size: int = 768, div_term: int = 16, dropout: float = .2):
        """contain of patch and positional embedding, also adds cls token to input"""
        super(Embedding, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.div_term = div_term
        self.patch_size = int(image_size / div_term)
        self.N = div_term * div_term

        assert image_size % div_term == 0, "can't divide picture by div term"

        # use a convolution layer for patch embedding
        self.patch_embedding = nn.Conv2d(3, hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros((1, self.N + 1, hidden_size)))
        # cls token
        self.cls_token = nn.Parameter(torch.zeros((1, 1, hidden_size)))

        self.embedding_dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: tensor of shape (B, C, H, W)
        :return: tensor of shape(B, N, P*P*C)
        """
        B = x.size(0)

        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        cls_token = self.cls_token.expand(B, 1, self.hidden_size)
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embedding

        x = self.embedding_dropout(x)

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

        # self.init_weights()

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
        attention_probs = self.attention_dropout(weights)

        # calculate new outputs
        new_values = torch.matmul(attention_probs, v)

        # reshape to original size
        new_values = new_values.permute(0, 2, 1, 3).contiguous().view(B, N, -1)

        output = self.projection_dropout(self.linear_projection(new_values))

        return output, weights

    def init_weights(self):
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.linear_projection.weight)

        nn.init.normal_(self.Q.bias, std=1e-6)
        nn.init.normal_(self.K.bias, std=1e-6)
        nn.init.normal_(self.V.bias, std=1e-6)
        nn.init.normal_(self.linear_projection.bias, std=1e-6)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, out_size: int, dropout: float):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(4 * hidden_size, out_size)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: Tensor):
        """

        :param x: output of multi-head attention of size (B, N, hidden_size)
        :return:
        """

        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
