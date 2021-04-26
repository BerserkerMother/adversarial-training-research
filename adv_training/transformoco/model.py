import torch
import torch.nn as nn
import torch.nn.functional

import math

from adv_training.models.transformer import Embedding, TransformerEncoderLayer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TransformerEncoder(nn.Module):
    def __init__(self, image_size: int = 33, hidden_size: int = 768,
                 num_head: int = 12, attention_size: int = 768,
                 num_encoder_layers: int = 12, dropout: float = .0,
                 num_classes: int = 10, patch_size: int = 11):
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
        self.mlp1 = nn.Linear(hidden_size, num_classes // 2)
        self.mlp2 = nn.Linear((self.N + 1) * num_classes // 2, num_classes)

    def forward(self, x):
        """

        :param x: image tensor of size (B, C, H, W)
        :return: prediction for each class of size (B, num_classes) & attention weights of all layers
        """

        B = x.size(0)

        # apply embedding to image
        x = self.embedding(x)

        for block in self.encoder_layers:
            x, _ = block(x)

        x = self.encoder_norm(x)
        # we only going to need class token output
        x = nn.functional.gelu(self.mlp1(x))
        x = x.view(B, -1)
        x = self.mlp2(x)

        return x


class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=True):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = TransformerEncoder(num_classes=dim)
        self.encoder_k = TransformerEncoder(num_classes=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss
