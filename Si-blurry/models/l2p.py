from typing import TypeVar
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs,_create_vision_transformer

logger = logging.getLogger()

T = TypeVar('T', bound = 'nn.Module')

# default_cfgs['vit_base_patch16_224_l2p'] = _cfg(
#         url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
#         num_classes=21843)

# # Register the backbone model to timm
# @register_model
# def vit_base_patch16_224_l2p(pretrained=False, **kwargs):
#     """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
#     """
#     model_kwargs = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224_l2p', pretrained=pretrained, **model_kwargs)
#     return model

class Prompt(nn.Module):
    def __init__(self,
                 pool_size            : int,
                 selection_size       : int,
                 prompt_len           : int,
                 dimention            : int,
                 _diversed_selection  : bool = False,
                 _batchwise_selection : bool = False,
                 **kwargs):
        super().__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention
        self._diversed_selection  = _diversed_selection
        self._batchwise_selection = _batchwise_selection

        self.key     = nn.Parameter(torch.randn(pool_size, dimention, requires_grad= True))
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad= True))
        
        torch.nn.init.uniform_(self.key,     -1, 1)
        torch.nn.init.uniform_(self.prompts, -1, 1)

        self.register_buffer('frequency', torch.ones (pool_size))
        self.register_buffer('counter',   torch.zeros(pool_size))
    
    def forward(self, query : torch.Tensor, **kwargs):

        B, D = query.shape
        assert D == self.dimention, f'Query dimention {D} does not match prompt dimention {self.dimention}'
        # Select prompts
        match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        if self.training and self._diversed_selection:
            topk = match * F.normalize(self.frequency, p=1, dim=-1)
        else:
            topk = match
        _ ,topk = topk.topk(self.selection_size, dim=-1, largest=False, sorted=True)
        # Batch-wise prompt selection
        if self._batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _,  mosts  = counts.topk(self.selection_size, largest=True, sorted=True)
            topk = idx[mosts].clone().expand(B, -1)
        # Frequency counter
        self.counter += torch.bincount(topk.reshape(-1).clone(), minlength = self.pool_size)
        # selected prompts
        selection = self.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, self.dimention).clone())
        simmilarity = match.gather(1, topk)
        # get unsimilar prompts also 
        return simmilarity, selection

    def update(self):
        if self.training:
            self.frequency += self.counter
        counter = self.counter.clone()
        self.counter *= 0
        if self.training:
            return self.frequency - 1
        else:
            return counter

class L2P(nn.Module):
    def __init__(self,
                 pool_size      : int   = 10,
                 selection_size : int   = 5,
                 prompt_len     : int   = 5,
                 num_classes    : int   = 100,
                 backbone_name  : str   = 'vit_base_patch16_224',
                 lambd          : float = 0.5,
                 _batchwise_selection  : bool = False,
                 _diversed_selection   : bool = True,
                 **kwargs):

        super().__init__()
        
        self.features = torch.empty(0)
        self.keys     = torch.empty(0)

        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        if pool_size < selection_size:
            raise ValueError('pool_size must be larger than selection_size')

        self.prompt_len     = prompt_len
        self.selection_size = selection_size
        self.lambd          = lambd
        self._batchwise_selection = _batchwise_selection
        self.class_num            = num_classes

        self.add_module('backbone', timm.models.create_model(backbone_name, pretrained=True, num_classes=num_classes,
                                                             drop_rate=0.,drop_path_rate=0.,drop_block_rate=None))
        for name, param in self.backbone.named_parameters():
                param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad   = True

        self.prompt = Prompt(
            pool_size,
            selection_size,
            prompt_len,
            self.backbone.num_features,
            _diversed_selection  = _diversed_selection,
            _batchwise_selection = _batchwise_selection)

        self.register_buffer('simmilarity', torch.zeros(1), persistent=False)
        self.register_buffer('unsimmilarity', torch.zeros(1), persistent=False)
    
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        self.backbone.eval()
        x = self.backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        with torch.no_grad():
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0].clone()
        if self.training:
            self.features = torch.cat((self.features, query.detach().cpu()), dim = 0)
        simmilarity, prompts = self.prompt(query)
        simmilarity = simmilarity.mean()
        prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
        prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
        x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
        x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)
        
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        x = x.mean(dim=1)
        x = self.backbone.fc_norm(x)
        x = self.backbone.head(x)
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target) + self.lambd * self.simmilarity

    def convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task] = 0
        return

    def get_count(self):
        return self.prompt.update()