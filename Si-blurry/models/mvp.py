from typing import TypeVar, Iterable
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

class MVP(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (0,1),
                 len_g_prompt   : int   = 5,
                 pos_e_prompt   : Iterable[int] = (2,3,4),
                 len_e_prompt   : int   = 20,
                 selection_size : int   = 1,
                 prompt_func    : str   = 'prompt_tuning',
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 lambd          : float = 1.0,
                 use_mask       : bool  = True,
                 use_contrastiv : bool  = False,
                 use_last_layer : bool  = True,
                 backbone_name  : str   = 'vit_base_patch16_224',
                 **kwargs):

        super().__init__()
        
        self.features = torch.empty(0)
        self.keys     = torch.empty(0)

        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        self.lambd       = lambd
        self.class_num   = num_classes
        self.task_num    = task_num
        self.use_mask    = use_mask
        self.use_contrastiv  = use_contrastiv
        self.use_last_layer  = use_last_layer
        self.selection_size  = selection_size

        self.add_module('backbone', timm.models.create_model(backbone_name, pretrained=True, num_classes=num_classes,
                                                             drop_rate=0.,drop_path_rate=0.,drop_block_rate=None))
        for name, param in self.backbone.named_parameters():
                param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad   = True

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.zeros(1))
        
        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0
        g_pool = 1
        e_pool = task_num

        self.register_buffer('count', torch.zeros(e_pool))
        self.key     = nn.Parameter(torch.randn(e_pool, self.backbone.embed_dim))
        self.mask    = nn.Parameter(torch.zeros(e_pool, self.class_num) - 1)

        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_size = 1 * self.g_length * self.len_g_prompt
            self.e_size = 1 * self.e_length * self.len_e_prompt
            self.g_prompts = nn.Parameter(torch.randn(g_pool, self.g_size, self.backbone.embed_dim))
            self.e_prompts = nn.Parameter(torch.randn(e_pool, self.e_size, self.backbone.embed_dim))

        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_size = 2 * self.g_length * self.len_g_prompt
            self.e_size = 2 * self.e_length * self.len_e_prompt
            self.g_prompts = nn.Parameter(torch.randn(g_pool, self.g_size, self.backbone.embed_dim))
            self.e_prompts = nn.Parameter(torch.randn(e_pool, self.e_size, self.backbone.embed_dim))

        self.exposed_classes = 0
    
    @torch.no_grad()
    def set_exposed_classes(self, classes):
        len_classes = self.exposed_classes
        self.exposed_classes = len(classes)
    
    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, -1, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, -1, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e]), dim = 1)
            x = block(x)
            x = x[:, :N, :]
        return x
    
    def prefix_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, -1, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, -1, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat((xk, g_prompt[:, pos_g * 2 + 0].clone()), dim = 1)
                xv = torch.cat((xv, g_prompt[:, pos_g * 2 + 1].clone()), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat((xk, e_prompt[:, pos_e * 2 + 0].clone()), dim = 1)
                xv = torch.cat((xv, e_prompt[:, pos_e * 2 + 1].clone()), dim = 1)
            
            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias
            
            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path1(block.ls1(attention))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def forward_features(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = x.clone()
            for n, block in enumerate(self.backbone.blocks):
                if n == len(self.backbone.blocks) - 1 and not self.use_last_layer: break
                query = block(query)
            query = query[:, 0]
        if self.training:
            self.features = torch.cat((self.features, query.detach().cpu()), dim = 0)

        distance = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        if self.use_contrastiv:
            mass = (self.count + 1)
        else:
            mass = 1.
        scaled_distance = distance * mass
        topk = scaled_distance.topk(self.selection_size, dim=1, largest=False)[1]
        distance = distance[torch.arange(topk.size(0), device=topk.device).unsqueeze(1).repeat(1,self.selection_size), topk].squeeze().clone()
        e_prompts = self.e_prompts[topk].squeeze().clone()
        mask = self.mask[topk].mean(1).squeeze().clone()
        
        if self.use_contrastiv:
            key_wise_distance = 1 - F.cosine_similarity(self.key.unsqueeze(1), self.key, dim=-1)
            self.similarity_loss = -((key_wise_distance[topk] / mass[topk]).exp().mean() / ((distance / mass[topk]).exp().mean() + (key_wise_distance[topk] / mass[topk]).exp().mean()) + 1e-6).log()
        else:
            self.similarity_loss = distance.mean()

        g_prompts = self.g_prompts[0].repeat(B, 1, 1)
        if self.training:
            with torch.no_grad():
                num = topk.view(-1).bincount(minlength=self.e_prompts.size(0))
                self.count += num

        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_prompts, e_prompts)
        feature = self.backbone.norm(x)[:, 0]
        mask = torch.sigmoid(mask)*2.
        return feature, mask
    
    def forward_head(self, feature : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.fc_norm(feature)
        x = self.backbone.head(x)
        # if self.use_mask:
        #     x = x * mask
        return x
    
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x, mask = self.forward_features(inputs, **kwargs)
        x = self.forward_head(x, **kwargs)
        if self.use_mask:
            x = x * mask
        return x

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target) + self.similarity_loss

    def get_similarity_loss(self):
        return self.similarity_loss

    def get_count(self):
        return self.prompt.update()