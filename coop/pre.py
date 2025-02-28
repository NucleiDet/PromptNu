import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch
from torch import nn
# import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

from itertools import cycle
from copy import deepcopy
# from transformers import AdamW
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from sklearn.metrics import matthews_corrcoef, f1_score

_tokenizer = _Tokenizer()

def positional_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

class PromptEncoder(torch.nn.Module):
    def __init__(self,
                 module_type='LSTM1',
                 emb_dimension=512,
                 bottleneck_size=64,
                 nonlinearity='relu', # activation function
                 layer_norm=True,
                 dropout=0.0,
                 residual=True,
                 separate_network=False,
                 ):
        """ Soft prompt re-parameterization encoder to process prompt embeddings. Prompt Encoder can have a Residual connection.
        Args:
            module_type (str, optional): Type of network to be used.
                Currently supports 1-layer and 2-layerLSTM and MLPs, and simple transformer encoders.
                Defaults to 'LSTM1'.
            emb_dimension (int, optional): Dimension of CLIP model embeddings. Defaults to 512.
            bottleneck_size (int): Dimension of the MLP bottleneck.
            residual (bool, optional): Whether to use residual connection in Prompt Encoder. Defaults to True.
        """
        super().__init__()
        assert module_type in ['LSTM1', 'LSTM2', 'MLP1', 'MLP2', 'transformer']
        assert nonlinearity in ['relu', 'tanh', 'sigm']
        self.module_type = module_type

        if module_type in ['LSTM1', 'LSTM2']:
            self.lstm_head = torch.nn.LSTM(input_size=emb_dimension,
                                           hidden_size=emb_dimension // 2,
                                           num_layers=1 if module_type=='LSTM1' else 2,
                                           dropout=0.05,
                                           bidirectional=True,
                                           batch_first=True)

        elif module_type in ['MLP1', 'MLP2']:
            layers = [nn.Linear(emb_dimension, bottleneck_size)]

            if nonlinearity=='relu':
                layers.append(nn.ReLU())
            elif nonlinearity=='tanh':
                layers.append(nn.Tanh())
            elif nonlinearity=='sigm':
                layers.append(nn.Sigmoid())

            layers.append(nn.Linear(bottleneck_size, emb_dimension))

            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            if layer_norm:
                layers.append(nn.LayerNorm(emb_dimension))

            if module_type=='MLP2':
                layers = layers + layers # repeat twice
            self.module = torch.nn.Sequential(*layers)

        elif module_type=='transformer':
            self.pos_encodings = torch.FloatTensor(positional_encoding(max_length, emb_dimension))
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).cuda()
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).cuda()
            
        self.separate_network = separate_network
        self.residual = residual
        if self.residual:
            print('Using skip connection in Prompt Encoder')
        else:
            print('Not Using skip connection in Prompt Encoder')

    def forward(self, inputs):
        if self.module_type in ['LSTM1', 'LSTM2']:
            if self.separate_network:
                output_embeds = self.lstm_head(inputs)[0].clone()
            else:
                output_embeds = self.lstm_head(inputs)[0].clone().squeeze()
            if self.residual:
                output_embeds += inputs
            return output_embeds
        elif self.module_type=='transformer':
            inputs = inputs + self.pos_encodings

        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)


# Initialize new task prompt from random vocab. tokens
def init_new_prompt(self, prompt_len, random_init=False):
    if self.prefix_path==None:
        model = self.model
        N = model.encoder.embed_tokens.weight.shape[0]
        prompt_weigths = []

        # init from random uniform
        if random_init:
            r1, r2 = -0.5, 0.5
            x = (r1 - r2) * torch.rand(prompt_len, N) + r2
            prompt_weigths = x.numpy()
        else: # init from random tokens
            for i in range(prompt_len):
                with torch.no_grad():
                    j = np.random.randint(N) # random token
                    w = deepcopy(model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
                    prompt_weigths.append(w)
        prompt_weigths = np.array(prompt_weigths)

    else: # initializing from existing path
        prompt_weigths = np.load(self.prefix_path)
    return prompt_weigths


def load_clip_to_cpu_np(model_path = "./pretained_pth/RN50.pt"):
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def load_clip_to_cpu_ns(model_path = "./pretained_pth/RN50.pt"):
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def load_clip_to_cpu_nc(model_path = "./pretained_pth/RN50.pt"):
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, 
                 bottleneck_size=64, # bottleneck size in case of using MLP reparametrization
                 dropout=0,
                 layer_norm=False
                 ):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 8 # number of context vectors
        ctx_init = ""  # initialization words
        encoder_network = 'LSTM2'
        residual = True
        separate_network = False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            # if cfg.TRAINER.PRE.CSC:
            #     print("Initializing class-specific contexts")
            #     ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            # else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        if separate_network: # separate network for each prompt token in prompt encoder
            print('Creating a dictionary of networks in prompt encoder')
            self.encoder_network = {}
            for i in range(n_ctx):
                self.encoder_network[i] = PromptEncoder(
                                        module_type=encoder_network,
                                        bottleneck_size=bottleneck_size,
                                        dropout=dropout,
                                        emb_dimension=ctx_dim,
                                        nonlinearity='relu', # activation function
                                        layer_norm=layer_norm,
                                        residual=residual,
                                        separate_network=separate_network,
                                        )
            for i in list(self.encoder_network):
                self.encoder_network[i].cuda()

        else: # 1 shared network
            print(len(encoder_network))
            self.encoder_network = PromptEncoder(
                                            module_type=encoder_network,
                                            bottleneck_size=bottleneck_size,
                                            dropout=dropout,
                                            emb_dimension=ctx_dim,
                                            nonlinearity='relu', # activation function
                                            layer_norm=layer_norm,
                                            residual=residual,
                                            separate_network=separate_network,
                                            )
            # self.encoder_network.cuda()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.separate_network = separate_network
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    # passing each prompt token through a separate network
    def pass_separate_network(self):
        x = self.ctx
        out = []
        for i in range(self.n_ctx):
            # self.encoder_network[i] = self.encoder_network[i].cuda()
            self.encoder_network[i] = self.encoder_network[i]
            h = self.encoder_network[i](x[i:i+1])
            out.append(h)
        out = torch.concat(out)
        return out

    def forward(self):
        if self.separate_network:
            ctx = self.pass_separate_network()
        else:
            ctx = self.encoder_network(self.ctx)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )      

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
class Adapter_np(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter_np, self).__init__()
        self.fc_np = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc_np(x)
        return x

class CustomCLIP_np(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter_np = Adapter_np(1024, 4).to(clip_model.dtype)

    def forward(self, image_features):
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, 1024)
        # CLIP-Adapter
        # ratio = 0.2
        ratio = 0.5
        x = self.adapter_np(image_features)
        image_features = ratio * x + (1 - ratio) * image_features

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        out = logits.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        return out

class Adapter_ns(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter_ns, self).__init__()
        self.fc_ns = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc_ns(x)
        return x 
    
class CustomCLIP_ns(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter_ns = Adapter_ns(1024, 4).to(clip_model.dtype)

    def forward(self, image_features):
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, 1024)
        # CLIP-Adapter
        # ratio = 0.2
        ratio = 0.5
        x = self.adapter_ns(image_features)
        image_features = ratio * x + (1 - ratio) * image_features

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        out = logits.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        return out

class Adapter_nc(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter_nc, self).__init__()
        self.fc_nc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc_nc(x)
        return x

class CustomCLIP_nc(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter_nc = Adapter_nc(1024, 4).to(clip_model.dtype)

    def forward(self, image_features):
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, 1024)
        # CLIP-Adapter
        # ratio = 0.2
        ratio = 0.5
        x = self.adapter_nc(image_features)
        image_features = ratio * x + (1 - ratio) * image_features

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        out = logits.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        return out

