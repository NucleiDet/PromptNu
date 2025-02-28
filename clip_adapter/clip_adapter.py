import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

# from dassl.engine import TRAINER_REGISTRY, TrainerX
# from dassl.metrics import compute_accuracy
# from dassl.utils import load_pretrained_weights, load_checkpoint
# from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
    'Nuclei': 'a photo of a {}, a type of histopathology.',
}

def load_clip_to_cpu_cc(model_path = "./pretained_pth/RN50.pt"):
    # backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model

def load_clip_to_cpu_tc(model_path = "./pretained_pth/RN50.pt"):
    # backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


def load_clip_to_cpu_nc(model_path="./pretained_pth/RN50.pt"):
    # backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = clip.build_model(state_dict or model.state_dict())

    return model

def load_clip_to_cpu_sc(model_path="./pretained_pth/RN50.pt"):
    # backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = clip.build_model(state_dict or model.state_dict())

    return model

class Adapter_cc(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter_cc, self).__init__()
        self.fc_cc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc_cc(x)
        return x
    
    
class TextEncoder_cc(nn.Module):

    # def __init__(self, cfg, classnames, clip_model):
    def __init__(self, classnames_cc, clip_model_cc):
        super().__init__()
        # self.cfg = cfg
        self.classnames_cc = classnames_cc
        self.clip_model_cc = clip_model_cc
        self.dtype_cc = clip_model_cc.dtype
    
    def forward(self, device):
        # temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        temp = CUSTOM_TEMPLATES['Nuclei']
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames_cc]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)
        text_features = self.clip_model_cc.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP_cc(nn.Module):

    # def __init__(self, cfg, classnames, clip_model):
    def __init__(self, classnames_cc, clip_model_cc):
        super().__init__()
        self.image_encoder_cc = clip_model_cc.visual
        # self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.text_encoder_cc = TextEncoder_cc(classnames_cc, clip_model_cc)
        self.logit_scale_cc = clip_model_cc.logit_scale
        self.dtype_cc = clip_model_cc.dtype
        self.adapter_cc = Adapter_cc(1024, 4).to(clip_model_cc.dtype)

            
    def forward(self, image_features):
        ###Lseg###
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, 1024)
        x = self.adapter_cc(image_features)

        # ori
        # ratio = 0.2
        ratio = 0.4
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder_cc(image_features.device)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale_cc.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        logits = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        ###ori###
        # image_features = self.image_encoder(image.type(self.dtype))
        # x = self.adapter(image_features)
        #
        # ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features
        #
        # text_features = self.text_encoder()
        #
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        ####

        return logits

class Adapter_tc(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter_tc, self).__init__()
        self.fc_tc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc_tc(x)
        return x
    
    
class TextEncoder_tc(nn.Module):

    # def __init__(self, cfg, classnames, clip_model):
    def __init__(self, classnames_tc, clip_model_tc):
        super().__init__()
        # self.cfg = cfg
        self.classnames_tc = classnames_tc
        self.clip_model_tc = clip_model_tc
        self.dtype_tc = clip_model_tc.dtype
    
    def forward(self, device):
        # temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        temp = CUSTOM_TEMPLATES['Nuclei']
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames_tc]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)
        text_features = self.clip_model_tc.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP_tc(nn.Module):

    # def __init__(self, cfg, classnames, clip_model):
    def __init__(self, classnames_tc, clip_model_tc):
        super().__init__()
        self.image_encoder_tc = clip_model_tc.visual
        # self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.text_encoder_tc = TextEncoder_tc(classnames_tc, clip_model_tc)
        self.logit_scale_tc = clip_model_tc.logit_scale
        self.dtype_tc = clip_model_tc.dtype
        self.adapter_tc = Adapter_tc(1024, 4).to(clip_model_tc.dtype)

            
    def forward(self, image_features):
        ###Lseg###
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, 1024)
        x = self.adapter_tc(image_features)

        # ori
        # ratio = 0.2
        ratio = 0.4
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder_tc(image_features.device)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale_tc.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        logits = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        ###ori###
        # image_features = self.image_encoder(image.type(self.dtype))
        # x = self.adapter(image_features)
        #
        # ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features
        #
        # text_features = self.text_encoder()
        #
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        ####

        return logits


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


class TextEncoder_nc(nn.Module):

    # def __init__(self, cfg, classnames, clip_model):
    def __init__(self, classnames_nc, clip_model_nc):
        super().__init__()
        # self.cfg = cfg
        self.classnames_nc = classnames_nc
        self.clip_model_nc = clip_model_nc
        self.dtype_nc = clip_model_nc.dtype

    def forward(self, device):
        # temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        temp = CUSTOM_TEMPLATES['Nuclei']
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames_nc]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)
        text_features = self.clip_model_nc.encode_text(prompts)
        x = text_features
        return x

class CustomCLIP_nc(nn.Module):

    # def __init__(self, cfg, classnames, clip_model):
    def __init__(self, classnames_nc, clip_model_nc):
        super().__init__()
        self.image_encoder_nc = clip_model_nc.visual
        # self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.text_encoder_nc = TextEncoder_nc(classnames_nc, clip_model_nc)
        self.logit_scale_nc = clip_model_nc.logit_scale
        self.dtype_nc = clip_model_nc.dtype
        self.adapter_nc = Adapter_nc(1024, 4).to(clip_model_nc.dtype)

    def forward(self, image_features):
        ###Lseg###
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, 1024)
        x = self.adapter_nc(image_features)
         
        # ori
        # ratio = 0.2
        ratio = 0.4
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder_nc(image_features.device)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale_nc.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        logits = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        ###ori###
        # image_features = self.image_encoder(image.type(self.dtype))
        # x = self.adapter(image_features)
        #
        # ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features
        #
        # text_features = self.text_encoder()
        #
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        ####

        return logits


# @TRAINER_REGISTRY.register()
# class CLIP_Adapter(TrainerX):
#     """ CLIP-Adapter """

#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames

#         print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
#         clip_model = load_clip_to_cpu(cfg)
#         clip_model.float()

#         print('Building custom CLIP')
#         self.model = CustomCLIP(cfg, classnames, clip_model)

#         print('Turning off gradients in both the image and the text encoder')
#         for name, param in self.model.named_parameters():
#             if 'adapter' not in name:
#                 param.requires_grad_(False)

#         if cfg.MODEL.INIT_WEIGHTS:
#             load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        
#         self.model.to(self.device)
#         # NOTE: only give text_encoder.adapter to the optimizer
#         self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
#         self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        

#         self.register_model('clip_adapter', self.model.adapter, self.optim, self.sched)

#         device_count = torch.cuda.device_count()
#         if device_count > 1:
#             print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
#             self.model = nn.DataParallel(self.model)

#     def forward_backward(self, batch):
#         image, label = self.parse_batch_train(batch)
#         output = self.model(image)
#         loss = F.cross_entropy(output, label)
#         self.model_backward_and_update(loss)

#         loss_summary = {
#             'loss': loss.item(),
#             'acc': compute_accuracy(output, label)[0].item()
#         }

#         if (self.batch_idx + 1) == self.num_batches:
#             self.update_lr()

#         return loss_summary

#     def parse_batch_train(self, batch):
#         input = batch['img']
#         label = batch['label']
#         input = input.to(self.device)
#         label = label.to(self.device)
#         return input, label
    
#     def load_model(self, directory, epoch=None):
#         if not directory:
#             print(
#                 'Note that load_model() is skipped as no pretrained model is given'
#             )
#             return

#         names = self.get_model_names()

#         # By default, the best model is loaded
#         model_file = 'model-best.pth.tar'

#         if epoch is not None:
#             model_file = 'model.pth.tar-' + str(epoch)

#         for name in names:
#             model_path = osp.join(directory, name, model_file)

#             if not osp.exists(model_path):
#                 raise FileNotFoundError(
#                     'Model not found at "{}"'.format(model_path)
#                 )

#             checkpoint = load_checkpoint(model_path)
#             state_dict = checkpoint['state_dict']
#             epoch = checkpoint['epoch']
            
#             # Ignore fixed token vectors
#             if 'token_prefix' in state_dict:
#                 del state_dict['token_prefix']
            
#             if 'token_suffix' in state_dict:
#                 del state_dict['token_suffix']

#             print(
#                 'Loading weights to {} '
#                 'from "{}" (epoch = {})'.format(name, model_path, epoch)
#             )
#             # set strict=False
#             self._models[name].load_state_dict(state_dict, strict=False)
