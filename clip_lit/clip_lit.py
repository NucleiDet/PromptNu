import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
from collections import OrderedDict
from torch.utils.data import DataLoader
from clip_lit.dataset import InpaintingDataForPrompt


def sample_data(loader): 
    while True:
        for batch in loader:
            yield batch

def create_loader_prompt(args): 
    dataset = InpaintingDataForPrompt(args)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size_prompt,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    return sample_data(data_loader)

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
#load clip
model, preprocess = clip.load("/root/autodl-tmp/rubyyao/PycharmProjects/nuclei_ins_seg/code/pretrained/RN50.pt", device=torch.device("cpu"))
model.to(device)
for para in model.parameters():
    para.requires_grad = False

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

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

class Prompts(nn.Module):
    def __init__(self,args,initials=None):
        super(Prompts,self).__init__()
        # print("The initial prompts are:",initials)
        self.text_encoder = TextEncoder(model) # 导入TextEncoder
        self.args = args
        if isinstance(initials,list):  # 输入initials类型为list
            text = clip.tokenize(initials).cuda() # 将输入标准化
            # print(text) # 表述为数字列表的形式
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()
            # print(self.embedding_prompt) 归一化？
        elif isinstance(initials,str):  # 输入initials类型为str
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            ### modify, our save model does not have module

            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:] # remove `module.`
            #     new_state_dict[name] = v
            # self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            
            self.embedding_prompt = nn.Parameter(state_dict['embedding_prompt']).cuda()

            self.embedding_prompt.requires_grad = True
        else:
            self.embedding_prompt = torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*args.length_prompt)," ".join(["X"]*args.length_prompt)]).requires_grad_())).cuda()

    def forward(self, tensor, flag=1):
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*self.args.length_prompt)]])
        text_features = self.text_encoder(self.embedding_prompt, tokenized_prompts)
        
        for i in range(tensor.shape[0]):
            image_features = tensor[i]
            text_features_nor = torch.norm(text_features, dim=-1, keepdim=True)
            if flag==0:
                similarity = (100.0 * image_features @ (text_features / text_features_nor).T)#.softmax(dim=-1)
                if(i==0):
                    probs = similarity
                else:
                    probs = torch.cat([probs, similarity], dim=0)
            else:
                similarity = (100.0 * image_features @ (text_features / text_features_nor).T).softmax(dim=-1)#/nor
                if(i==0):
                    probs = similarity[:, 0]
                else:
                    probs = torch.cat([probs, similarity[:, 0]],dim=0)
        return probs
