import os
import math
import numpy as np
from glob import glob
import copy
from random import shuffle
from PIL import Image, ImageFilter

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import cv2
import clip


class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        # self.mask_type = args.mask_type
        # image and mask 
        self.args = args
        
        self.image_path = glob(os.path.join(args.dir_image, args.input_type, "*.png"))
        # augmentation 
        self.img_trans = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()])
        self.mask_trans = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        # load image
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])
        HSV_img = image.convert("HSV")
        
        
        gt_image = Image.open(self.image_path[index].replace(self.args.input_type, "target")).convert('RGB') # pair

        highlight_mask_path = self.image_path[index].replace(self.args.input_type, "mask") 
        highlight_mask = Image.open(highlight_mask_path).convert('L')

        # augment
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.img_trans(image) 
        image = image * 2. - 1.

        torch.random.manual_seed(seed)
        HSV_img = self.img_trans(HSV_img) 
        HSV_img = HSV_img * 2. - 1.

        torch.random.manual_seed(seed)
        gt_image = self.img_trans(gt_image) # ([1, 512, 512])
        gt_image = gt_image * 2. - 1.
        
        torch.random.manual_seed(seed)
        highlight_mask = self.mask_trans(highlight_mask)
        highlight_mask = np.array(highlight_mask)
        highlight_mask = torch.tensor(highlight_mask)
        highlight_mask = torch.unsqueeze(highlight_mask, dim=0)
        highlight_mask = highlight_mask / 255.

        return image, gt_image, highlight_mask, HSV_img, filename


    
class InpaintingDataTest(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        # image and mask 
        self.args = args
        self.image_path = glob(os.path.join(args.dir_test, args.test_input_type, "*.png"))
        # self.image_path_gt = glob(os.path.join(args.dir_test, "target", "*.png"))

        # augmentation 
        self.img_trans = transforms.Compose([
            transforms.ToTensor()])

        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])
        HSV_img = image.convert("HSV")

        gt_image = Image.open(self.image_path[index].replace(self.args.test_input_type, 'target').replace('.jpg', '.png')).convert('RGB') # pair

        highlight_mask_path = self.image_path[index].replace(self.args.test_input_type, 'mask')
        highlight_mask = Image.open(highlight_mask_path).convert('L')

        # augment
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.img_trans(image) 
        image = image * 2. - 1.

        torch.random.manual_seed(seed)
        HSV_img = self.img_trans(HSV_img) 
        HSV_img = HSV_img * 2. - 1.

        torch.random.manual_seed(seed)
        gt_image = self.img_trans(gt_image) # ([1, 512, 512])
        gt_image = gt_image * 2. - 1.
        
        torch.random.manual_seed(seed)
        highlight_mask = np.array(highlight_mask)
        highlight_mask = torch.tensor(highlight_mask)
        highlight_mask = torch.unsqueeze(highlight_mask, dim=0)
        highlight_mask = highlight_mask / 255.

        return image, gt_image, highlight_mask, HSV_img, filename
    
class InpaintingDataTestNoMask(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        # image and mask 
        self.args = args
        self.image_path = glob(os.path.join(args.dir_test, args.test_input_type, "*.png"))
        # self.image_path_gt = glob(os.path.join(args.dir_test, "target", "*.png"))

        # augmentation 
        self.img_trans = transforms.Compose([
            transforms.ToTensor()])

        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])


        gt_image = Image.open(self.image_path[index].replace(self.args.test_input_type, 'target').replace('.jpg', '.png')).convert('RGB') # pair

        # augment
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.img_trans(image) 
        image = image * 2. - 1.

        torch.random.manual_seed(seed)
        gt_image = self.img_trans(gt_image) # ([1, 512, 512])
        gt_image = gt_image * 2. - 1.


        return image, gt_image, filename
    
###################################### step 1 dataloader ####################################################
# 导入数据
device = "cpu"
#load clip
model, preprocess = clip.load("/root/autodl-tmp/rubyyao/PycharmProjects/nuclei_ins_seg/code/pretrained/RN50.pt", device=device)#ViT-B/32
for para in model.parameters():
    para.requires_grad = False

class InpaintingDataForPrompt(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        # 同时导入input和target数据
        input_images_path = os.path.join(args.dir_image, args.input_type)
        target_images_path = os.path.join(args.dir_image, args.input_type.replace("input", "target")) # unpair
        pred_images_path = None
        self.data_list = self.populate_train_list(input_images_path,target_images_path, pred_images_path)
        self.img_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.image_size_clip,args.image_size_clip))
            ]) 
        self.clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path).convert('RGB')
        filename = os.path.basename(image_path)

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.img_trans(image)
        
        image=self.clip_normalizer((image.reshape(1,3,224,224)))
        # image_features = self.clip_model.encode_image(image) 
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
		# 根据图片路径确定是哪一类别
        if ("target" in image_path):
            label=torch.from_numpy(np.array(1))
        else:
            label=torch.from_numpy(np.array(0))
	
        return image_features,label

    def populate_train_list(self,input_images_path,target_images_path,pred_images_path=None):
        # 此处是否需要使用pred_images_path存疑
        if pred_images_path!=None:
            image_list_overlight = glob(pred_images_path + "/*")
            image_list_lowlight = glob(input_images_path + "/*")
            image_list_normallight = glob(target_images_path + "/*")
            train_list = image_list_lowlight+image_list_normallight+image_list_overlight
        else:
            image_list_lowlight = glob(input_images_path + "/*")
            image_list_normallight = glob(target_images_path + "/*")
            image_ref_list=image_list_normallight.copy()
            image_input_list=image_list_lowlight.copy()
            if len(image_list_normallight)==0 or len(image_list_lowlight)==0:
                raise Exception("one of the image lists is empty!", len(image_list_normallight),len(image_list_lowlight))
            if len(image_list_normallight)<len(image_list_lowlight):
                while(len(image_ref_list)<len(image_list_lowlight)):
                    for i in image_list_normallight:
                        image_ref_list.append(i)
                        if(len(image_ref_list)>=len(image_list_lowlight)):
                            break
            else:
                while(len(image_input_list)<len(image_list_normallight)):
                    for i in image_list_lowlight:
                        image_input_list.append(i)
                        if(len(image_input_list)>=len(image_list_normallight)):
                            break
            
            train_list = image_input_list+image_ref_list
        # print(train_list)
        random.shuffle(train_list)
        return train_list
    
################# use in test ##########################
class InpaintingDataForPromptTest(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        # 同时导入input和target数据
        target_images_path = os.path.join(args.dir_image, "target_400")

        self.data_list = glob(target_images_path + "/*")
        self.img_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.image_size_clip,args.image_size_clip))
            ]) 
        self.clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path).convert('RGB')
        filename = os.path.basename(image_path)

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.img_trans(image)
        
        image=self.clip_normalizer((image.reshape(1,3,224,224)))
        # image_features = self.clip_model.encode_image(image) 
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
		# 根据图片路径确定是哪一类别
        if ("target" in image_path):
            label=torch.from_numpy(np.array(1))
        else:
            label=torch.from_numpy(np.array(0))
	
        return image_features,label
    

###################################### step 3 dataloader ####################################################

# 训练复原网络中的数据推理过程 在成对的测试集进行
class InpaintingDataInference(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        # image and mask 
        self.args = args
        self.image_path = glob(os.path.join(args.dir_image, args.input_type, "*.png"))
        # self.image_path_gt = glob(os.path.join(args.dir_test, "target", "*.png"))

        # augmentation 
        self.img_trans = transforms.Compose([
            transforms.ToTensor()])

        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])
        HSV_img = image.convert("HSV")

        gt_image = Image.open(self.image_path[index].replace(self.args.input_type, 'target').replace('.jpg', '.png')).convert('RGB')

        highlight_mask_path = self.image_path[index].replace(self.args.input_type, 'mask')
        highlight_mask = Image.open(highlight_mask_path).convert('L')

        # augment
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.img_trans(image) 
        image = image * 2. - 1.

        torch.random.manual_seed(seed)
        HSV_img = self.img_trans(HSV_img) 
        HSV_img = HSV_img * 2. - 1.

        torch.random.manual_seed(seed)
        gt_image = self.img_trans(gt_image) # ([1, 512, 512])
        gt_image = gt_image * 2. - 1.
        
        torch.random.manual_seed(seed)
        highlight_mask = np.array(highlight_mask)
        highlight_mask = torch.tensor(highlight_mask)
        highlight_mask = torch.unsqueeze(highlight_mask, dim=0)
        highlight_mask = highlight_mask / 255.

        return image, gt_image, highlight_mask, HSV_img, filename

class InpaintingDataInferenceNoMask(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        # image and mask 
        self.args = args
        self.image_path = glob(os.path.join(args.dir_image, args.input_type, "*.png"))
        # self.image_path_gt = glob(os.path.join(args.dir_test, "target", "*.png"))

        # augmentation 
        self.img_trans = transforms.Compose([
            transforms.ToTensor()])

        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])

        gt_image = Image.open(self.image_path[index].replace(self.args.input_type, 'target').replace('.jpg', '.png')).convert('RGB')

        # augment
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.img_trans(image) 
        image = image * 2. - 1.

        torch.random.manual_seed(seed)
        gt_image = self.img_trans(gt_image) # ([1, 512, 512])
        gt_image = gt_image * 2. - 1.
        

        return image, gt_image, filename
    
#################### function for DataForFineTuning #################
def populate_train_list(lowlight_images_path,normallight_images_path=None,overlight_images_path=None):
    image_list_lowlight = glob(os.path.join(lowlight_images_path, "*.png"))
    image_list_normallight = glob(os.path.join(normallight_images_path, "*.png"))
    
    image_ref_list=image_list_normallight.copy()
    image_input_list=image_list_lowlight.copy()
    if len(image_list_normallight)==0 or len(image_list_lowlight)==0:
        raise Exception("one of the image lists is empty!", len(image_list_normallight),len(image_list_lowlight))
    if len(image_list_normallight)<len(image_list_lowlight):
            while(len(image_ref_list)<len(image_list_lowlight)):
                for i in image_list_normallight:
                    image_ref_list.append(i)
                    if(len(image_ref_list)>=len(image_list_lowlight)):
                        break
    else:
        while(len(image_input_list)<len(image_list_normallight)):
            for i in image_list_lowlight:
                image_input_list.append(i)
                if(len(image_input_list)>=len(image_list_normallight)):
                    break
    train_list1=image_input_list
    train_list2=image_ref_list
    # print(train_list1)
    random.shuffle(train_list1)
    random.shuffle(train_list2)

    return train_list1,train_list2


def transform_matrix_offset_center(matrix, x, y):
    """Return transform matrix offset center.

    Parameters
    ----------
    matrix : numpy array
        Transform matrix
    x, y : int
        Size of image.

    Examples
    --------
    - See ``rotation``, ``shear``, ``zoom``.
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix 

def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.
    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0),)
    return rotated_img

def zoom(x, zx, zy, row_axis=0, col_axis=1):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]

    matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = cv2.warpAffine(x, matrix[:2, :], (w, h),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0),)
    return x



def augmentation(img,hflip,vflip,rot90,rot,zo,angle,zx,zy):
    if hflip:
        img=cv2.flip(img,1)
    if vflip:
        img=cv2.flip(img,0)
    if rot90:
        img = img.transpose(1, 0, 2)
    if zo:
        img=zoom(img, zx, zy)
    if rot:
        img=img_rotate(img,angle)
    return img

def preprocess_aug(img_list):
    hflip=random.random() < 0.5
    vflip=random.random() < 0.5
    rot90=random.random() < 0.5
    rot=random.random() <0.3
    zo=random.random()<0.3
    angle=random.random()*180-90
    zoom_range=(0.5, 1.5)
    zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    aug_img_list=[]
    for img in img_list:
        img = np.uint8((np.asarray(img)))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img=augmentation(img,hflip,vflip,rot90,rot,zo,angle,zx,zy)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        aug_img_list.append(img)
    return aug_img_list

def preprocess_feature(img):
    img = (np.asarray(img)/255.0) 
    img = torch.from_numpy(img).float()
    img=img.permute(2,0,1).to(device)
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_resize = transforms.Resize((224,224))
    img=img_resize(img)
    img=clip_normalizer(img.reshape(1,3,224,224))
    image_features = model.encode_image(img)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

#################### step 3 prompt fune-tuning #################
class DataForFineTuning(Dataset):

    def __init__(self, args, lowlight_images_path,normallight_images_path,semi1_path=None,semi2_path=None):
        self.train_list1,self.train_list2 = populate_train_list(lowlight_images_path,normallight_images_path)
        self.args = args
        self.size = self.args.image_size
        self.neg_path=lowlight_images_path
        self.semi1_path=semi1_path
        self.semi2_path=semi2_path
        self.data_list = self.train_list1
        # print("Total training examples (Well-lit):", len(self.train_list2))
        

    def __getitem__(self, index):

        data_lowlight_path = self.data_list[index]
        ref_path = self.train_list2[index]
        
        data_lowlight = Image.open(data_lowlight_path)
        ref = Image.open(ref_path)

        data_lowlight = data_lowlight.resize((self.size,self.size), Image.LANCZOS)
        ref = ref.resize((self.size,self.size), Image.LANCZOS)
        if self.semi1_path==None:
            img_list=preprocess_aug([data_lowlight,ref])
        elif self.semi2_path==None:
            file = data_lowlight_path.split('/')[-1]
            # semi_file = "/root/autodl-tmp/code/0423-baseline_v4/"
            semi1 = Image.open(os.path.join(self.semi1_path, file))
            img_list=preprocess_aug([data_lowlight,semi1,ref])
        else:
            file = data_lowlight_path.split('/')[-1]
            semi1 = Image.open(os.path.join(self.semi1_path, file))
            semi2 = Image.open(os.path.join(self.semi2_path, file))
            img_list=preprocess_aug([data_lowlight,semi1,semi2,ref])
            
        img_feature_list=[]
        for img in img_list:
            img_feature=preprocess_feature(img)
            img_feature_list.append(img_feature)
        
        return img_feature_list,1

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__': 

    # from attrdict import AttrDict
    # args = {
    #     'dir_image': '../../../dataset',
    #     'data_train': 'image',
    #     'dir_mask': '../../../dataset',
    #     'mask_type': 'pconv',
    #     'image_size': 512
    # }
    # args = AttrDict(args)

    import argparse
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--dir_image', default='/data/dataset/hg/SpecularHighlightRemoval/code/SHR_inputwork/SHIQ_data_10825/train', type=str)
    parser.add_argument('--image_size', default='200', type=int)
    args = parser.parse_args()

    data = InpaintingData(args)
    print(len(data))
    img, highlight, hsi, filename = data[10]
    from  torchvision import utils as vutils
    vutils.save_image(img, './image.jpg', normalize=True)

    highlight = highlight.type_as(img)
    vutils.save_image(highlight, './hmask.jpg', normalize=True)
    print(img.size(), highlight.size(), filename)