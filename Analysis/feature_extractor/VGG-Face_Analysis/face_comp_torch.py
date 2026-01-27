import numpy as np
import cv2

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

IMG_HEIGHT_VGG16 = 224
IMG_WIDTH_VGG16  = 224

# label def.
smiling_comp_labels = ['smiling_descent', 'smiling_ascent', 'ambiguous' ]
smiling_comp_labels_short = ['>', '<', 'A' ]
smiling_comp_values = { 'smiling_descent':0, 'smiling_ascent':1, 'ambiguous':2 }
smiling_comp_values_short = { '>':0, '<':1, 'A':2 }


###############################
###### Network
###############################

class VGGFace_conv(nn.Module):
    def __init__(self):
        super(VGGFace_conv, self).__init__()
        self.features = nn.ModuleDict(OrderedDict(
            {
                # === Block 1 ===
                'conv_1_1': nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                'relu_1_1': nn.ReLU(inplace=True),
                'conv_1_2': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                'relu_1_2': nn.ReLU(inplace=True),
                'maxp_1_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 2 ===
                'conv_2_1': nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                'relu_2_1': nn.ReLU(inplace=True),
                'conv_2_2': nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                'relu_2_2': nn.ReLU(inplace=True),
                'maxp_2_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 3 ===
                'conv_3_1': nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                'relu_3_1': nn.ReLU(inplace=True),
                'conv_3_2': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_2': nn.ReLU(inplace=True),
                'conv_3_3': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_3': nn.ReLU(inplace=True),
                'maxp_3_3': nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                # === Block 4 ===
                'conv_4_1': nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                'relu_4_1': nn.ReLU(inplace=True),
                'conv_4_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_2': nn.ReLU(inplace=True),
                'conv_4_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_3': nn.ReLU(inplace=True),
                'maxp_4_3': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 5 ===
                'conv_5_1': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_1': nn.ReLU(inplace=True),
                'conv_5_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_2': nn.ReLU(inplace=True),
                'conv_5_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_3': nn.ReLU(inplace=True),
                'maxp_5_3': nn.MaxPool2d(kernel_size=2, stride=2)
            }))

    def forward(self, x):
        # Forward through feature layers
        for k, layer in self.features.items():
            x = layer(x)

        # Flatten convolution outputs
        x = x.view(x.size(0), -1)

        return x

class siamese_vgg16based(nn.Module):
    def __init__(self, extractor):
        super(siamese_vgg16based, self).__init__()
        self.extractor = extractor
        self.comp_layer = nn.Sequential(
            nn.Linear(512 * 7 * 7 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 2))

    def forward(self, x1, x2):
        # Forward through feature layers
        x1 = self.extractor(x1)
        x2 = self.extractor(x2)
        # concatenate two features
        x_org = torch.cat((x1, x2), 1)
        x_swp = torch.cat((x2, x1), 1)
        # Forward through comparison layers
        x_org = self.comp_layer(x_org)
        x_swp = self.comp_layer(x_swp)
        return x_org, x_swp

def face_comp_siamese_model_vgg16based_for_train(weights=None, freeze_conv=True):
    ## build model
    # With a shared layer implementation layers containing the same weights are applied to both of multiple inputs because each layer was defined(=instanced) only once
    feature_extractor = VGGFace_conv()
    model = siamese_vgg16based(feature_extractor)
    # load weights from vggface
    if weights is not None:
        model.extractor.load_state_dict(torch.load(weights))
    if freeze_conv:
        # freeze parameters of extractor
        for param in model.extractor.parameters():
            param.requires_grad = False
    return model

def face_comp_siamese_model_vgg16based_for_predict(weights=None, on_device=torch.device('cpu')):
    ## build model
    # With a shared layer implementation layers containing the same weights are applied to both of multiple inputs because each layer was defined(=instanced) only once
    feature_extractor = VGGFace_conv()
    model = siamese_vgg16based(feature_extractor)
    # load weights from vggface
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=on_device))
    model.eval()
    return model

###############################
###### Confirm of Network
###############################
def denormalize_img(img):
    denormalize = transforms.Normalize((-129.1863/255, -104.7624/255, -93.5940/255), (1.0, 1.0, 1.0)) #RGB
    return denormalize(img)

def feature_foward(features, x):
    for k, layer in features.items():
        x = layer(x)
    return x


def grad_cam_each_input(feature):
    feature_vec = feature.grad.view(512, 7 * 7)
    alpha = torch.mean(feature_vec, axis = 1)
    feature = feature.squeeze(0)

    L = F.relu(torch.sum(feature * alpha.view(-1, 1, 1), 0))
    L = L.detach().cpu().numpy()
    
    L_min = np.min(L)
    L_max = np.max(L - L_min)
    L = (L - L_min)/(L_max+10e-4)

    # resize
    grad_cam = cv2.resize(L, (IMG_HEIGHT_VGG16,IMG_WIDTH_VGG16), cv2.INTER_LINEAR)

    # apply pseudo color visualization (heat map style)
    grad_cam = cv2.applyColorMap(np.uint8(grad_cam*255), cv2.COLORMAP_JET)

    # BGR to RGB
    grad_cam = cv2.cvtColor(grad_cam, cv2.COLOR_BGR2RGB)
    
    return grad_cam

def grad_cam(model, input_img_ref, input_img_tgt, on_device=torch.device('cpu')):
    # see the feature map of the last layer of extractor: conv_5_3
    features = model.extractor.features.eval().to(on_device)
    classifier = model.comp_layer.eval().to(on_device)
    if len(input_img_ref.shape) == 3:
        input_img_ref = input_img_ref.unsqueeze(0)
    if len(input_img_tgt.shape) == 3:
        input_img_tgt = input_img_tgt.unsqueeze(0)
        
    feature_ref = feature_foward(features, input_img_ref).clone().detach().requires_grad_(True)
    feature_ref_swp = feature_ref.clone().detach().requires_grad_(True)
    feature_tgt = feature_foward(features, input_img_tgt).clone().detach().requires_grad_(True)    
    feature_tgt_swp = feature_tgt.clone().detach().requires_grad_(True)
    
    pred = classifier(torch.cat((feature_ref.view(feature_ref.size(0), -1), feature_tgt.view(feature_tgt.size(0), -1)), 1))
    
    pred[0][0].backward()
    gradcam_ref_less = grad_cam_each_input(feature_ref)
    gradcam_tgt_less = grad_cam_each_input(feature_tgt)

    pred_ = classifier(torch.cat((feature_ref_swp.view(feature_ref_swp.size(0), -1), feature_tgt_swp.view(feature_tgt_swp.size(0), -1)), 1))
    pred_[0][1].backward()
    gradcam_ref_more = grad_cam_each_input(feature_ref_swp)
    gradcam_tgt_more = grad_cam_each_input(feature_tgt_swp)
    
    return pred, ((gradcam_ref_less, gradcam_tgt_less), (gradcam_ref_more, gradcam_tgt_more))


def make_result_image_with_gradcam(filename_input_img_ref, filename_input_img_tgt, grad_cam):
    ((gradcam_ref_top, gradcam_tgt_top), (gradcam_ref_bottom, gradcam_tgt_bottom)) = grad_cam
    img_raw_ref = Image.open(filename_input_img_ref)
    img_rsz_ref = img_raw_ref.resize((IMG_HEIGHT_VGG16,IMG_WIDTH_VGG16), Image.BICUBIC)
    img_ref = np.asarray(img_rsz_ref)
    img_raw_tgt = Image.open(filename_input_img_tgt)
    img_rsz_tgt = img_raw_tgt.resize((IMG_HEIGHT_VGG16,IMG_WIDTH_VGG16), Image.BICUBIC)            
    img_tgt = np.asarray(img_rsz_tgt)
    # overlay gradcam contribution maps on the original images
    gradcam_ref_top    = gradcam_ref_top/2    + img_ref/2
    gradcam_ref_bottom = gradcam_ref_bottom/2 + img_ref/2
    gradcam_tgt_top    = gradcam_tgt_top/2    + img_tgt/2
    gradcam_tgt_bottom = gradcam_tgt_bottom/2 + img_tgt/2
    # concatenating 
    img_input = np.concatenate([ img_ref, img_tgt], axis=1)
    gradcam_t = np.concatenate([ gradcam_ref_top,    gradcam_tgt_top   ], axis=1)
    gradcam_b = np.concatenate([ gradcam_ref_bottom, gradcam_tgt_bottom], axis=1)
    gradcam = np.concatenate([ img_input, gradcam_t, gradcam_b], axis=0)                
    # generate image
    img = Image.fromarray(np.uint8(gradcam))

    return img

def stream_consistency(y_true, y_pred):
    # y_true, y_pred : [ li_descent_org, li_ascent_org, li_descent_swp, li_ascent_swp ]
    # this criterion quantizes how much two stream likelihoods in y_pred are similar -> histogram similarity -> histogram intersection

    consistency = torch.minimum(y_pred[:,0],y_pred[:,2]) + torch.minimum(y_pred[:,1],y_pred[:,3])  # 'minimum' for element-wize min
    consistency = torch.mean(consistency)
    
    return consistency


def accuracy(y_true, y_pred):
    y_true_label_org = y_true[:,0:2].argmax(1)
    y_true_label_swp = y_true[:,2:4].argmax(1)
    y_pred_label_org = y_pred[:,0:2].argmax(1)
    y_pred_label_swp = y_pred[:,2:4].argmax(1)
    acc_org = y_true_label_org.eq(y_pred_label_org).sum().item()
    acc_swp = y_true_label_swp.eq(y_pred_label_swp).sum().item()
    acc = 0.5*(torch.mean(acc_org)+torch.mean(acc_swp))

    return acc

def show_model_info(model, batch_size = 16):
    from torchinfo import summary
    print(summary(model=model, input_size=((batch_size, 3, IMG_HEIGHT_VGG16, IMG_WIDTH_VGG16), (batch_size, 3, IMG_HEIGHT_VGG16, IMG_WIDTH_VGG16))))

###############################
###### Construct dataset
###############################
import os
import pandas as pd
from PIL import Image

class load_dataset(torch.utils.data.Dataset):
    def __init__(self, annotfile, img_dir, transform=None):
        self.annotfile = annotfile
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotfile, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img1 = Image.open(img1_path)
        #img1 = self.transform(img1)
        img2_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        img2 = Image.open(img2_path)
        #img2 = self.transform(img2)
        label = 0 if self.img_labels.iloc[idx, 2] > 0 else 1
        return img1, img2, label
    
class MySubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img1, img2, label = self.dataset[self.indices[idx]]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.indices)

from torchvision import transforms
class myTransform():
    def __init__(self):
        self.transform_dict = {
            'train': transforms.Compose([
                            transforms.Resize((IMG_HEIGHT_VGG16, IMG_WIDTH_VGG16)),
                            transforms.RandomHorizontalFlip(),  # randomly flip and rotate
                            transforms.RandomRotation(5),
                            transforms.ToTensor(),
                            transforms.Normalize((129.1863/255, 104.7624/255, 93.5940/255), (1.0, 1.0, 1.0)) #RGB
                            ]),
            'val': transforms.Compose([
                            transforms.Resize((IMG_HEIGHT_VGG16, IMG_WIDTH_VGG16)),
                            transforms.RandomHorizontalFlip(),  # randomly flip and rotate
                            transforms.RandomRotation(5),
                            transforms.ToTensor(),
                            transforms.Normalize((129.1863/255, 104.7624/255, 93.5940/255), (1.0, 1.0, 1.0)) #RGB
                            ]),
            'test': transforms.Compose([
                            transforms.Resize((IMG_HEIGHT_VGG16, IMG_WIDTH_VGG16)),
                            transforms.ToTensor(),
                            transforms.Normalize((129.1863/255, 104.7624/255, 93.5940/255), (1.0, 1.0, 1.0)) #RGB
                            ])
        }
        
