from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from skimage.transform import resize
import torch

# train_path = 'sample/handsup_train.txt'

class YOLODataLoader(Dataset):
    def __init__(self, parameters, train_path, is_train):
        assert os.path.exists(train_path), '{} not exists'.format(train_path)
        self.img_files = open(train_path,'r').readlines()
        self.label_files = [path.replace('.png','.txt').replace('.jpg','.txt') for path in self.img_files]
        self.img_shape = (parameters['img_size'], parameters['img_size'])
        self.max_objects = 50
        self.is_train = is_train

    def __getitem__(self,index):        
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))
        assert len(img.shape) == 3, 'what!'+img_path
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if self.is_train:
            input_img,filled_labels = processInputImageLabel(self.img_shape,img,self.max_objects, label_path,self.is_train)
            return img_path, input_img, filled_labels
        else:
            input_img = processInputImageLabel(self.img_shape,img,self.max_objects, label_path,self.is_train)
            return img_path, input_img


    def __len__(self):
        return len(self.img_files)

def getDataLoader(parameters, train_path,is_train):
    dataset = YOLODataLoader(parameters, train_path, is_train)
    batch_size = 1 if not is_train else parameters['batch_size']
    dataloader = torch.utils.data.DataLoader(dataset=dataset,\
        batch_size = batch_size,\
        shuffle=True,\
        num_workers=parameters['n_cpu'])

    return dataloader

def processInputImageLabel(img_shape,img,max_objects,label_path, is_train):
    h, w, _ = img.shape
    dim_diff = np.abs(h-w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1,pad2),(0,0),(0,0)) if h <= w else ((0,0),(pad1,pad2),(0,0))
    input_img = np.pad(img, pad, 'constant', constant_values=128)/255.0
    padded_h, padded_w, _ = input_img.shape
    input_img = resize(input_img, (*img_shape, 3), mode='reflect',anti_aliasing=True)
    input_img = np.transpose(input_img, (2,0,1))
    input_img = torch.from_numpy(input_img).float()

    if not is_train:
        return input_img

    labels = None
    assert os.path.exists(label_path), '{} not exists'.format(label_path)
    labels = np.loadtxt(label_path).reshape(-1,5)
    x1 = w * (labels[:,1] - labels[:,3]/2)
    y1 = h * (labels[:,2] - labels[:,4]/2)
    x2 = w * (labels[:,1] + labels[:,3]/2)
    y2 = h * (labels[:,2] + labels[:,4]/2) 

    x1 += pad[1][0]
    y1 += pad[0][0]
    x2 += pad[1][0]
    y2 += pad[0][0]

    labels[:,1] = ((x1+x2)/2)/padded_w
    labels[:,2] = ((y1+y2)/2)/padded_h
    labels[:,3] *= w/padded_w
    labels[:,4] *= h/padded_h

    filled_labels = np.zeros((max_objects, 5))
    if labels is not None:
        filled_labels[range(len(labels))[:max_objects]] = labels[:max_objects]
    filled_labels = torch.from_numpy(filled_labels)


    return input_img, filled_labels

# img = np.array(Image.open('sample/handsup/000000.jpg'))
# cv2.rectangle(img, (1327, 1221), (1446, 1326), (0, 255, 0), 3)
# plt.imshow(img);

# D:\retail-product-checkout-dataset\data\instances_train2019.json
