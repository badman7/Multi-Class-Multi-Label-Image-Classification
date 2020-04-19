import os
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import os
from torch.utils.data import Dataset, DataLoader

class train_dataset(Dataset):
    def __init__(self, Labels,images , encoder , dataset,transform=None):
        self.images = images
        self.label = Labels
        self.transform = transform
        self.encoder = encoder
        self.dataset_dir=dataset
    def __getitem__(self, idx):
        #print(self.label[idx])
        image_name = self.images[idx]
        image_name=os.path.join(self.dataset_dir,self.images[idx])
        image = cv2.imread(image_name)
        b, g, r = cv2.split(image)
        #plt.imshow(image)
        image = cv2.merge((r, g, b))
        target = torch.zeros(len(self.encoder))
        #print(self.label[idx])
        lab=self.label[idx].split("|")
        for x in range(len(lab)):
            target[self.encoder[lab[x]]]=1
        if self.transform:
            image = self.transform(image)
        dict_data={}
        if target[3]==0: #### if the label is not 'No Finding'
            class_list=torch.zeros(2) ## we will consider abnormal class to be equal to 1
            class_list[1]=1;
            dict_data={
              'img':image,
              'labels':{
                  'Type':class_list,
                  'targets':target,
              }
          }           ######### Type : [Normal Class,Abnormal Class]
                  ######### Labels : [class1,class2,class3,...........,class15]
        else:
            class_list=torch.zeros(2) ### if the label is 'No Finding'
            class_list[0]=1; ## we will consider normal class to be equal to 1
            dict_data={
              'img':image,
              'labels':{
                  'Type':class_list,
                  'targets':target,
              }
          }
        return dict_data
    
    def __len__(self):
        return len(self.images)
    
class test_dataset(Dataset):
    def __init__(self, labels, images ,encoder,dataset,transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.encoder = encoder
        self.dataset_dir=dataset
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_name=os.path.join(self.dataset_dir,self.images[idx])
        image = cv2.imread(image_name)
        b, g, r = cv2.split(image)
        #plt.imshow(image)
        image = cv2.merge((r, g, b))
        target = torch.zeros(len(self.encoder))
        lab=labels[idx].split("|")
        for x in range(len(lab)):
            target[encoder[lab[x]]]=1
        if self.transform:
            image = self.transform(image)
        dict_data={}
        if target[3]==0: #### if the label is not 'No Finding'
            class_list=torch.zeros(2) ## we will consider abnormal class to be equal to 1
            class_list[1]=1;
            dict_data={
              'img':image,
              'labels':{
                  'Type':class_list,
                  'targets':target,
              }
          }           ######### Type : [Normal Class,Abnormal Class]
                  ######### Labels : [class1,class2,class3,...........,class15]
        else:
            class_list=torch.zeros(2) ### if the label is 'No Finding'
            class_list[0]=1; ## we will consider normal class to be equal to 1
            dict_data={
              'img':image,
              'labels':{
                  'Type':class_list,
                  'targets':target,
              }
          }
        return dict_data
    
    def __len__(self):
        return len(self.images)