import torchvision.transforms as transforms
import PIL
from PIL import Image
import os

def image_transform(scale_size=None, crop_size=None, mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]  # resnet imagenet
    std = std or [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(scale_size),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.CenterCrop(size),
        transforms.ToTensor(),  # divide by 255 automatically
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def encode_labels(labels):
    encoder=dict()
    cnt=0
    for i in range(len(labels)):
        x=labels[i].split("|")
        for j in range(len(x)):
            if x[j] in encoder.keys():
                continue
            else:
                encoder[x[j]]=cnt
                cnt=cnt+1
    print(len(encoder))
    return encoder
def get_existing_images(dataset_dir,train_images,train_labels):
    local_images=[]
    local_labels=[]
    for i in range(len(train_images)):
        image_dir=os.path.join(dataset_dir,train_images[i])
        if os.path.exists(image_dir):
            local_images.append(train_images[i])
            local_labels.append(train_labels[i])
    train_images=local_images
    train_labels=local_labels
    return train_images,train_labels