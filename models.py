import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from torch import nn

class Multimobilenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel # size of the layer before the classifier

        # the input for the classifier should be two-dimensional, but we will have
        # [<batch_size&gt;, <channels&gt;, <width&gt;, <height&gt;]
        # so, let's do the spatial averaging: reduce <width&gt; and <height&gt; to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.Type = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=2)
        )
        self.target = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=15)
        )
    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        
      # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, start_dim=1)
        
        return {
          'Type':F.softmax(self.Type(x)),
          'targets':F.softmax(self.target(x))
         }



class Multialexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.alexnet()  # take the model without classifier
                # the input for the classifier should be two-dimensional, but we will have
        # [<batch_size&gt;, <channels&gt;, <width&gt;, <height&gt;]
        # so, let's do the spatial averaging: reduce <width&gt; and <height&gt; to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.Type = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1000, out_features=2)
        )
        self.target = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1000, out_features=15)
        )
    def forward(self, x):
        #print('Passing 1st phase')
        x = self.base_model(x)
        #print('Passing 2nd phase')
        #x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        #print('Flattening')
        #x = torch.flatten(x, start_dim=1)

        return {
          'Type': F.softmax(self.Type(x)),
          'targets': F.softmax(self.target(x))
        }