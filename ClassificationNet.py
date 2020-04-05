#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:41:37 2020

@author: dongshuai
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.models as models
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#from collections import Counter
import smp

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x
        
class ClassificationNet(nn.Module):
    
    def __init__(self,classes,channel=1):
        super(ClassificationNet,self).__init__()
        
        self.classes = classes
        self.channel = channel
        if self.channel != 3:
            self.input_layer = nn.Sequential(
                nn.Conv2d(self.channel, 3, kernel_size=(3,3), padding=(3, 3)),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
        
        self.calssification_model = models.resnet18(pretrained=False) 
            
        self.classification_output_layer = nn.Linear(1000,classes)
#        print(self)
        
    def forward(self,x):
        if self.channel != 3:
            x = self.input_layer(x)
        
        x = self.calssification_model.conv1(x)
        x = self.calssification_model.bn1(x)
        x = self.calssification_model.relu(x)
        x = self.calssification_model.maxpool(x)
        x = self.calssification_model.layer1(x)
        x = self.calssification_model.layer2(x)
        x = self.calssification_model.layer3(x)
        x = self.calssification_model.layer4(x)
        x = self.calssification_model.avgpool(x)
        x_res = x.view(x.size()[0], -1)
        cls_x = self.calssification_model.fc(x_res)

        x = self.classification_output_layer(cls_x)

        return x,x_res        
            
    def fit(self,x,y,y_true=None, lr=0.001, num_epochs=50,batch_size=256,start_epoch=10,alpha=0.5):

        x = torch.tensor(x,dtype=torch.float32)
        y = torch.tensor(y,dtype=torch.long)
        
        torch_dataset = Data.TensorDataset(x,y)
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.cuda()
        
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss().cuda()
            
        tag = ''
        for epoch in range(num_epochs):
            
            if epoch >= start_epoch:
                tag = 'with smp'
                _,_,z = self.predict(x)
                y_pseudo = smp.produce_pseudo_labels(z,y.numpy(),self.classes)
                self.y_pseudo = torch.tensor(y_pseudo,dtype=torch.long)
            
            train_loss = 0.0
            for step, (batch_x,batch_y) in enumerate(loader):                
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

                optimizer.zero_grad() 
                outputs,_ = self.forward(batch_x)

                if epoch >= start_epoch:
                    batch_y_pseudo = self.y_pseudo[step*batch_size:(step+1)*batch_size].cuda()
                    loss = (1-alpha)*criterion(outputs,batch_y)+alpha*criterion(outputs,batch_y_pseudo)
                
                else:
                    loss = criterion(outputs,batch_y)
                
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
                
            if (epoch+1)%1==0:
                y_pred,_,_ = self.predict(x)
                if y_true is not None:
                    correct = np.sum(y_pred==y_true)
                    acc = correct/len(y_true)
                    print('# %s Epoch %3d: Classification Loss: %.6f acc: %.6f' % (
                        tag,epoch+1, train_loss / len(loader.dataset),acc))
                else:
                    print('# %s Epoch %3d: Classification Loss: %.6f' % (tag,epoch+1, train_loss / len(loader.dataset)))
        
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        
    def predict(self,x):
        x = torch.tensor(x,dtype=torch.float32)
        with torch.no_grad():
            self.eval()
            self.cuda()
            
            outputs = None
            resnet_outputs = None
            torch_dataset = Data.TensorDataset(x)
            loader = Data.DataLoader(
                dataset=torch_dataset,
                batch_size=5000,
                shuffle=False
            )
            for step, data in enumerate(loader):                
                batch_x = data[0].cuda()

                output,resnet_output = self.forward(batch_x)
                output = output.cpu().numpy()
                resnet_output = resnet_output.cpu().numpy()
                
                if outputs is None:
                    outputs = output
                    resnet_outputs = resnet_output
                else:
                    outputs = np.vstack((outputs,output))
                    resnet_outputs = np.vstack((resnet_outputs,resnet_output))

            y_pred = np.argmax(outputs, 1)
            y_pred_prob = softmax(outputs)
            y_pred_prob = np.max(y_pred_prob,axis=1)
            return y_pred,y_pred_prob,resnet_outputs
