import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from zmq import device
from model import *
from adai_optim import *

import Bortfeld as bfpy

bf = bfpy.Bortfeld()

class BortfeldFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z):
        # x : input bf parameters
        # z : depth
        ctx.save_for_backward(x, z)
        y = torch.FloatTensor(bf.bf_dose(z.numpy(),x.numpy()))
        return y
    @staticmethod
    def backward(ctx, grad_out):
        x, z = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            jacobian = torch.FloatTensor(bf.bf_grad(z.numpy(),x.numpy())).view(-1,z.size(0))
            grad_out = jacobian*grad_out#.to(dtype=torch.float32)
        return grad_out, None

class SimpleModel(nn.Module):
    def __init__(self, init_para):
        super(SimpleModel, self).__init__()
        if init_para is None:
            self.para = torch.nn.parameter.Parameter(torch.rand(3*4,1))
        else:
            self.para = torch.nn.parameter.Parameter(init_para)#.to(dtype=torch.float32)
        # self.bf_layer = BortfeldFunction.apply
    def forward(self, z):
        y = BortfeldFunction.apply(self.para, z)
        return y
def train(net,z,idd,N,lb,ub,optimizer):
    loss_fn = nn.MSELoss()
    Loss = np.zeros((N,1))
    for i in range(N):
        optimizer.zero_grad()
        outputs = net(z)
        loss = loss_fn(outputs, idd)
        loss.backward()
        optimizer.step()
        p = net.para.data
        p[p < lb] = lb[p < lb]
        p[p > ub] = ub[p > ub]
        net.para.data = p
        Loss[i] = loss.item()
    return outputs.detach().numpy(),p.detach().numpy(),Loss

idd = [1.43307517079307,1.52749778550888,1.52176209822888,1.45988988754245,1.51362774581819,1.55749438718526,1.54100767071415,1.61445040533812,1.59649346050251,1.57476749200874,1.56147753425800,1.72913868261474,1.58427687134312,1.76582730218775,1.62118329083257,1.67578661494004,1.68131109633927,1.59926887420751,1.47957121655549,1.55632743150149,1.48645604677248,1.43502736027428,1.47330786307187,1.30498464998204,1.33495103124399,1.28779046693401,1.19608691503579,1.30555049111702,1.13980852323674,1.16296306964750,1.16329785889346,1.16876542889985,1.04585817582407,1.16368977273727,0.993400646947983,1.06190066043062,1.02952202622885,0.894655564846541,0.977834595290845,0.869826970767857,0.777334627629335,0.679732448356666,0.579143210490526,0.491131383427371,0.373332812061214,0.235808190003479,0.170628290751918,0.0942006765248990,0.0768708470694356,0.0600464947844232,0.0589945834410412,0.0373502674553598,0.0277157436702544,0.0164482893607179,0.00926209253195324,0.0124864084696967,0.00621275529675850,0.0113050934663174,0.00617991496081731,0.0191256879736600,0.00408123248028887,0.0175982288096169,0.000324068510980994,0.0142730837067861]
idd = torch.FloatTensor(idd)
z = torch.linspace(0,19,64,requires_grad= False).view(-1,1)
x = torch.FloatTensor([3.9199, 0.2744, 0.0010, 0.0057,13.5097, 0.9457, 0.0010, 0.0186,10.0225, 0.7016, 0.0010, 0.0118]).view(-1,1)
lb = torch.FloatTensor([1e-8,  1e-8,-10,  1e-8,  1e-8,  1e-8,-10,  1e-8,  1e-8,  1e-8,-10,  1e-8]).view(-1,1)
ub = torch.FloatTensor([22.8373,10.0000,10.0000,10.0000,22.8373,10.0000,10.0000,10.0000,22.8373,10.0000,10.0000,10.0000]).view(-1,1)

# x0 = torch.FloatTensor([3.9199, 0.2744, 0.0010, 0.0057,13.5097, 0.9457, 0.0010, 0.0186,10.0225, 0.7016, 0.0010, 0.0118]).requires_grad_()
# torch.autograd.gradcheck(BortfeldFunction.apply, (x0, z))


lr =1e-2

N = 200
net = SimpleModel(x)
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
outputs1,para,loss1 = train(net,z,idd,N,lb,ub,optimizer)
print('Adam',loss1[-1])

x = torch.FloatTensor([3.9199, 0.2744, 0.0010, 0.0057,13.5097, 0.9457, 0.0010, 0.0186,10.0225, 0.7016, 0.0010, 0.0118]).view(-1,1)
net = SimpleModel(x)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=False)
outputs2,para,loss2 = train(net,z,idd,N,lb,ub,optimizer)
print('SGD',loss2[-1])

x = torch.FloatTensor([3.9199, 0.2744, 0.0010, 0.0057,13.5097, 0.9457, 0.0010, 0.0186,10.0225, 0.7016, 0.0010, 0.0118]).view(-1,1)
net = SimpleModel(x)
optimizer = Adai(net.parameters(), lr=lr, betas=(0.1, 0.99), eps=1e-03, weight_decay=0)
outputs3,para,loss3 = train(net,z,idd,N,lb,ub,optimizer)
print('Adai',loss3[-1])

fig,ax = plt.subplots()
iter = range(N)
line1, = ax.semilogy(iter, loss1, label='Adam')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line2, = ax.semilogy(iter, loss2, label='SGD')
line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line3, = ax.semilogy(iter, loss3, label='Adai')
ax.legend()
plt.show()

# fig,ax = plt.subplots()
# line1, = ax.plot(z.numpy(), outputs1, label='Adam')
# line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
# line2, = ax.plot(z.numpy(), outputs2, label='SGD')
# line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
# line3, = ax.plot(z.numpy(), outputs3, label='Adai')
# line3.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
# line4, = ax.plot(z.numpy(), idd.numpy(), label='IDD')
# ax.legend()
# plt.show()
