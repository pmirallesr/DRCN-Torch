import numpy as np
import torch
import os
import matplotlib.pyplot as plt

# Fix the random seeds
def setSeeds(randomSeed):
  """ Based on: https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752"""
  np.random.seed(randomSeed)
  torch.manual_seed(randomSeed)
  os.environ['PYTHONASHSEED']=str(randomSeed)
  #random.seed(randomSeed)

# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.rcParams["figure.figsize"] = [16,9]    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# Some padding calculation tools.

def calcConvSamePadding(input, convNet, nDim = 2):
  """ Calculates padding required for convNet to produce output of same dimensions as input """
  pad_h = math.ceil((convNet.kernel_size[0] - input.shape[2] * (1 - convNet.stride[0]) - convNet.stride[0]) / 2)
  pad_w = math.ceil((convNet.kernel_size[1] - input.shape[3] * (1 - convNet.stride[1]) - convNet.stride[1]) / 2)
  return (pad_h, pad_h, pad_w, pad_w)

def calcPoolSamePadding(input, poolNet, nDim = 2): 
  """ Calculates padding required for poolNet to produce output of same dimensions as input """
  pad_h = math.ceil(((input.shape[2] - 1)*poolNet.stride[0] + 1 + poolNet.dilation*(poolNet.kernel_size[0] - 1) - input.shape[2])/2)
  pad_w = math.ceil(((input.shape[3] - 1)*poolNet.stride[1] + 1 + poolNet.dilation*(poolNet.kernel_size[1] - 1) - input.shape[3])/2)
  return (pad_h, pad_h, pad_w, pad_w)

# Custom init for conv weights

def weights_init(m):
      if isinstance(m, nn.Conv2d):
          nn.init.xavier_uniform_(m.weight.data, gain=1)
          nn.init.constant_(m.bias.data, 0.1)

# Evaluate a model wrt to a dataLoader. Assumes the model's output can be compared with the targets from data
def getLabellingAccuracy(dataLoader, model):
  correct = 0
  total = 0
  for batch_id, (data, target) in enumerate(dataLoader):
    data = data.to(device)
    target = target.to(device)
    forward = model(data)
    pred = torch.max(forward, 1).indices
    total += target.size(0)
    # print(total)
    correct += (pred == target).sum().item()
    # print(correct)
  return(correct, total, correct/total*100)

# Given a dictionary representing experimental settings, calculate the amount of total experiments
def calcExperimentsAmount(experimentRanges):
    totalExp = 1
    for paramSet in experimentRanges.values():
        totalExp *= len(paramSet)
    return totalExp

# Create a string for naming a folder containing a set of experiments from the experimental ranges
# String = Var1_Var2_Var3 essentially
def create_zip_name(experimentRanges):
    zipName = ""
    for expVariable in experimentRanges.keys():
        if(len(experimentRanges[expVariable]) > 1 ):
            zipName += expVariable + "_"
    return zipName[:-1]
    