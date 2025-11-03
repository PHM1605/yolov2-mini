import os, cv2, random, torch
import torch.nn as nn 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET 
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCDataset, split_import_data
from model import Yolov1
from loss import YoloLoss
from utils import get_bboxes, mean_average_precision, save_checkpoint

root_dir = "import_data"
classes_file = "classes.txt"
train_img_files, val_img_files, test_img_files = split_import_data(root_dir)

class Compose(object):
  def __init__(self, transforms): 
    self.transforms = transforms 
  
  def __call__(self, img, bboxes):
    for t in self.transforms:
      img, bboxes = t(img), bboxes 
    return img, bboxes 

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 300 
NUM_WORKERS = 2
PIN_MEMORY = True 
LOAD_MODEL = False 
LOAD_MODEL_FILE = "best.pt"

def train_fn(train_loader, model, optimizer, loss_fn):
  loop = tqdm(train_loader, leave=True)
  mean_loss = [] 

  for batch_idx, (x, y) in enumerate(loop):
    x, y = x.to(DEVICE), y.to(DEVICE)
    out = model(x)
    loss = loss_fn(out, y)
    mean_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # update progress bar 
    loop.set_postfix(loss=loss.item())
  
  print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

def main():
  print(DEVICE) 
  model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  loss_fn = YoloLoss()

  # if LOAD_MODEL:
  #   load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

  train_dataset = VOCDataset(img_files=train_img_files, root_dir=root_dir, transform=transform)
  test_dataset = VOCDataset(img_files=val_img_files, root_dir=root_dir, transform=transform)
  train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False, drop_last=True)
  
  for epoch in range(EPOCHS):
    pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)
    mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
    print(f"epoch: {epoch} Train mAP: {mean_avg_prec}")

    k=0
    if (mean_avg_prec > 0.9) and (k==0):
      k = 1
      checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
      }
      save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE, exit_training=True)
    
    train_fn(train_loader, model, optimizer, loss_fn)

main()