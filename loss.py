import torch
import torch.nn as nn 
from utils import intersection_over_union

class YoloLoss(nn.Module):
  def __init__(self, S=7, B=2, C=20):
    super().__init__()
    self.mse = nn.MSELoss(reduction="sum")
    self.S, self.B, self.C = S, B, C 
    self.lambda_coord, self.lambda_noobj = 5, 0.5
  
  # predictions, target: (N,S,S,C+5*B)
  def forward(self, predictions, target):
    predictions = predictions.reshape(-1, self.S, self.S, self.B*5+self.C)
    # cat 2 (N,S,S,1) => (2,N,S,S,1)
    ious = torch.cat((
      intersection_over_union(predictions[...,21:25], target[...,21:25]).unsqueeze(0),
      intersection_over_union(predictions[...,26:30], target[...,21:25]).unsqueeze(0)
    ))
    _, bestbox = torch.max(ious, dim=0) # which box is best based on ious; (N,S,S,1)
    exists_box = target[...,20:21] # (N,S,S,1)
    # (N,S,S,4), most are zeros, except where there are boxes (then keep its xywh)
    box_predictions = exists_box*(
      (1-bestbox)*predictions[...,21:25] + bestbox*predictions[...,26:30]
    )
    box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4]+1e-6))
    box_targets = exists_box * target[...,21:25]
    box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])
    # box loss 
    box_loss = self.mse(
      torch.flatten(box_predictions, end_dim=-2), # (N,S,S,4)=>(N*S*S,4)
      torch.flatten(box_targets, end_dim=-2) # (N*S*S,4)
    )
    # object loss 
    object_loss = self.mse(
      torch.flatten(exists_box*((1-bestbox)*predictions[...,20:21] + bestbox*predictions[...,25:26])),
      torch.flatten(exists_box)
    )
    # no_object_loss 
    no_object_loss = (
      self.mse(
        torch.flatten((1-exists_box)*predictions[...,20:21], end_dim=-2),
        torch.flatten((1-exists_box)*target[...,20:21], end_dim=-2) 
      ) + 
      self.mse(
        torch.flatten((1-exists_box)*predictions[...,25:26], end_dim=-2),
        torch.flatten((1-exists_box)*target[...,20:21], end_dim=-2)
      )
    )
    # class loss
    class_loss = self.mse(
      torch.flatten(exists_box * predictions[...,:20], end_dim=-2),
      torch.flatten(exists_box*target[...,:20], end_dim=-2)
    )
    loss = (
      self.lambda_coord*box_loss
      + object_loss
      + self.lambda_noobj*no_object_loss
      + class_loss 
    )
    return loss 
