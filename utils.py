import torch 
from collections import Counter

# boxes_preds: (N,S,S,4) with 4=xc,yc,w,h
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
  if box_format == "midpoint":
    box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3]/2
    box1_y1 = boxes_preds[...,1:2] - boxes_preds[...,3:4]/2
    box1_x2 = boxes_preds[...,0:1] + boxes_preds[...,2:3]/2
    box1_y2 = boxes_preds[...,1:2] + boxes_preds[...,3:4]/2

    box2_x1 = boxes_labels[...,0:1] - boxes_labels[...,2:3]/2
    box2_y1 = boxes_labels[...,1:2] - boxes_labels[...,3:4]/2
    box2_x2 = boxes_labels[...,0:1] + boxes_labels[...,2:3]/2
    box2_y2 = boxes_labels[...,1:2] + boxes_labels[...,3:4]/2
  
  if box_format == "corners":
    box1_x1, box1_y1 = boxes_preds[...,0:1], boxes_preds[...,1:2]
    box1_x2, box1_y2 = boxes_preds[...,2:3], boxes_preds[..., 3:4]
    box2_x1, box2_y1 = boxes_labels[...,0:1], boxes_labels[...,1:2]
    box2_x2, box2_y2 = boxes_labels[...,2:3], boxes_labels[...,3:4]
  
  # each (N,S,S,1)
  x1, y1 = torch.max(box1_x1, box2_x1), torch.max(box1_y1, box2_y1)
  x2, y2 = torch.max(box1_x2, box2_x2), torch.min(box1_y2, box2_y2)
  inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
  box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
  box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
  uni = box1_area + box2_area - inter 
  return inter / (uni + 1e-6)

# bboxes: list of 7*7=49 elements of [class_pred, prob_score, x1, y1, x2, y2]
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
  assert type(bboxes) == list 
  bboxes = [box for box in bboxes if box[1]>threshold]
  bboxes = sorted(bboxes, key=lambda x:x[1], reverse=True) # sort prob_score in descending order
  bboxes_after_nms = []
  
  while bboxes:
    chosen_box = bboxes.pop(0)
    # keep: different-class-boxes OR iou small with the chosen_box
    bboxes = [
      box for box in bboxes 
      if box[0] != chosen_box
      or intersection_over_union(
        torch.tensor(chosen_box[2:]),
        torch.tensor(box[2:]),
        box_format=box_format
      ) < iou_threshold
    ]
    bboxes_after_nms.append(chosen_box)
  
  return bboxes_after_nms

# pred_boxes: list of preds, each row [image_idx, class_pred, prob, x1, y1, x2, y2]
# true_boxes: list of trues, each row [image_idx, class_true, prob, x1, y1, x2, y2]
def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
  avg_precisions = []
  epsilon = 1e-6
  for c in range(num_classes):
    # save list of detections and list of ground_truths OF ONE CLASS
    detections, ground_truths = [], []
    for detection in pred_boxes:
      if detection[1] == c: 
        detections.append(detection)
    for true_box in true_boxes:
      if true_box[1] == c:
        ground_truths.append(true_box)
    
    # amount of true boxes per image
    # {<imageId>:<number-boxes>} => {"0":3, "1":2, ...}
    amount_bboxes = Counter([gt[0] for gt in ground_truths])
    # to flag which true boxes have been detected
    # {"0":[0,0,0],"1":[0,0]}
    for key, val in amount_bboxes.items():
      amount_bboxes[key] = torch.zeros(val)

    

# Convert bboxes output from Model in CELL COORDS to ENTIRE IMAGE COORDS
def convert_cellboxes(predictions, S=7):
  predictions = predictions.to("cpu")
  batch_size = predictions.shape[0]
  predictions = predictions.reshape(batch_size, 7, 7, 30)
  # w_cell, h_cell (NOT w_norm/h_norm)
  bboxes1 = predictions[..., 21:25] # anchor 1 # (N,S,S,4)
  bboxes2 = predictions[..., 26:30] # anchor 2 # (N,S,S,4)
  scores = torch.cat((
    predictions[...,20].unsqueeze(0),
    predictions[...,25].unsqueeze(0)
  ), dim=0) # (2,N,S*S)
  best_box = scores.argmax(0).unsqueeze(-1) # (N,S,S,1)
  best_boxes = bboxes1 * (1-best_box) + best_box * bboxes2 # (N,S,S,4) = dimension of the best boxes of each cell
  # repeat: (N,7,7,1) of [[0,1,...,6],[0,1,...,6],...]
  cell_indices = torch.arange(7).repeat(batch_size,7,1).unsqueeze(-1) 
  # NORM coords
  x = 1 / S * (best_boxes[...,:1] + cell_indices) 
  y = 1 / S * (best_boxes[...,1:2] + cell_indices.permute(0,2,1,3))
  w_h = 1 / S * best_boxes[..., 2:4]
  converted_bboxes = torch.cat((x, y, w_h), dim=-1) # (N,S,S,4)
  predicted_class = predictions[...,:20].argmax(-1).unsqueeze(-1) # (N,S,S,1)
  best_confidence = torch.max(predictions[...,20], predictions[...,25]).unsqueeze(-1) # (N,S,S,1)
  # (N,S,S,6) with [class_int, prob, x_norm, y_norm, w_norm, h_norm]
  converted_preds = torch.cat(
    (predicted_class, best_confidence, converted_bboxes), 
    dim=-1
  )
  return converted_preds

# convert model prediction to list of boxes (in ENTIRE IMAGE COORDS)
def cellboxes_to_boxes(out, S=7):
  # out: (N,S,S,C+5*B)
  # converted_pred: (N,S*S,6) with [class_int, prob, x_norm, y_norm, w_norm, h_norm]
  converted_pred = convert_cellboxes(out).reshape(out.shape[0], S*S, -1)
  converted_pred[...,0] = converted_pred[...,0].long()
  all_bboxes = [] # of 1 batch
  for ex_idx in range(out.shape[0]):
    bboxes = [] # of 1 image
    for bbox_idx in range(S*S):
      bboxes.append([x.item() for x in converted_pred[ex_idx,bbox_idx,:]])
    all_bboxes.append(bboxes) # list of 16 lists (if batch=16); each 7*7=49 elements
  
  return all_bboxes

# return 1 list of preds and 1 list of true
# each element [image_idx, class_pred, prob, x1, y1, x2, y2]
def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells", box_format="midpoint", device="cuda"):
  all_pred_boxes = []
  all_true_boxes = []
  
  model.eval()
  image_idx = 0
  for batch_idx, (x, labels) in enumerate(loader):
    x = x.to(device)
    labels = labels.to(device) # (N,S,S,B*5+C)
    with torch.no_grad():
      predictions = model(x) # (N,S,S,B*5+C)
    
    batch_size = x.shape[0]
    # list of 16 lists (if batch=16); each 7*7=49 elements
    true_bboxes = cellboxes_to_boxes(labels)
    bboxes = cellboxes_to_boxes(predictions)

    for idx in range(batch_size):
      # nms_boxes: list of [class_pred, prob_score, x1, y1, x2, y2]
      nms_boxes = non_max_suppression(
        bboxes[idx],
        iou_threshold=iou_threshold,
        threshold=threshold,
        box_format=box_format
      )
      for nms_box in nms_boxes:
        # list of [train_idx, class_pred, prob, x1, y1, x2, y2]
        all_pred_boxes.append( [image_idx] + nms_box )
      
      # loop runs 7*7=49 times, pick <box> from each cell 
      for box in true_bboxes[idx]:
        if box[1] > threshold:
          all_true_boxes.append( [image_idx] + box )

      image_idx += 1

  model.train()
  # return 1 list of preds and 1 list of true
  # each element [image_idx, class_pred, prob, x1, y1, x2, y2]
  return all_pred_boxes, all_true_boxes

