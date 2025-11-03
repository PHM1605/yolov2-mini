
# boxes_preds: (N,4)
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
  if box_format == "midpoint":
    pass 

# Convert bboxes output from Model in CELL COORDS to ENTIRE IMAGE COORDS
def convert_cellboxes(predictions, S=7):

# convert model prediction to list of boxes (in ENTIRE IMAGE COORDS)
def cellboxes_to_boxes(out, S=7):

def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells", box_format="midpoint", device="cuda"):
  all_pred_boxes = []
  all_true_boxes = []
  
  model.eval()
  train_idx = 0
  for batch_idx, (x, labels) in enumerate(loader):
    x = x.to(device)
    labels = labels.to(device)
    with torch.no_grad():
      predictions = model(x) 
    
    batch_size = x.shape[0]
    true_bboxes = cellboxes_to_boxes(labels)
    bboxes = cellboxes_to_boxes(predictions)

    for idx in range(batch_size):
      nms_boxes = non_max_suppression(
        bboxes[idx],
        
      )