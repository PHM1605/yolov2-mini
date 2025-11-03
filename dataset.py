import json, torch, os
import numpy as np
import random
from torchvision import transforms 

def convert_json_to_txt(label_dir, label_file):
  json_path = os.path.join(label_dir, label_file)
  with open(json_path, "r") as f:
    data = json.load(f)

  img_lookup = {img_info["id"]: img_info for img_info in data["images"]}
  annotations = data["annotations"]

  for anno in annotations:
    img_info = img_lookup.get(anno["image_id"])
    if img_info is None:
      continue 

    img_name = img_info["file_name"]
    img_width, img_height = img_info["width"], img_info["height"]
    x, y, w, h = anno["bbox"]

    x_center = (x+w/2) / img_width
    y_center = (y+h/2) / img_height 
    w_norm = w / img_width 
    h_norm = h / img_height 

    category_id = anno["category_id"]
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join("import_data", txt_name)
    
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "a") as f:
      f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def prepare_data():
  label_dir = "data/labels_json"
  for label_file in ["pascal_train2007.json", "pascal_val2007.json", "pascal_test2007.json"]:
    convert_json_to_txt(label_dir, label_file)

def split_import_data(root_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
  images = [f for f in os.listdir(root_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
  random.shuffle(images)
  n_total = len(images)
  n_train = int(train_ratio * n_total)
  n_val = int(val_ratio * n_total)

  train_files = images[:n_train]
  val_files = images[n_train:n_train+n_val]
  test_files = images[n_train+n_val:]

  print(f"Total {n_total} => Train {len(train_files)}, Val {len(val_files)}, Test {len(test_files)}")
  return train_files, val_files, test_files 

class VOCDataset(torch.utils.data.Dataset):
  def __init__(self, img_files, root_dir, S=7, B=2, C=20, transform=None):
    self.files = img_files 
    self.classes_file = "classes.txt"
    self.root_dir = root_dir
    self.transform = transform  
    with open(os.path.join(root_dir, self.classes_file)) as f:
      self.classes = [line.strip() for line in f.readlines()]
    self.S, self.B, self.C = S, B, C

  def __len__(self):
    return len(self.img_files)
  
  def __getitem__(self, index):
    img_name = self.files[index]
    img_path = os.path.join(self.root_dir, img_name)
    label_path = os.path.join(self.root_dir, os.path.splitext(img_name)[0] + ".txt")
    image = Image.open(img_path).convert("RGB")

    boxes = []
    with open(label_path) as f:
      for label in f.readlines():
        class_label, x, y, width, height = [
          float(x) 
          if float(x)!=int(float(x))
          else int(x) for x in label.strip().split()
        ]
        boxes.append([class_label, x, y, width, height])
    
    if self.transform:
      image, boxes = self.transform(image, torch.tensor(boxes))

    label_matrix = torch.zeros(self.S, self.S, self.B*5+self.C)
    for box in boxes:
      class_label, x, y, width, height = box.tolist()
      i, j = int(y*self.S), int(x*self.S)
      x_cell, y_cell, w_cell, h_cell = x*self.S-j, y*self.S-i, width*self.S, height*self.S 
      if label_matrix[i,j,20] == 0:
        label_matrix[i,j,20] = 1
        label_matrix[i,j,int(class_label)] = 1
        label_matrix[i,j,21:25] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
    return image, label_matrix 
