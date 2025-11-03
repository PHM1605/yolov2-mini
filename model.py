architecture_config = [
  # kernel, out_dim, stride, padding
  (7, 64, 2 ,3),
  "M",
  (3, 192, 1, 1),
  "M",
  (1, 128, 1, 0),
  (3, 256, 1, 1),
  (1, 256, 1, 0),
  (3, 512, 1 ,1),
  "M",
  [(1, 256, 1, 0), (3, 512, 1, 1), 4],
  (1, 512, 1, 0),
  (3, 1024, 1, 1),
  "M", 
  [(1, 512, 1, 0), (3, 512, 1, 1), 2],
  (3, 1024, 1, 1),
  (3, 1024, 2, 1),
  (3, 1024, 1, 1),
  (3, 1024, 1, 1)
]

class Yolov1(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super().__init__()
    self.architecture = []