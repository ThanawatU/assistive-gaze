import torch
import torch.nn as nn

class GazeNN(nn.Module):
  def __init__(self, backbone=None):
    super(GazeNN, self).__init__()

    self.backbone = backbone
    self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Calculate flattened size after convolutions
    # For input size (2, 36, 60)
    # After pool1: (32, 18, 30)
    # After pool2: (64, 9, 15)
    # After pool3: (128, 4, 7)
    flat_size = 128 * 4 * 7

    # print(f"GazeNN __init__: flat_size = {flat_size}")

    self.fc_eye = nn.Linear(flat_size, 256)
    self.fc_combined = nn.Linear(256 + 3, 128) # +3 for head pose
    self.fc_out = nn.Linear(128, 3) # Correctly define the output layer

    # print(f"GazeNN __init__: self.fc_eye = {self.fc_eye}")
    # print(f"GazeNN __init__: self.fc_combined = {self.fc_combined}")
    # print(f"GazeNN __init__: self.fc_out = {self.fc_out}")

    self.dropout = nn.Dropout(0.2)

    self.relu = nn.ReLU()

  def forward(self, eye_input, head_pose):
    # print(f"GazeNN forward: eye_input shape = {eye_input.shape}, head_pose shape = {head_pose.shape}")
    x = self.relu(self.conv1(eye_input))
    x = self.pool1(x)
    x = self.relu(self.conv2(x))
    x = self.pool2(x)
    x = self.relu(self.conv3(x))
    x = self.pool3(x)

    x = x.view(x.size(0), -1)
    # print(f"GazeNN forward: x shape after convolutional blocks and flattening = {x.shape}")
    x = self.relu(self.fc_eye(x))
    # print(f"GazeNN forward: x shape after fc_eye = {x.shape}")
    x = torch.cat([x, head_pose], dim=1)
    # print(f"GazeNN forward: x shape after concatenating with head_pose = {x.shape}")
    x = self.relu(self.fc_combined(x)) # Use fc_combined here, which is nn.Linear(259, 128)
    # print(f"GazeNN forward: x shape after fc_combined = {x.shape}")
    x = self.dropout(x)

    gaze = self.fc_out(x) # Use the correctly defined fc_out
    # print(f"GazeNN forward: gaze shape after fc_out = {gaze.shape}")

    gaze = gaze / torch.norm(gaze, dim=1, keepdim=True)
    # print(f"GazeNN forward: gaze shape after normalization = {gaze.shape}")

    return gaze