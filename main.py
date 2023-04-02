import time
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
 
device = "cuda" if torch.cuda.is_available() else "cpu"
class Net(nn.Module):
  def __init__(self, input_size, output_size, hidden_size = 800, layer_num = 3):
    super(Net, self).__init__()
    self.layer_num = layer_num
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.inputlayer = nn.Linear(self.input_size, self.hidden_size)
    self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.layer_num - 2)])
    self.activation = nn.ReLU(self.hidden_size)
    self.outputlayer = nn.Linear(self.hidden_size, self.output_size)
  @autocast()
  def forward(self, x):
    x = self.inputlayer(x)
    x = self.activation(x)
    for i in range(self.layer_num - 2):
      x = self.linears[i](x) 
      x = self.activation(x)
    x = self.outputlayer(x)
    return x

learning_rate = 1e-3
epochs = 10
batch_size = 100
input_size = 28 * 28
output_size = 10

model = Net(input_size = input_size, output_size = output_size)
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root = "./data", train = True, transform = transform, download = True)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = torchvision.datasets.MNIST(root = "./data", train = False, transform = transform, download = True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


datasize = len(train_data)
n_iterations = math.ceil(datasize / batch_size)

datapoints = [[], []]
for epoch in range(epochs):
  avg_loss = 0
  for i, [images, labels] in enumerate(train_dataloader):
    optimizer.zero_grad()
    images = images.view(-1, input_size)
    images = images.to(device)
    labels = labels.to(device)
    with autocast():
      pred = model(images)
      loss = criterion(pred, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    avg_loss += loss.item() * images.shape[0]
    if (i + 1) % 100 == 0:
      print(f"Loss: {loss:.7f} [iteration {i + 1}/{n_iterations} in epoch {epoch + 1}/{epochs}]")
  avg_loss /= datasize   

  datapoints[0].append(epoch + 1)
  datapoints[1].append(avg_loss)
  if (epoch + 1) % 10 == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "scaler": scaler.state_dict()
    }
    filename = "./checkpoints/" + time.asctime() + ".pt"
    torch.save(checkpoint, filename)

import matplotlib.pyplot as plt
plt.style.use('classic')
fig = plt.figure()
loss_fig = fig.add_subplot(1,1,1)
loss_fig.set_title("Loss curve")
loss_fig.set_xlabel("Epochs")
loss_fig.set_ylabel("Loss")
loss_fig.plot(datapoints[0], datapoints[1])
plt.show()

correct = 0
with torch.no_grad():
  for i, [images, labels] in enumerate(test_dataloader):
    images = images.view(batch_size, -1)
    images = images.to(device)
    labels = labels.to(device)
    with autocast():
      pred = model(images)
      loss = criterion(pred, labels)
    _, guess = torch.max(pred, dim = 1)
    correct += torch.sum(guess == labels)
print(f"Accracy: {correct / len(test_data) * 100}% in {len(test_data)} tests")
