import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.v2 import ToTensor    
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader 
LR = 1e-3
epochs = 20
BATCH_SIZE = 128
train_data = FashionMNIST(root='./fashion_data', train=True, download=True, 
                          transform=ToTensor())
test_data = FashionMNIST(root='./fashion_data', train=False, download=True,
                         transform=ToTensor())
trian_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
from torch_nn import TorchNN

model = TorchNN()
print(model)
loss_fn = nn.CrossEntropyLoss()  
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()  
for epoch in range(epochs):
    for data, target in trian_dl:
        output = model(data.reshape(-1, 784))
        loss = loss_fn(output, target)
        model.zero_grad()  
        loss.backward()   
        optimizer.step()  
    print(f'Epoch:{epoch} Loss: {loss.item()}')
from sklearn.datasets import fetch_olivetti_faces
olivetti_faces = fetch_olivetti_faces(data_home='./face_data', shuffle=True)
print(olivetti_faces.data.shape)
print(olivetti_faces.target.shape)
print(olivetti_faces.images.shape)
import matplotlib.pyplot as plt

face = olivetti_faces.images[1]
plt.imshow(face, cmap='gray')
plt.show()
olivetti_faces.data[2]
olivetti_faces.target
import torch
import torch.nn as nn
images = torch.tensor(olivetti_faces.data)
targets = torch.tensor(olivetti_faces.target)
images.shape
targets.shape
dataset = [(img,lbl) for img,lbl in zip(images, targets)]
dataset[0]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
