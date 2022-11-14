import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    #img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# specify the image classes
classes = ['0','1','2','3','4','5','6','7','8','9']

#MODEL
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1=nn.Conv2d(1, 6, 3, padding=1)
        self.relu1=nn.LeakyReLU()
        self.pool1=nn.MaxPool2d(2,stride=2)
        self.conv2=nn.Conv2d(6, 16, 3, padding=1)
        self.relu2=nn.LeakyReLU()
        self.pool2=nn.MaxPool2d(2,stride=2)

        self.tconv1=nn.ConvTranspose2d(16, 6, 2, stride=2)
        self.relu3=nn.LeakyReLU()
        self.tconv2=nn.ConvTranspose2d(6, 1, 2, stride=2)


    def forward(self, x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        x=self.tconv1(x)
        x=self.relu3(x)
        x=self.tconv2(x)
        return torch.tanh(x)

model = Net()
#model = model.float()

#Load data    
transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.5), (0.5))
     ])

batch_size = 200

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.pool2.register_forward_hook(get_activation('pool2'))
#print(activation['encoder'])


#TRAINING  
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
  
# Using an Adam Optimizer with lr = 0.01
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-2,
                             weight_decay = 1e-8)

epochs = 1
for epoch in range(epochs):
    train_loss=0.0
    for data in trainloader:
        
        image, labels = data
        
        reconstructed = model(image.float())
        loss = loss_function(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*image.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(trainloader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))


print(count_parameters(model))
#TESTING

dataiter = iter(testloader)
images, labels = dataiter.next()

# get sample outputs
#images = images.float()
#noise_factor = 0.2
#images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape) 
#images = np.clip(x_test_noisy, 0., 1.)

output = model(images.float())
#images = images.numpy
output = output.view(batch_size, 1, 28, 28)
output = output.detach().numpy() #detach because out requires_grad

#Original Images
print("Original Images")
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(1):
    ax = fig.add_subplot(1,1, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
plt.show()

#Reconstructed Images
print('Reconstructed Images')
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(1):
    ax = fig.add_subplot(1, 1, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    ax.set_title(classes[labels[idx]])
plt.show() 

bias = open("bias.txt", "w") 
out = open("output.txt", "w") 
kernel = open("kernel.txt", "w") 
act = open("activation.txt", "w") 

print("\nBIAS")
for i in range(6):bias.write(str(model.conv1.bias.detach().numpy()[i])+'\n')
for i in range(16):bias.write(str(model.conv2.bias.detach().numpy()[i])+'\n')
for i in range(6):bias.write(str(model.tconv1.bias.detach().numpy()[i])+'\n')
bias.write(str(model.tconv2.bias.detach().numpy()[0])+'\n')

print("\nACTIVATION")
for k in range(16):
    for i in range(7):
        #for j in range(28):act.write(str(F.leaky_relu(activation['conv1'],2,2).numpy()[0][k][i][j])+'\n')
        for j in range(7):act.write(str(activation['pool2'].numpy()[0][k][i][j])+'\n')

print("\nKERNEL")
for i in range(6):
    for j in range(1):
        for k in range(3):
            for l in range(3):kernel.write(str(model.conv1.weight.detach().numpy()[i][j][k][l])+'\n')
for i in range(16):
    for j in range(6):
        for k in range(3):
            for l in range(3):kernel.write(str(model.conv2.weight.detach().numpy()[i][j][k][l])+'\n')
for i in range(16):
    for j in range(6):
        for k in range(2):
            for l in range(2):kernel.write(str(model.tconv1.weight.detach().numpy()[i][j][k][l])+'\n')
for i in range(6):
    for j in range(1):
        for k in range(2):
            for l in range(2):kernel.write(str(model.tconv2.weight.detach().numpy()[i][j][k][l])+'\n')

print("\nOUTPUT")
for k in range(28):
    for l in range(28):out.write(str(output[0][0][k][l])+'\n')

bias.close();out.close();kernel.close();act.close()
