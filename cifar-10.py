#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# In[2]:


batch_size = 4


# In[3]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# In[5]:


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

net.to(device)

# In[6]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[7]:


print(len(trainset))


# In[12]:


import time

def trainer(dataloader, net, optimizer, criterion, printing=False):
    data_size = len(trainloader.__dict__['dataset'].__dict__['data'])
    batch_size = trainloader.__dict__['batch_size']
    total_loss = 0.0
    t0 = time.time()
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        if printing:
            elapsed_time = time.time() - t0
            progress = (i + 1) * batch_size / data_size
            time_left = elapsed_time / progress - elapsed_time
            #print('\r{0:4d}'.format(i), end='')
            print('\r{0:5d}/{1} Elapsed time: {2:.2f} s Estimated time left: {3:.2f} s     '.format(i * batch_size, data_size, elapsed_time, time_left), end='')
            #print('\r{0:5d}/{0} Elapsed time: {0:.2f} s'.format(i * batch_size, end=''))

    # Print new line
    print('\rTraining with {0} inputs done after {1:.2f} seconds.'.format(data_size, time.time() - t0))
    print()
    return total_loss


# In[13]:


for epoch in range(2):
    trainer(trainloader, net, optimizer, criterion, True)

# In[23]:


#print(trainloader.__dict__)
#print(len(trainloader.__dict__['dataset'].__dict__['data']))
#batch_size also possible to get from DataLoader
print(trainloader.__dict__['batch_size'])


# In[9]:


PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# In[10]:


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        #images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# In[11]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        #images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# In[ ]:




