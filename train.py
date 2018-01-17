#import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np

datadir = '../data/images_memory/'
traindir = '../data/images/subject1/'
valdir = '../data/images/subject2/'
num_epochs = 10
valid_size = 0.2

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),
                                normalize])

#train = datasets.ImageFolder(traindir, transform)
#valid = datasets.ImageFolder(valdir, transform)
#train_loader = torch.utils.data.DataLoader(
#        train, batch_size=1, shuffle=True, num_workers=1)
#valid_loader = torch.utils.data.DataLoader(
#        valid, batch_size=1, shuffle=True, num_workers=1)

train = datasets.ImageFolder(datadir, transform)
valid = datasets.ImageFolder(datadir, transform)
num_train = len(train)
indices = list(range(num_train))
split = int(np.floor(valid_size*num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:],indices[:split]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=1, sampler=train_sampler, num_workers=1)
valid_loader = torch.utils.data.DataLoader(
    valid, batch_size=1, sampler=valid_sampler, num_workers=1)

from models import Net
#model = models.vgg11(num_classes=2,pretrained=False)
model = Net(num_classes=4)

#inputs, labels = next(iter(train_loader))
#inputs, labels = Variable(inputs), Variable(labels)
#print( inputs.size() )
#outputs = model(inputs)
#print( outputs.size() )
#
#
#import sys
#sys.exit()


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

dloaders = {'train':train_loader, 'valid':valid_loader}

best_model_wts = model.state_dict()
best_acc = 0.0
dataset_sizes = {'train': len(dloaders['train'].dataset),
                 'valid': len(dloaders['valid'].dataset)}
for epoch in range(num_epochs):
    print('Epoch',epoch)
    for phase in ['train', 'valid']:
        if phase == 'train':
            scheduler.step()
            model.train(True)
        else:
            model.train(False)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dloaders[phase]:
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
                
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            train_epoch_loss = running_loss / dataset_sizes[phase]
            train_epoch_acc = running_corrects / dataset_sizes[phase]
        else:
            valid_epoch_loss = running_loss / dataset_sizes[phase]
            valid_epoch_acc = running_corrects / dataset_sizes[phase]

        if phase == 'valid' and valid_epoch_acc > best_acc:
            best_acc = valid_epoch_acc
            best_model_wts = model.state_dict()


    print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
          'valid loss: {:.4f} acc: {:.4f}'.format(
              epoch, num_epochs - 1,
              train_epoch_loss, train_epoch_acc,
              valid_epoch_loss, valid_epoch_acc))

print('Best val Acc: {:4f}'.format(best_acc))
model.load_state_dict(best_model_wts)

