import torch
from byol_pytorch import BYOL
from torchvision import models
from dataset.ClassifierDataset import ClassifierDataset
#resnet = timm.create_model('resnet18', pretrained=True,in_chans=1,num_classes=10)
import numpy as np


resnet = models.resnet18(pretrained=True).to('cuda:0')
resnet.fc = torch.nn.Linear(512, 2)

learner = BYOL(
    resnet,
    image_size = 32,
    hidden_layer = 'avgpool'
).to('cuda:0')
dataPrefix="data/patchesIQ_small_shuffled/"
opt = torch.optim.Adam(learner.parameters(), lr=1e-3)
train_dataset = ClassifierDataset(dataPrefix, 'trainMB.h5', 'trainNoMB.h5',num_frames=16)
n  =len(train_dataset)
idxs = np.arange(n)
np.random.shuffle(idxs)
train_idx = idxs[:int(0.8*n)]
val_idx = idxs[int(0.8*n):]
batch_size = 1024

def sample_unlabelled_images():
    batch = []
    for _ in range(batch_size):
        #sample a random image from the training dataset
        idx = np.random.choice(train_idx)
        image = train_dataset[idx][0]
        image = image.repeat(3,1,1)
        batch.append(image)
    #retrieve next sample from the dataloader
    return torch.stack(batch)

def sample_unlabelled_validation_images() :
    batch = []
    for _ in range(batch_size):
        # sample a random image from the training dataset
        idx = np.random.choice(val_idx)
        image = train_dataset[idx][0]
        image = image.repeat(3, 1, 1)
        batch.append(image)
    # retrieve next sample from the dataloader
    return torch.stack(batch)


for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images.to('cuda:0'))
    val_images = sample_unlabelled_validation_images()
    val_loss = learner(val_images.to('cuda:0'))
    print(f"train loss : {loss}, Validation loss : {val_loss}")
    print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(resnet.state_dict(), f'./improved-netBoth.pt')