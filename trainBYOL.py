import torch
from byol_pytorch import BYOL
from torchvision import models
from dataset.GanDataset import GanDataset
#resnet = timm.create_model('resnet18', pretrained=True,in_chans=1,num_classes=10)


for dataset in ["MB","NoMB"] :
    resnet = models.resnet18(pretrained=True).to('cuda:0')

    learner = BYOL(
        resnet,
        image_size = 32,
        hidden_layer = 'avgpool'
    ).to('cuda:0')
    dataPrefix="data/"
    opt = torch.optim.Adam(learner.parameters(), lr=1e-3)
    train_dataset = GanDataset(dataPrefix, f'train{dataset}.h5',label=1, num_frames=16)
    n  =len(train_dataset)
    batch_size = 1024

    def sample_unlabelled_images():
        batch = []
        for _ in range(batch_size):
            idx = torch.randint(0,n,(1,))
            image = train_dataset[idx][0]
            image = image.repeat(3,1,1)
            batch.append(image)
        #retrieve next sample from the dataloader
        return torch.stack(batch)

    for _ in range(100):
        images = sample_unlabelled_images()
        loss = learner(images.to('cuda:0'))
        print(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder

    # save your improved network
    torch.save(resnet.state_dict(), f'./improved-net{dataset}.pt')