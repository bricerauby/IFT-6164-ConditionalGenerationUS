import numpy as np
import os
import timm.utils.metrics

from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score
import inspect
from PytorchTemplate import names
import tqdm
import torch
import json
class Metrics:
    def __init__(self,train_loader,debug=False):

        inception = timm.create_model("inception_v3",pretrained=True,num_classes=len(names)).eval()
        inception.load_state_dict(
            torch.load("models_weights/inception_v3.pt"))  # TODO : load the pretrained model trained on
        if torch.__version__>"2.0" and not debug and False :
            inception = torch.compile(inception)
        self.inception = inception
        
        # the "conditionnal data

        if os.path.exists(f"inception_stats_{train_loader.dataset.__class__.__name__}.json") :
            with open(f"inception_stats_{train_loader.dataset.__class__.__name__}.json") as f :
                self.features_map = json.load(f)
        else :
            self.inception = self.inception.to("cuda:0")
            self.features_map = {str(i) : [] for i in range(0,10)}
            with torch.no_grad() :
                for ex,(images,labels) in enumerate(tqdm.tqdm(train_loader)) :
                    images = images.to("cuda:0")
                    features = self.inception.forward_features(images)

                    features = features.detach().cpu()
                    for feature,label in zip(features,labels) :
                        self.features_map[str(label.item())].append(feature)
                    if debug and ex == 100:
                        break

                for key,value in self.features_map.items() :
                    value = torch.stack(value)
                    self.features_map[key] = (torch.mean(value,dim=[0,1]).tolist(),torch.std(value,dim=[0,1]).tolist()) #TODO : std or cov???


            json_object = json.dumps(self.features_map)
            with open(f"inception_stats_{train_loader.dataset.__class__.__name__}.json", "w") as outfile:
                outfile.write(json_object)

            self.inception = self.inception.cpu()
    @torch.no_grad()
    def FID(self, image, cond):


        self.inception = self.inception.to("cuda:0")
        features = self.inception.forward_features(image)

        fids = []
        for feature,c in zip(features,cond) :

            mu2, sigma2 = self.features_map[str(int(c.item()))]

            mu2 = torch.tensor(mu2).to("cuda:0")
            sigma2 = torch.tensor(sigma2).to("cuda:0")
            mu1 = torch.mean(feature,dim=[0,])
            sigma1 = torch.std(feature,dim=[0,])

            covmean = torch.cov(sigma1.T @ sigma2)**.5
            print(mu1.shape, mu2.shape, sigma1.shape, sigma2.shape, covmean.shape)
            fid = (torch.mean((mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2) + torch.trace(
                sigma1 + sigma2 - 2.0 * covmean)).cpu().item()
            fids.append(fid)

        self.inception = self.inception.cpu()
        return np.mean(fids)


    def metrics(self) :

        dict = {
            "FID": self.FID,
        }

        return dict




if __name__=="__main__" :
    from torchvision import datasets,transforms
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(299),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    metrics = Metrics(train_loader=train_loader,debug=True).metrics()
    for i in np.random.randint(0,1000,100) :
        image,label = train_dataset[i]
        label = np.array([label])
        image = image.reshape(1,3,299,299).to("cuda:0")

        fid = metrics["FID"](image,label)
        print("FID : ",fid)
        print("Out of Distribution FID : ",metrics["FID"](image,(label+1)%10))