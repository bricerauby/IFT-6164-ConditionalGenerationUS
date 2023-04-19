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

        inception = timm.create_model("resnet18",pretrained=True,num_classes=len(names),in_chans=1).eval()
        if torch.__version__>"2.0" and not debug :
            inception = torch.compile(inception)
        modelPath = 'checkpoint/65241799_overseas_square_1733'

        self.inception = inception
        
        # the "conditionnal data

        if os.path.exists(f"inception_stats_{train_loader.dataset.__class__.__name__}.json") :
            with open(f"inception_stats_{train_loader.dataset.__class__.__name__}.json") as f :
                self.features_map = json.load(f)
        else :
            self.inception = self.inception.to("cuda:0")
            self.features_map = {str(i) : [] for i in range(len(names))}

            print(f"Computing the mean and std of the features of the dataset with {len(train_loader.dataset)} samples")
            with torch.no_grad() :
                for ex,(images,labels) in enumerate(tqdm.tqdm(train_loader)) :
                    images = images.to("cuda:0")
                    features = self.inception.forward_features(images)

                    features = features.detach().cpu()
                    for feature,label in zip(features,labels) :


                        self.features_map[str(label.item())].append(feature.squeeze()[None,:])
                    if debug and ex == 100:
                        break

                for key,value in self.features_map.items() :
                    print(value[0].shape)
                    value = torch.stack(value)
                    mean, std = (torch.mean(value, dim=0).tolist(), torch.std(value, dim=0).tolist())

                    self.features_map[key] = (mean,std) #TODO : std or cov???


            json_object = json.dumps(self.features_map)
            with open(f"inception_stats_{train_loader.dataset.__class__.__name__}.json", "w") as outfile:
                outfile.write(json_object)

            self.inception = self.inception.cpu()
    @torch.no_grad()
    def FID(self, image, cond):


        self.inception = self.inception.to("cuda:0")
        features = self.inception.forward_features(image).squeeze()[None,:]

        fids = []
        for feature,c in zip(features,cond) :

            mu2, sigma2 = self.features_map[str(int(c.item()))]

            mu2 = torch.tensor(mu2).to("cuda:0")
            sigma2 = torch.tensor(sigma2).to("cuda:0")
            mu1 = features
            sigma1 = torch.eye(sigma2.shape[0]).to("cuda:0") #TODO : SKETCHY

            covmean = torch.cov(sigma1.T @ sigma2)**.5

            try :
                fid = (torch.mean((mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2) + torch.trace(
                    sigma1 + sigma2 - 2.0 * covmean)).cpu().item()
            except :
                print(mu1.shape, mu2.shape, sigma1.shape, sigma2.shape, covmean.shape)
                raise Exception("FID error")
            fids.append(fid)

        self.inception = self.inception.cpu()
        return np.mean(fids)


    def metrics(self) :

        dict = {
            "FID": self.FID,
        }

        return dict




if __name__=="__main__" :
    from dataset.ClassifierDataset import ClassifierDataset
    dataPrefix = "/mnt/f/IFT6164/data"
    train_dataset = ClassifierDataset(dataPrefix, 'trainMB.h5', 'trainNoMB.h5', num_frames=16)
    val_dataset = ClassifierDataset(dataPrefix, 'testMB.h5', 'testNoMB.h5', num_frames=16)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    metrics = Metrics(train_loader=train_loader,debug=True).metrics()
    for i in np.random.randint(0,1000,100) :
        image,label = train_dataset[i]
        label = np.array([label])
        image = image.to("cuda:0")

        fid = metrics["FID"](image[None,:,:,:],label)
        print("FID : ",fid)
        print("Out of Distribution FID : ",metrics["FID"](image[None,:,:,:],(label+1)%2))