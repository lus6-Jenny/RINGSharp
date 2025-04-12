# Model from "Self-supervising Fine-grained Region Similarities for Large-scale Image Localization" - https://arxiv.org/abs/2006.03926
# Parts of this code are from https://github.com/cvg/Hierarchical-Localization

import torch
import torchvision.transforms as tfm

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


class SFRSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True).eval()

        self.un_normalize = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.normalize = tfm.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                       std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])

    def forward(self, batch):
        images = batch["img"]
        if images.dim() == 5:
            B, S, C, H, W = images.shape
            images = images.view(B*S, C, H, W)
        images = self.normalize(self.un_normalize(images))
        descriptors = self.net(images)
        return {'global': descriptors}
    
    def print_info(self):
        for name, param in self.net.named_parameters():
            print(name, param.requires_grad)        
        print('Model class: SFRS')
        n_params = sum([param.nelement() for param in self.net.parameters()])
        print('Total parameters: {}'.format(n_params))
