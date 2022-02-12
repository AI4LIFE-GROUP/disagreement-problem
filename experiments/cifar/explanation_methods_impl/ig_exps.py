import torch
import torch.nn.functional as F

from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum.attr import LimeBase, KernelShap
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from torchinfo import summary
import os
import json
from captum.attr import IntegratedGradients
import time
from torchvision.models import resnet18
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from captum.attr._core.lime import get_exp_kernel_similarity_function

# from PIL import Image
# import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = resnet18(pretrained=True).to(device)
resnet = resnet.eval()
summary(resnet, input_size=(32, 3, 32, 32))



labels_path = os.getenv('HOME') + '/.torch/models/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = {idx: label for idx, [_, label] in json.load(json_data).items()}

voc_ds = VOCSegmentation(
    './VOC',
    year='2012',
    image_set='train',
    download=False,
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    target_transform=T.Lambda(
        lambda p: torch.tensor(p.getdata()).view(1, p.size[1], p.size[0])
    )
)

ig = IntegratedGradients(resnet)
attrs = []
start_time = time.time()
for i, (img, seg_img) in enumerate(voc_ds):
    img = img.cuda()
    outputs = resnet(img.unsqueeze(0))
    print(outputs.shape)
    output_probs = F.softmax(outputs, dim=1).squeeze(0)
    label_idx = output_probs.argmax().unsqueeze(0)
    baseline = 0

    attrs_IG = ig.attribute(img.unsqueeze(0), baseline, n_steps=300,
                            target=label_idx)
    attrs.extend(attrs_IG.to("cpu"))
    if len(attrs) % 128 == 0:
        print(len(attrs), "  Time taken :", time.time() - start_time)
        start_time = time.time()


attrs_ig = torch.stack([i[1] for i in attrs])
torch.save(attrs_ig, 'ig_exp.pt')



