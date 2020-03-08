from dataset import ZindiInferenceDataset
import albumentations as A
from aug import RandomTileAug
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from model import ZindiModel
import torch
import pandas as pd
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="path to .pth file")
parser.add_argument("img_folder", type=str, help="path to image folfer")
parser.add_argument("output", type=str, help="path to save result")
args = parser.parse_args()

device = torch.device('cpu')
model = ZindiModel('resnet18')
model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'))['model_state_dict'])
model.to(device)


augs = A.Compose([
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.ShiftScaleRotate(),
    A.LongestMaxSize(256),
    A.PadIfNeeded(256, 256)
])

transforms = A.Compose([
    RandomTileAug(augs, 1, 1, 0),
    A.Normalize()
])


test_dataset = ZindiInferenceDataset(
    img_folder=args.img_folder,
    transform=transforms,
)

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=8,
    shuffle=False
)


test_count = 10
result = []
logits = defaultdict(int)


for test_index in range(test_count):
    print('Current test:', test_index + 1, 'of', test_count)
    model.eval()

    for step, batch in tqdm(enumerate(data_loader)):
        inputs = batch["image"]
        image_ids = batch["image_id"]
        inputs = inputs.to(device, dtype=torch.float)
        logits[step] += model(inputs).data.cpu()


for step, batch in tqdm(enumerate(data_loader)):
    inputs = batch["image"]
    image_ids = batch["image_id"]
    out_logits = logits[step] / test_count
    out_logits = F.softmax(out_logits, dim=1).numpy()

    for index, image_id in enumerate(image_ids):
        result.append([image_id] + list(out_logits[index]))


dataset = pd.DataFrame(data=result, columns=['ID', 'leaf_rust', 'healthy_wheat', 'stem_rust'])
dataset = dataset[['ID', 'leaf_rust', 'stem_rust', 'healthy_wheat']]
dataset.to_csv(args.output, index=False)
