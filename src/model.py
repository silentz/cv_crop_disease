import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model


class ZindiModel(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=3):
        super(ZindiModel, self).__init__()
        self.model = make_model(model_name=model_name, pretrained=pretrained, num_classes=1000)
        in_features = self.model._classifier.in_features
        self.head = nn.Linear(in_features, num_classes)

    def freeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.model._features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        logits = self.head(features)
        return logits
