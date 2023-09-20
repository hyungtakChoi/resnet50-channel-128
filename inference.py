from torchvision import models
import torch
import torch.nn as nn
import poptorch
import custom_model
class ClassificationModel(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.model = custom_model.ResNet50()
        self.criterion = criterion

    def forward(self, x, labels):
        output = self.model(x.half())
        if labels is None:
            return output
        loss = self.criterion(output, labels.squeeze(dim=-1))
        return output, poptorch.identity_loss(loss, reduction='sum')


model = ClassificationModel(nn.CrossEntropyLoss())

model.half()
model.eval()
test = torch.randn(1,128,24,24).half()
print(model)
poptorch_model = poptorch.inferenceModel(model)
print(poptorch_model(test, None))