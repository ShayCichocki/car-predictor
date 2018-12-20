import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms

from helpers.helpers import train, test

MODEL_LOCATION = './trained-models/model_cars.pt'
DATASET_LOCATION = "./car_data/train"
TESTSET_LOCATION = "./car_data/test"

n_epochs = 50
learning_rate = 0.01
valid_size = 500

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize(size=(299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

v_t_transform = transforms.Compose([
    transforms.Resize(size=(299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_set = datasets.ImageFolder(
    root=DATASET_LOCATION,
    transform=transform
)
train_size = full_set.__len__() - valid_size

train_data, valid_data = torch.utils.data.random_split(full_set, [train_size, valid_size])

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_data = datasets.ImageFolder(TESTSET_LOCATION, transform=v_t_transform)

num_workers = 0

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=True)
data_transfer = {
    'train': train_data,
    'valid': valid_data,
    'test': test_data
}

loaders_transfer = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
}

model = models.inception_v3(pretrained=True)

classes = len(train_data.dataset.classes)
use_cuda = torch.cuda.is_available()

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, classes)
if use_cuda:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

torch.cuda.empty_cache()

model = train(n_epochs, loaders_transfer, model,
              optimizer, criterion, use_cuda, MODEL_LOCATION)


model.load_state_dict(torch.load(MODEL_LOCATION))
test(loaders_transfer, model, criterion, use_cuda)
