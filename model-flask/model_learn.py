import json

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as tfs

import torch
import torch.nn as nn

from PIL import Image
from pathlib import Path
from torch.optim import lr_scheduler
from torchvision import models
from tqdm.autonotebook import trange
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid", font_scale=1.4)

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


def balance_classes(files):
    labels = sorted(set([path.parent.name for path in files]))
    classes_dict = {
        label: [file for file in files if file.parent.name == label] for label in labels
    }

    new_files = []
    min_count = len(files) / len(labels)
    for label in labels:
        if len(classes_dict[label]) < min_count:
            classes_dict[label] *= int(min_count // len(classes_dict[label]) + 1)

        for file in classes_dict[label]:
            new_files.append(file)

    return new_files


def load_sample(file):
    image = Image.open(file)
    image.load()
    return image


def imshow(image, title=None, plt_ax=None, default=False):
    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)

    plt_ax.imshow(image)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def fit(model, criterion, optimizer, scheduler, num_epochs=25):
    best_params = model.state_dict()
    best_accuracy = 0.0

    loss_history = {"train": [], "val": []}
    accuracy_history = {"train": [], "val": []}

    progress_bar = trange(num_epochs, desc="Epoch:")

    for _ in progress_bar:
        for mode in MODES:
            running_loss = 0.0
            running_corrects = 0
            processed_size = 0

            if mode == "train":
                model.train()
            else:
                model.eval()

            for X_batch, y_batch in dataloaders[mode]:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                if mode == "train":
                    optimizer.zero_grad()
                    logits = model(X_batch)
                else:
                    with torch.no_grad():
                        logits = model(X_batch)

                loss = criterion(logits, y_batch)
                predictions = torch.argmax(nn.functional.softmax(logits), -1)

                if mode == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += int(torch.sum(predictions == y_batch))
                processed_size += len(predictions)

            epoche_accuracy = running_corrects / processed_size
            accuracy_history[mode].append(epoche_accuracy)
            loss_history[mode].append(running_loss / processed_size)

            progress_bar.set_description('{} Acc: {:.4f}'.format(
                mode, epoche_accuracy
            ))
            if mode == "train":
                scheduler.step()

            if mode == "val" and epoche_accuracy > best_accuracy:
                best_params = model.state_dict()
                best_accuracy = epoche_accuracy

    return model, loss_history, accuracy_history, best_params


class Simpsons(Dataset):
    def __init__(self, files, mode):
        super().__init__()

        self.files = sorted(files)
        self.mode = mode
        self.len_ = len(self.files)
        self.label_encoder = LabelEncoder()

        IMAGE_SIZE = 224
        self.transforms = {
            "train": tfs.Compose([
                tfs.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                tfs.RandomHorizontalFlip(),
                tfs.RandomGrayscale(),
                tfs.ToTensor(),
                tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

            "val": tfs.Compose([
                tfs.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                tfs.ToTensor(),
                tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

            "test": tfs.Compose([
                tfs.Resize((256, 256)),
                tfs.CenterCrop(IMAGE_SIZE),
                tfs.ToTensor(),
                tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        X = load_sample(self.files[index])
        X = self.transforms[self.mode](X)

        if self.mode == 'test':
            return X
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return X, y


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_folder = Path("train/simpsons_dataset")
test_folder = Path("test/testset")

train_val_files = balance_classes(sorted(list(train_folder.rglob('*.jpg'))))
test_files = sorted(list(test_folder.rglob('*.jpg')))

train_val_labels = [path.parent.name for path in train_val_files]
train_files, val_files = train_test_split(train_val_files, test_size=0.25, stratify=train_val_labels, shuffle=True)

labels = sorted(set([path.parent.name for path in train_files]))

train_dataset = Simpsons(train_files, "train")
val_dataset = Simpsons(val_files, "val")

dataloaders = {
    "train": DataLoader(train_dataset, batch_size=64, shuffle=True),
    "val": DataLoader(val_dataset, batch_size=64)
}

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), sharey=True, sharex=True)
for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0, 1000))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(
        map(lambda x: x.capitalize(), val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    imshow(im_val.data.cpu(), title=img_label, plt_ax=fig_x)

MODES = ["train", "val"]

model = models.mobilenet_v3_large(pretrained=True)

layers_to_unfreeze = 16

for param in model.features[:-layers_to_unfreeze].parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(in_features=960, out_features=768),
    nn.BatchNorm1d(768),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(in_features=768, out_features=768),
    nn.BatchNorm1d(768),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(in_features=768, out_features=len(labels)),
)

model = model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model, loss_history, accuracy_history, params = fit(model, criterion, optimizer, scheduler, num_epochs=10)
model.load_state_dict(params)
model_scripted = torch.jit.script(model)
model_scripted.save('model_scripted.pt')

plt.figure(figsize=(12, 8))
plt.plot(loss_history['train'], label="train")
plt.plot(loss_history['val'], label="val")
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(accuracy_history['train'], label="train")
plt.plot(accuracy_history['val'], label="val")
plt.legend()
plt.show()
