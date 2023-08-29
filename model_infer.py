import os
import random
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import argparse
# from torchvision.transforms import Compose, Normalize
# from sklearn.model_selection import train_test_split

import numpy as np
# from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
# from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import copy

BASE_PATH = './'

SAVE_PATH = os.path.join(BASE_PATH, 'model_files/best-model-parameters-v0-576-c2000.pt')

# import tracemalloc
class SpecDataset(Dataset):
    def __init__(self, data, label2class, maxval, minval, normalize=False, do_label_trans = True):
        self.data = data
        self.labels = self.data.iloc[:, -1].map(label2class).values if do_label_trans else self.data.iloc[:, -1].values
        self.features = self.data.iloc[:, :-1].values
        self.normalize = normalize
        self.maxval, self.minval = maxval, minval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.unsqueeze(torch.tensor(self.features[idx], dtype=torch.float), 0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # feature = torch.tensor(self.features[idx], dtype=torch.float)
        # label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.normalize:
            #   feature = self.transform(feature)
            noise = torch.randn_like(feature) * 0.000
            feature = feature + noise
            feature = (feature - self.minval) / self.maxval
            # feature = (feature - self.mean) / self.std


        return feature, label

# @title
class Conv1dNet3(nn.Module):
    def __init__(self, base_filters, kernel_size, stride, n_classes):
        super(Conv1dNet3, self).__init__()

        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride


        self.conv1 = torch.nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=self.kernel_size,
                stride=self.stride)
        self.relu1 = nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=self.kernel_size, stride=2)
        self.conv2 = torch.nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=self.kernel_size,
                stride=self.stride)
        self.relu2 = nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=self.kernel_size, stride=2)
        self.conv3 = torch.nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=self.kernel_size,
                stride=self.stride)
        self.relu3 = nn.ReLU()
        self.maxpool3 = torch.nn.MaxPool1d(kernel_size=self.kernel_size, stride=2)

        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(15552, n_classes)
        self.final_sm = nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        out = self.maxpool1(self.relu1(self.conv1(out)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.maxpool3(self.relu3(self.conv3(out)))
        # out = self.final_relu(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        # out = self.final_sm(out)

        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for doing inference with the CNN model')
    parser.add_argument('--input_file', type=str, help='name of oringal input spectrum')


    args = parser.parse_args()

    # make model
    device_str = "cpu"
    device = torch.device("cpu")
    kernel_size = 5
    stride = 2
    n_block = 3
    downsample_gap = 1
    increasefilter_gap = 1

    model = Conv1dNet3(
        base_filters = 16,
        kernel_size = 5,
        stride = 1,
        n_classes = 241
    )
    model.to(device)

    summary(model, (1, 2000), device=device_str)


    data_path = f"temp_files/{args.input_file}_in.csv"
    obs_data_path = os.path.join(BASE_PATH, data_path)
    obs_data = pd.read_csv(obs_data_path)

    with open(os.path.join(BASE_PATH, 'model_files/label2class_mapping_c2000.json'), 'r') as json_file:
        label2class = json.load(json_file)

    with open(os.path.join(BASE_PATH, 'model_files/class2label_mapping_c2000.json'), 'r') as json_file:
        class2label = json.load(json_file)



    obs_data.iloc[:, -1] = obs_data.iloc[:, -1].apply(lambda s: str(int(s)))
    last_key = str(len(obs_data.columns)-1)
    print(len(obs_data))
    obs_data = obs_data[obs_data[last_key].map(label2class).notna()]
    print(len(obs_data))
    # data[data.iloc[:, -1].map(label2class).notna()]

    # print(obs_data.iloc[:, -1].map(label2class).values)

    all_obs_values = obs_data.iloc[:, :-1].values.tolist()
    print(len(all_obs_values))
    max_value = np.max(all_obs_values)
    min_value = np.min(all_obs_values)
    obs_set = SpecDataset(obs_data, label2class, max_value, min_value, normalize=True, do_label_trans=True)
    obs_loader = DataLoader(obs_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model.load_state_dict(torch.load(SAVE_PATH, map_location=torch.device('cpu')))

    prog_iter_obs = tqdm(obs_loader, desc=f"Testing data {data_path}", position=0, leave=True)
    all_pred_prob = []

    obs_loss = []
    obs_accs = []
    obs_ys = []
    correct = []
    incorrect = []

    # print("26502" in label2class)
    output_dict = dict()
    na_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_obs):
            input_x, labels = tuple(t.to(device) for t in batch)
            logits = model(input_x)


            # Compute the accuracy for current batch.
            for logit, lab in zip(logits, labels.to(device)):
                top_probs, top_indices = torch.topk(logit, k=3)
                # print(logit)
                confidence_scores = torch.sigmoid(logit)
                # print(f"molecule: {class2label[int(lab)]}, pred: {class2label[int(pred)]}, logit: {torch.max(confidence_scores, dim=0).values}")
                if not str(int(lab)) in label2class:
                    # print(label2class)
                    # print(f"mole_id: {lab}, pred: {int(pred)}, logit: {torch.max(confidence_scores, dim=0).values}")
                    # print(f"mole_id: {class2label[str(int(lab))]}, 1-pred: {class2label[str(int(top_indices[0]))]}, 2-pred: {class2label[str(int(top_indices[1]))]}, 1-prob: {top_probs[0]}, 2-prob: {top_probs[1]}")

                    na_count += 1
                # if int(lab) == int(pred):
                moleid = class2label[str(int(lab))]
                if not moleid in output_dict:
                    output_dict[moleid] = dict()
                if int(lab) == int(top_indices[0]):
                    correct.append(moleid)
                    output_dict[moleid]["correct"] = True
                else:
                    incorrect.append(moleid)
                    output_dict[moleid]["correct"] = False

                output_dict[moleid]["1-pred"] = class2label[str(int(top_indices[0]))]
                output_dict[moleid]["2-pred"] = class2label[str(int(top_indices[1]))]
                output_dict[moleid]["3-pred"] = class2label[str(int(top_indices[2]))]


            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # print(f"logits: {logits.argmax(dim=-1)}, labels: {labels.to(device)}")

            # Record the loss and accuracy.
            # obs_loss.append(loss.item())
            obs_accs.append(acc)

        # The average loss and accuracy for entire testing set is the average of the recorded values.
        # obs_loss = sum(obs_loss) / len(obs_loss)
        obs_acc = sum(obs_accs) / len(obs_accs)

        # Print the information.
        print(f"\n[ Testing: acc = {obs_acc:.5f} ]")
        print(f"{len(correct)} in total")
        print(f"correct: {correct}")
        print(f"incorrect: {incorrect}")
        print(f"na_count: {na_count}")

        # output_dict = {"correct": correct, "incorrect": incorrect}

        output_filename = f"temp_files/{args.input_file}_result.pkl"
        with open(output_filename, 'wb') as f:
            pickle.dump(dict(output_dict), f)
        print(f"Classification result written in {output_filename}.")
