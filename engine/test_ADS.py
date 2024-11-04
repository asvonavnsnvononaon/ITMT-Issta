
import os
import glob
#from skimage.io import imread
#from skimage.transform import resize
import json
import os
import re
import pandas as pd
import numpy
import torch
import engine.ADS_model as auto_models
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import glob
#from skimage.io import imread
#from skimage.transform import resize
import json
import tqdm
from openpyxl import Workbook
import pandas as pd
import os
import re
import pandas as pd
import numpy
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import ConcatDataset
def get_model(model_name, device,Use_states,Pred_mode):
    model_classes = {
        'Epoch': auto_models.Epoch,
        'Resnet101': auto_models.Resnet101,
        'Vgg16': auto_models.Vgg16,
        'EgoStatusMLPAgent':auto_models.EgoStatusMLPAgent,
        'PilotNet':auto_models.PilotNet,
        'CNN_LSTM': auto_models.CNN_LSTM,
        'Weiss_CNN_LSTM': auto_models.Weiss_CNN_LSTM,
        'CNN_3D': auto_models.CNN_3D
    }
    if model_name in model_classes:
        model = model_classes[model_name](Use_states=Use_states,Pred_mode=Pred_mode).to(device)
        return model
    else:
        raise ValueError(f"Model '{model_name}' not found.")


def pre_load(dataset, batch_size,shuffle):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    all_images = []
    all_states = []
    all_labels = []
    for images, states, labels in loader:
        all_images.append(images)
        all_states.append(states)
        all_labels.append(labels)
    all_images = torch.cat(all_images, dim=0)
    all_states = torch.cat(all_states, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    new_dataset = TensorDataset(all_images, all_states, all_labels)

    loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
    return loader


def prepare_data(args,data_dir, save_path_):
    if args.Use_time_series == False:
        datasets = Get_Dataset(args,data_dir, save_path_)
    else:
        datasets = Get_Dataset_series(args,data_dir, save_path_)
    return datasets

def EXP3_test_ads(args,data_dir, save_path_):
    Test_results = []
    args.Use_time_series = True
    model_name = "CNN_LSTM"
    device = "cuda"
    use_state = args.Use_vehicle_states
    pred_mode = "steering"
    datasets = prepare_data(args, data_dir, save_path_)
    model = get_model(model_name, device, use_state, pred_mode)
    save_path = os.path.join("models", f"{model_name}_{pred_mode}_{str(int(use_state))}.pth")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    pred_mode_1 = "speed"
    model_1 = get_model(model_name, device, use_state, pred_mode_1)
    save_path_1 = os.path.join("models", f"{model_name}_{pred_mode_1}_{str(int(use_state))}.pth")
    model_1.load_state_dict(torch.load(save_path_1))
    model_1.eval()

    input_loader = pre_load(datasets, batch_size=128, shuffle=False)



    all_outputs = []
    all_outputs_1 = []
    with torch.no_grad():
        for images, states, labels in input_loader:
            images = images.to(device)
            states = states.to(device)
            labels = labels.to(device)
            outputs, label_out = model(images, states, labels)
            outputs_1, label_out = model_1(images, states, labels)
            if all_outputs == []:
                all_outputs = outputs
                all_outputs_1 = outputs_1
            else:
                all_outputs = torch.cat((all_outputs, outputs), dim=0)
                all_outputs_1 = torch.cat((all_outputs_1, outputs_1), dim=0)

    torch.cuda.empty_cache()
    data = []
    for idx in range(len(all_outputs)):
        prediction_data = {
            "index": idx,
            "model": model_name,
            "steering": float(all_outputs[idx][0]),  # 直接获取数值
            "speed": float(all_outputs_1[idx][0])  # 直接获取数值
        }
        data.append(prediction_data)

    return data


def test_ads(args,data_dir, save_path_):
    args.Use_time_series = False
    models = ["Resnet101", "Vgg16", "Epoch", "PilotNet"]

    Test_results = []
    device = "cuda"
    use_state = args.Use_vehicle_states
    pred_mode = args.pre_model
    datasets = prepare_data(args,data_dir, save_path_)
    input_loader = pre_load(datasets, batch_size=128, shuffle=False)

    for model_name in models:
        all_outputs = []
        all_labels = []
        model = get_model(model_name, device, use_state, pred_mode)
        save_path = os.path.join("models", f"{model_name}_{pred_mode}_{str(int(use_state))}.pth")
        model.load_state_dict(torch.load(save_path))
        model.eval()
        total_items = len(input_loader)
        with torch.no_grad():
            for images, states, labels in input_loader:

                images = images.to(device)
                states = states.to(device)
                labels = labels.to(device)
                outputs, label_out = model(images, states, labels)
                if all_outputs == []:
                    all_outputs = outputs
                    all_labels = label_out
                else:
                    all_outputs = torch.cat((all_outputs, outputs), dim=0)
                    all_labels = torch.cat((all_labels, label_out), dim=0)

        torch.cuda.empty_cache()
        data={
            "model":model_name,
            "output":all_outputs,
            "label":all_labels
        }
        Test_results.append(data)
    args.Use_time_series = True
    models = ["CNN_LSTM", "CNN_3D"]
    device = "cuda"
    use_state = args.Use_vehicle_states
    pred_mode = args.pre_model
    datasets = prepare_data(args, data_dir, save_path_)
    input_loader = pre_load(datasets, batch_size=128, shuffle=False)
    for model_name in models:

        all_outputs = []
        all_labels = []
        model = get_model(model_name, device, use_state, pred_mode)
        save_path = os.path.join("models", f"{model_name}_{pred_mode}_{str(int(use_state))}.pth")
        model.load_state_dict(torch.load(save_path))
        model.eval()
        with torch.no_grad():
            for images, states, labels in input_loader:

                images = images.to(device)
                states = states.to(device)
                labels = labels.to(device)
                outputs, label_out = model(images, states, labels)
                if all_outputs == []:
                    all_outputs = outputs
                    all_labels = label_out
                else:
                    all_outputs = torch.cat((all_outputs, outputs), dim=0)
                    all_labels = torch.cat((all_labels, label_out), dim=0)

        torch.cuda.empty_cache()
        data = {
            "model": model_name,
            "output": all_outputs,
            "label": all_labels
        }
        Test_results.append(data)


    return Test_results
def get_violation(args,data_dir,original_,follow_up_,maneuver):
    Test_results = []
    if maneuver=="slow down":
        alpha,beta = -0.3,-0.05
        pre_models = ["speed"]
    elif maneuver=="stop":
        alpha, beta = -0.5, -0.1
    elif maneuver=="turn right":
        alpha, beta = 0.05,0.3
        pre_models = ["steering"]
    elif maneuver=="turn left":
        alpha, beta = -0.3,-0.05
        pre_models = ["steering"]
    elif maneuver=="keep the same":
        alpha, beta = -0.1, 0.1
        pre_models = ["steering"]
    original = Test(args, data_dir, original_, pre_models)
    follow_up = Test(args, data_dir, follow_up_, pre_models)

    for i in range(len(original)):
        model_name = original[i]['model']
        original_ = original[i]['output']
        follow_up_ = follow_up[i]['output']
        violation_tests = count_violations(original_,follow_up_,alpha,beta)
        data = {
            "model": model_name,
            "violations": violation_tests
        }
        Test_results.append(data)
    return Test_results


def count_violations(original,follow_up,alpha,beta):
    original = torch.clone(original)
    violation_tests=0
    for i in range(len(original)):
        if original[i][0] != 0:
            change = (follow_up[i][0] - original[i][0]) / original[i][0]
        else:
            change = follow_up[i][0]
        violation = not (alpha <= change <= beta)
        if violation:
            violation_tests += 1
    return violation_tests

def save_data_lists_to_json(data_lists, file_path):
    existing_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        except json.JSONDecodeError:
            pass
    if isinstance(existing_data, list):
        existing_data.extend(data_lists)
    else:
        existing_data = data_lists
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

class Get_Dataset(Dataset):
    def __init__(self, args,data_dir, save_path_, size_control="resize"):
        self.data_dir = save_path_
        self.size_control = size_control
        self.matched_data = pd.read_excel(os.path.join(data_dir, "matched_data.xlsx"), sheet_name="Matched Data")
        self.sequence_length = 25
    def __len__(self):
        return len(self.matched_data)-2+1

    def __getitem__(self, idx):
        if (idx + 1) % 25==0:
            return None,None,None


        matched = self.matched_data.iloc[idx:idx + 2]
        sequence = []
        prev_sequence = []
        cont = 0
        for _, row in matched.iterrows():
            data = [ row['Steering Angle'],row['Vehicle Speed']]
            if cont < 1:
                image_path = os.path.join(self.data_dir, os.path.basename(row['Image File']))
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                if self.size_control=="resize":
                    current_height, current_width = img.shape[:2]
                    if (current_width, current_height) != (320, 160):
                        img = cv2.resize(img, (320, 160), interpolation=cv2.INTER_AREA)

                img = img.astype(numpy.float32) / 127.5 - 1.0
                prev_sequence = data
            else:
                sequence= data
            cont = cont + 1

        img_sequence = torch.from_numpy(numpy.array(img)).float().permute(2, 0, 1)

        sequence = torch.tensor(sequence).float()
        prev_sequence = torch.tensor(prev_sequence).float()
        return img_sequence, prev_sequence, sequence


def custom_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item[0] is not None]

    # Separate the items in the batch
    img_sequences, prev_sequences, sequences = zip(*batch)

    # Stack the tensors
    img_sequences = torch.stack(img_sequences)
    prev_sequences = torch.stack(prev_sequences)
    sequences = torch.stack(sequences)

    return img_sequences, prev_sequences, sequences


class Get_Dataset_series(Dataset):
    def __init__(self, args,data_dir, save_path_, size_control="resize"):
        self.data_dir = save_path_
        self.size_control = size_control
        self.matched_data = pd.read_excel(os.path.join(data_dir, "matched_data.xlsx"), sheet_name="Matched Data")

    def __len__(self):
        return len(self.matched_data) - 5 + 1

    def __getitem__(self, idx):
        if any((i % 25 == 0) for i in range(idx + 1, idx + 5)):
            return None,None,None
        matched = self.matched_data.iloc[idx:idx + 5]

        img_sequence = []
        sequence = []
        prev_sequence = []
        cont = 0
        for _, row in matched.iterrows():
            data = [row['Steering Angle'],row['Vehicle Speed']]
            image_path = os.path.join(self.data_dir, os.path.basename(row['Image File']))
            if cont < 4:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                if self.size_control=="resize":
                    current_height, current_width = img.shape[:2]
                    if (current_width, current_height) != (320, 160):
                        img = cv2.resize(img, (320, 160), interpolation=cv2.INTER_AREA)
                img = img.astype(numpy.float32) / 127.5 - 1.0
                img_sequence.append(img)
                prev_sequence.append(data)
            else:
                sequence.append(data)
            cont = cont + 1
        img_sequence = torch.from_numpy(numpy.array(img_sequence)).float().permute(0, 3, 1, 2)

        sequence = torch.tensor(sequence).float()
        prev_sequence = torch.tensor(prev_sequence).float()
        return img_sequence, prev_sequence, sequence

def Test(args,data_dir, save_path_,pre_models):
    #,"steering"]  # [0, 1] ,"steering" ,"steering"
    for pre_model in pre_models:

        Test_results = test_ads(args,data_dir, save_path_)
    return Test_results




"""

plt = 1
    if plt==True:
        import matplotlib.pyplot as mpl
        original_data = outputs.detach().cpu().numpy()
        label_data =label_out.detach().cpu().numpy()

        mpl.plot(original_data[:, 0], label='pred', color='blue')
        mpl.plot(label_data[:, 0], label='Original', color='red')
        plt.show()
"""