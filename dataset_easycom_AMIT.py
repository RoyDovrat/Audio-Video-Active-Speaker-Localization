import pickle

import torch
import torchvision
from torch.utils.data import Dataset

import os
import pandas as pd
import time

import sys
sys.path.insert(1, "/home/dsi/amiteli/Master/Thesis/git/utils")
from utils import choose_cuda

class EasyComDataset(Dataset):
    def __init__(self, dir_data_audio, dir_data_video, dir_csv_table,
                 mode, label_col_name, context_frames, av_key, color_mode="RGB"):
        """
        --- CSV table: ----
        session: session number between 1-12,
        file: file number in each session,
        frame_num: most of the file had 1200 frames, numbered between 0-1199
        audio_path, video_path: path to the pickle data
        speakers_IDs: the speaker ID for the active speakers in the frame
        speakers_num: number of active speaker in the frame
        cartesian_coords, quater_rotation: the coordinates for the active speakers
        speakers_in_picture_IDs: the active speakers that are in the FOV of the video
        places_in_picture: head box
                you can use:
                    import ast
                    string = "[(275, 422, 453, 655), (1153, 566, 1311, 722)]"
                    lst = ast.literal_eval(string)
                    to convert back to list
        is_wearer_active:  bool value, if the glasses wearer is an active speaker
        -----------------

        Audio:
            - was downsampled to 16kHz
            - was normalized for each file.
            - was split to the same length of a single video frame
            - result in (6,800) data  (6 mics, 800 samples)
        Video:
            - was downsampled to (360,640)
            - was split into single frames
            - result in: (3,360,640) data

        Labels:
            - for csd - use speakers_num
            -

        """
        # Data folders
        self.dir_data_audio = dir_data_audio
        self.dir_data_video = dir_data_video

        # CSD's folder and path
        self.table_path     = dir_csv_table
        self.table = pd.read_csv(self.table_path)

        # Dataset mode:  Train, Val, Test
        self.mode             = mode

        # Choose the labels for the dataset
        self.label_col_name = label_col_name

        # Data type to return: 'a' (audio), 'v' (video), 'av' (audio+video)
        self.av_key = av_key

        # Choose number of context frames
        self.context_frames_pre  = context_frames[0]
        self.context_frames_post = context_frames[1]

        # Choose color mode: "grayscale", "RGB"
        self.color_mode = color_mode

        # Prints
        print(f"EasyCome dataset, mode: {self.mode}")
        print(f"Dataset's color mode: {self.color_mode}")
        print(f"Dataset's set with context frames [pre,post]: {context_frames}")
        print(f"There are :[{len(self.table.index)}] samples in the dataset") #???????
        print(f"-----------")

    def get_context_idx(self, idx):
        """
        For the given idx, and the number context frames needed
        we get the idx's which are needed to be loaded
        the return is a tuple of 2 lists pre and post

        The frames are indexed between 0-1199
        """
        # print(f"**** get_context_idx ****")
        # print(f"{idx=}")

        session   = self.table.loc[idx, "session"]
        file      = self.table.loc[idx, "file"]
        frame_num = self.table.loc[idx, "frame_num"]

        # print(f"{session=}")
        # print(f"{file=}")
        # print(f"{frame_num=}")

        df_session_file = self.table.loc[(self.table['session'] == session) & (self.table['file'] == file)]
        max_frame_num = df_session_file["frame_num"].max()

        # Pre-context frames
        idx_pre = []
        for i in range(1, self.context_frames_pre+1):
            if frame_num-i >= 0:
                frame_idx_pre = idx-i
                idx_pre.append(frame_idx_pre)

        # Post-context frame
        idx_post = []
        for j in range(1, self.context_frames_post+1):
            if frame_num+j <= max_frame_num:
                frame_idx_post = idx + j
                idx_post.append(frame_idx_post)

        idx_pre  = sorted(idx_pre)
        idx_post = sorted(idx_post)

        return idx_pre, idx_post

    def get_audio(self, idx, idx_pre, idx_post):
        relative_path = self.table.loc[idx, "audio_path"]
        path = f"{self.dir_data_audio}/{relative_path}"
        with open(path, 'rb') as handle:
            data = pickle.load(handle)

        n_mics    = data.shape[0]
        frame_len = data.shape[1]

        # ==== Load the pre-context frames =====
        data_pre = torch.tensor((), device=data.device)
        count_loaded = 0
        for frame_idx in sorted(idx_pre):
            count_loaded += 1
            relative_path = self.table.loc[frame_idx, "audio_path"]
            path = f"{self.dir_data_audio}/{relative_path}"
            with open(path, 'rb') as handle:
                frame_data = pickle.load(handle)
                data_pre = torch.cat((data_pre,frame_data), 1)
        if count_loaded < self.context_frames_pre:
            num_pad = self.context_frames_pre - count_loaded
            zero_pad = torch.zeros((n_mics, frame_len*num_pad),
                                   device=data.device)
            data_pre = torch.cat((zero_pad, data_pre), 1)  # pad BEFORE the data_pre

        # ==== Load the post-context frames =====
        data_post = torch.tensor((), device=data.device)
        count_loaded = 0
        for frame_idx in sorted(idx_post):
            count_loaded += 1
            relative_path = self.table.loc[frame_idx, "audio_path"]
            path = f"{self.dir_data_audio}/{relative_path}"
            with open(path, 'rb') as handle:
                frame_data = pickle.load(handle)
                data_post = torch.cat((data_post, frame_data), 1)
        if count_loaded < self.context_frames_post:
            num_pad = self.context_frames_post - count_loaded
            zero_pad = torch.zeros((n_mics, frame_len*num_pad),
                                   device=data.device)
            data_post = torch.cat((data_post, zero_pad), 1)
        # ---------------------------------------
        # === Concat the pre-context and post-context frames ====
        data = torch.cat((data_pre, data, data_post), 1)
        # ---------------------------------------
        return data

    def get_video(self, idx, idx_pre, idx_post):
        relative_path = self.table.loc[idx, "video_path"]
        path = f"{self.dir_data_video}/{relative_path}"
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
            data = torch.permute(data, (2, 0, 1))  # To be: CxHxW
            if self.color_mode == "grayscale":
                data = torchvision.transforms.Grayscale(num_output_channels=1)(data)
            data = data.unsqueeze(0)  # adding time-axis

        T, C, H, W    = data.shape

        # ==== Load the pre-context frames =====
        data_pre = torch.tensor((), device=data.device)
        count_loaded = 0
        for frame_idx in sorted(idx_pre):
            count_loaded += 1
            relative_path = self.table.loc[frame_idx, "video_path"]
            path = f"{self.dir_data_video}/{relative_path}"
            with open(path, 'rb') as handle:
                frame_data = pickle.load(handle)
                frame_data = torch.permute(frame_data, (2, 0, 1)).unsqueeze(0)  # adding time-axis, 1XCxHxW
                if self.color_mode == "grayscale":
                    frame_data = torchvision.transforms.Grayscale(num_output_channels=1)(frame_data)
                data_pre = torch.cat((data_pre,frame_data), 0) # concat in the time-axis
        if count_loaded < self.context_frames_pre:
            num_pad = self.context_frames_pre - count_loaded
            zero_pad = torch.zeros((num_pad, C, H, W), device=data.device)
            data_pre = torch.cat((zero_pad, data_pre), 0)  # pad BEFORE the data_pre, in the time axis(=0)

        # ==== Load the post-context frames =====
        data_post = torch.tensor((), device=data.device)
        count_loaded = 0
        for frame_idx in sorted(idx_post):
            count_loaded += 1
            relative_path = self.table.loc[frame_idx, "video_path"]
            path = f"{self.dir_data_video}/{relative_path}"
            with open(path, 'rb') as handle:
                frame_data = pickle.load(handle)
                frame_data = torch.permute(frame_data, (2, 0, 1)).unsqueeze(0)  # adding time-axis, 1XCxHxW
                if self.color_mode == "grayscale":
                    frame_data = torchvision.transforms.Grayscale(num_output_channels=1)(frame_data)
                data_post = torch.cat((data_post, frame_data), 0)  # concat in the time-axis
        if count_loaded < self.context_frames_post:
            num_pad = self.context_frames_post - count_loaded
            zero_pad = torch.zeros((num_pad, C, H, W), device=data.device)
            data_post = torch.cat((data_post, zero_pad), 0)
        # ---------------------------------------
        # === Concat the pre-context and post-context frames ====
        data = torch.cat((data_pre, data, data_post), 0)  # concat in the time-axis (=0)
        # ---------------------------------------

        return data


    def __len__(self):
        return len(self.table.index)

    def __getitem__(self, idx):
        # Get labels
        if type(self.label_col_name) is list:   # is_wearer_active and place_in_picture
            labels = {}
            for col_name in self.label_col_name:
                labels[col_name] = self.table.loc[idx, col_name]
        else:
            labels = self.table.loc[idx, self.label_col_name]

        # Get the context frame idx's
        idx_pre, idx_post = self.get_context_idx(idx)

        # Get the data
        if self.av_key == "a":
            data = self.get_audio(idx, idx_pre, idx_post)
        if self.av_key == "v":
            data = self.get_video(idx, idx_pre, idx_post)
        if self.av_key == "av":
            data = {"audio": self.get_audio(idx, idx_pre, idx_post),
                    "video": self.get_video(idx, idx_pre, idx_post)}

        return data, labels


if __name__ == "__main__":
    cuda_num = 1
    device = choose_cuda(cuda_num)

    # Define a dataset
    dir_data_audio = "/dsi/scratch/from_netapp/users/elnatan_k/Audio"
    dir_data_video = "/dsi/scratch/from_netapp/users/roy_do/Video"
    dir_csv_table = "/dsi/scratch/from_netapp/users/elnatan_k/Table.csv"
    mode = "train"
    label_col_name = "places_in_picture"  # speakers_num, places_in_picture
    context_frames  = [4, 4]
    av_key = "av"  # a, v, av
    color_mode = "RGB"  # "RGB", "grayscale"
    ds = EasyComDataset(dir_data_audio=dir_data_audio,
                        dir_data_video=dir_data_video,
                        dir_csv_table=dir_csv_table,
                        mode=mode,
                        label_col_name=label_col_name,
                        context_frames=context_frames,
                        av_key=av_key,
                        color_mode=color_mode)

    # --- Test specific index ---
    # ds.__getitem__(idx=381827-776)
    # ---------------------

    # Define a dataloader
    batch       = 32
    shuffle     = True
    num_workers = 10
    dataloader_mode = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers)

    time_start = time.time()
    for batch_idx, all_data in enumerate(dataloader_mode):
        if batch_idx % 100 == 0:
            print(f"{batch_idx=}")
        data, labels = all_data
        # labels = labels.to(device)
        # print(f"{labels.shape=}")
        print(f"{ labels=}")

        if av_key == "a":
            data_audio = data.to(device)
            # print(f"{data_audio.shape=}")
        elif av_key == "v":
            data_video = data.to(device)
            # print(f"{data_video.shape=}")
        elif av_key == "av":
            data_audio = data[0].to(device)
            data_video = data[1].to(device)
            # print(f"{data_audio.shape=}, {data_video.shape=}")
        break
    time_end = time.time()
    print(f"1 Epoch took: {(time_end-time_start):.3f}")
