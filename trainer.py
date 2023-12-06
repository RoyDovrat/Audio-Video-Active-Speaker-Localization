import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as TF
from torchinfo import summary
import segmentation_models_pytorch
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import time
import datetime
import pprint
import os
from sklearn.metrics import classification_report, confusion_matrix
from av_model import av_model
from dataset_easycom_AMIT import EasyComDataset

import sys
from utils import choose_cuda, epoch_time, calc_metrics

EPS = sys.float_info.epsilon
PRINT_FACTOR = 100
loss_print_factor = 1000


class Trainer(object):
    def __init__(self, config, load_trained_model=False, checkpoint=None):  # , wandb_run_name=""):
        """Initialize configurations"""
        self.config = config
        self.load_trained_model = load_trained_model
        self.checkpoint = checkpoint
        # self.wandb_run_name = wandb_run_name

        self.av_key = self.config['av_key']
        self.num_epochs = self.config['num_epochs']
        self.epoch = 0
        if self.load_trained_model:
            self.epoch = self.checkpoint['epoch'] + 1
        self.num_epochs_wearer = self.config['num_epochs_wearer']
        self.num_classes = self.config['num_classes']

        # --- All 'set's for the training ---
        self.set_all()
        # ----------------------------------

    def set_all(self):
        print(f"==== Start with all set() methods before start training ===")
        start = time.time()
        # self.is_first_epoch = True

        print(f"==========================")
        self.set_trainer_path()
        self.save_config_file()
        self.set_dataset_path()
        self.set_device()
        self.set_dataset()
        self.set_audio_params()
        self.set_video_params()
        self.set_dataloader()
        self.set_model()
        self.set_loss()
        self.set_optimizer()
        self.set_DataParallel()

        print(f"==========================")

        end = time.time()
        print(f"Finish all set() methods, took:{(end - start):3f}[sec]")
        print(f"==========================")

    # ----- All set() methods -------
    def set_trainer_path(self):
        # set saving name
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y_%m_%d__%H_%M")  # Include year, month, day, hour, and minute
        str_model_details = ''
        if self.config['choose_model'] == "av_model":
            str_model_details = f"{self.config['av_combination_level']}"
        if 'only_wearer_mode' in self.config.keys():
            if self.config['only_wearer_mode']:
                str_model_details += '_only_wearer_mode'

        str_loss_params = ''
        for param_name, param_val in self.config['loss_params'].items():
            str_loss_params += f"{param_name}_{param_val}_"


        if not self.load_trained_model:
            # {self.config['data_name']}_  _{self.config['choose_model']}
            self.save_name = f"{dt_string}_LossType_{self.config['loss_type']}_{str_loss_params}" \
                             f"unet_size_{self.config['unet_size']}_" \
                             f"{str_model_details}_" \
                             f"n_epochs_{self.config['num_epochs']}_" \
                             f"B_{self.config['batch_train']}_" \
                             f"color_mode_{self.config['color_mode']}"
        else:  # self.load_trained_model:
            self.save_name = self.checkpoint['save_name']

        print(f"save_name:\n{self.save_name}")

        # ----- Folder for saving plots etc. ---------------------------------
        self.dir_root = "/dsi/scratch/from_netapp/users/elnatan_k"
        self.dir_results = f"{self.dir_root}/new_results"
        self.dir_model_folder = f"{self.dir_results}/{self.save_name}"
        self.dir_fig = f"{self.dir_model_folder}/figures"
        self.dir_models = f"{self.dir_model_folder}/model"
        # -------------------------------------------------------

        folders = [self.dir_root, self.dir_results, self.dir_model_folder, self.dir_fig, self.dir_models]
        # Create the folders
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def save_config_file(self):
        # Open a file for writing
        save_path = f"{self.dir_model_folder}/trainer_config.txt"
        with open(save_path, 'w') as file:
            for k, v in self.config.items():
                file.write(str(k) + ' : \n ' + str(v) + '\n\n')
            file.close()
        file.close()

    def set_dataset_path(self):
        # --- datastes Paths ----
        if self.config["data_name"] == "easycom":
            self.dir_data_audio = "/dsi/scratch/from_netapp/users/elnatan_k/Audio"
            self.dir_data_video = f"/dsi/scratch/from_netapp/users/roy_do/Video_{self.config['downsample_ratio']}_ratio"
            self.dir_csv_table = "/dsi/scratch/from_netapp/users/elnatan_k"  # /Table.csv"
        else:
            print("!ERROR!")

    def set_device(self):
        self.device = choose_cuda(self.config["cuda_num"])

    def set_dataset(self):
        print("==== start setting the dataset ===")
        start = time.time()
        # === Set the datasets ====
        if self.config["data_name"] == "easycom":
            print("set EasyCom dataset dataset and loaders")
            if 'test_one_batch_mode' in self.config.keys() and self.config['test_one_batch_mode']:
                dir_csv_table_train = f"{self.dir_csv_table}/one_batch_train.csv"
            elif self.config['visualize_mode']:
                dir_csv_table_train = self.config['dir_visualize_csv_table']
            elif self.config['train_on_speakers_table']:
                dir_csv_table_train = f"{self.dir_csv_table}/speakers_train_table.csv"
            else:
                dir_csv_table_train = f"{self.dir_csv_table}/Train_table.csv"

            self.dataset_train = EasyComDataset(dir_data_audio=self.dir_data_audio,
                                                dir_data_video=self.dir_data_video,
                                                dir_csv_table=dir_csv_table_train,
                                                mode="train",
                                                label_col_name=self.config['label_col_name'],
                                                context_frames=self.config['context_frames'],
                                                av_key=self.config['av_key'],
                                                color_mode=self.config['color_mode'])
            self.dataset_val = EasyComDataset(dir_data_audio=self.dir_data_audio,
                                              dir_data_video=self.dir_data_video,
                                              dir_csv_table=f"{self.dir_csv_table}/Validation_table.csv",
                                              mode="val",
                                              label_col_name=self.config['label_col_name'],
                                              context_frames=self.config['context_frames'],
                                              av_key=self.config['av_key'],
                                              color_mode=self.config['color_mode'])
            self.dataset_test = EasyComDataset(dir_data_audio=self.dir_data_audio,
                                               dir_data_video=self.dir_data_video,
                                               dir_csv_table=f"{self.dir_csv_table}/Test_table.csv",
                                               mode="test",
                                               label_col_name=self.config['label_col_name'],
                                               context_frames=self.config['context_frames'],
                                               av_key=self.config['av_key'],
                                               color_mode=self.config['color_mode'])

            self.len_train = len(self.dataset_train)
            self.len_val = len(self.dataset_val)
            self.len_test = len(self.dataset_test)
        else:
            print(f"!ERROR! NO dataset: {self.config['data_name']}")
        # ----------------------

        print(f'Number of train examples: {self.len_train}')
        print(f'Number of  val  examples: {self.len_val}')
        print(f'Number of test  examples: {self.len_test}')
        end = time.time()
        print(f"loading the dataset took:{(end - start):3f}[sec]")
        print("-------------------------------------------")

    def set_audio_params(self):
        self.sr = self.config['sr']
        self.num_audio_channels = self.config['num_audio_channels']

        self.n_fft = self.config['num_fft']
        self.n_fft_effective = self.config['n_fft_effective']

        self.win = torch.hann_window(window_length=self.n_fft, device=self.device)

    def set_video_params(self):
        self.fps = self.config['video_fps']

    def set_dataloader(self):
        print(" === start setting the dataloader ===")
        start = time.time()
        #  === Set the dataloaders: ====
        if self.config['visualize_mode']:
            shuffle_train = False
        else:
            shuffle_train = True
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train,
                                                            batch_size=self.config['batch_train'],
                                                            shuffle=shuffle_train,
                                                            num_workers=self.config['num_workers'])
        self.dataloader_val = torch.utils.data.DataLoader(self.dataset_val, batch_size=self.config['batch_val'],
                                                          shuffle=True, num_workers=self.config['num_workers'])
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.config['batch_test'],
                                                           shuffle=True, num_workers=self.config['num_workers'])
        # -----------------------------

        print(f'Number of batches in train: {len(self.dataloader_train)}')
        print(f'Number of batches in val  : {len(self.dataloader_val)}')
        print(f'Number of batches in test : {len(self.dataloader_test)}')
        end = time.time()
        print(f"setting the dataloader took:{(end - start):3f}[sec]")
        print("-------------------------------------------")

    def set_model(self):
        print("=== Start setting the model ===")

        start = time.time()
        if self.config['choose_model'] == "v1":
            print("v1 not supported yet!")

        elif self.config['choose_model'] == "av_model":
            # n_time = [4, 10, 16, 22][self.config["context_frames"][0]]
            if self.config['downsample_ratio'] == 6:
                n_time = 11
            else:  # self.config['downsample_ratio'] == 3
                n_time = 22
            self.data_param_audio = {
                'n_mics': self.config['num_audio_channels'],
                'n_fft': self.config['n_fft_effective'],
                'fs': self.config['sr'],
                'n_time': n_time
            }
            self.data_param_video = {
                'n_frames': self.config['context_frames'][0] + self.config['context_frames'][1] + 1,
                # center frame + (context frames pre+post)
                'n_channels': self.config['video_n_channels'],
                'frame_h': self.config['video_frame_h'],
                'frame_w': self.config['video_frame_w'],
                'downsample_ratio': self.config['downsample_ratio']
            }

            self.av_combination_level = self.config['av_combination_level']
            self.unet_size = self.config['unet_size']

            print(self.config)
            self.model = av_model(data_param_audio=self.data_param_audio,
                                  data_param_video=self.data_param_video,
                                  num_classes=self.num_classes,
                                  device=self.device,
                                  av_combination_level=self.av_combination_level,
                                  unet_size=self.unet_size,
                                  ).to(self.device)
            if self.load_trained_model:
                self.model.load_state_dict(self.checkpoint['model'])
            # self.print_model_summary()

        else:
            print(f"choose_model:{self.config['choose_model']}, Not supported")

        end = time.time()
        print(f"setting the model took:{(end - start):3f}[sec]")
        print("-------------------------------------------")

    def set_loss(self):
        print("=== Start setting the loss function ===")
        start = time.time()
        self.loss_params = self.config["loss_params"]
        self.loss_weight = None
        if self.config['loss_weight'] is not None:
            self.loss_weight = torch.tensor(self.config['loss_weight'])

        # -- Choose loss type ----
        print(f"loss_type: {self.config['loss_type']}")
        # print(f"weight_speaking_class:{self.config['loss_params']['weight_speaking_class']}")
        print(f"{self.config['loss_type']} params:\n{self.config['loss_params']}")

        if self.config['loss_type'] == "bce":
            pos_weight_val = self.config['loss_params']['pos_weight_wearer']
            if not pos_weight_val:
                pos_weight = None
            else:
                pos_weight = torch.tensor(self.config['loss_params']['pos_weight_wearer'])

            self.loss_fn_wearer = nn.BCEWithLogitsLoss(weight=self.loss_weight,
                                                       size_average=None,
                                                       reduce=None,
                                                       reduction='mean',
                                                       pos_weight=pos_weight).to(self.device)

            pos_weight_val = self.config['loss_params']['pos_weight_speaker_segmentation']
            if not pos_weight_val:
                pos_weight = None
            else:
                pos_weight = torch.tensor(self.config['loss_params']['pos_weight_speaker_segmentation'])

            self.loss_fn_speaker_segmentation = nn.BCEWithLogitsLoss(
                weight=self.config['loss_params']['weight'],
                # torch.tensor([1.0, self.config['loss_params']['weight_speaking_class']]).to("cpu"),
                size_average=None,
                reduce=None,
                reduction='mean',
                pos_weight=pos_weight).to(self.device)  # None # self.config['loss_params']['pos_weight']


        elif self.config['loss_type'] == "dc":
            self.loss_fn_wearer = segmentation_models_pytorch.losses.DiceLoss(mode="binary",
                                                                              classes=self.loss_params["classes"],
                                                                              log_loss=False,
                                                                              from_logits=True,
                                                                              smooth=0.0,
                                                                              ignore_index=None,
                                                                              eps=1e-07).to(self.device)
            self.loss_fn_speaker_segmentation = segmentation_models_pytorch.losses.DiceLoss(mode="binary",
                                                                                            classes=self.loss_params[
                                                                                                "classes"],
                                                                                            log_loss=False,
                                                                                            from_logits=True,
                                                                                            smooth=0.0,
                                                                                            ignore_index=None,
                                                                                            eps=1e-07).to(
                self.device).to(self.device)

        elif self.config['loss_type'] == "ftl":
            self.loss_fn_wearer = segmentation_models_pytorch.losses.TverskyLoss(mode="binary",
                                                                                 classes=self.loss_params["classes"],
                                                                                 log_loss=False,
                                                                                 from_logits=True,
                                                                                 smooth=0.0,
                                                                                 ignore_index=None,
                                                                                 eps=1e-07,
                                                                                 alpha=self.loss_params["alpha"],
                                                                                 beta=self.loss_params["beta"],
                                                                                 gamma=self.loss_params["gamma"]).to(
                self.device)
            self.loss_fn_speaker_segmentation = segmentation_models_pytorch.losses.TverskyLoss(mode="binary",
                                                                                               classes=self.loss_params[
                                                                                                   "classes"],
                                                                                               log_loss=False,
                                                                                               from_logits=True,
                                                                                               smooth=0.0,
                                                                                               ignore_index=None,
                                                                                               eps=1e-07,
                                                                                               alpha=self.loss_params[
                                                                                                   "alpha"],
                                                                                               beta=self.loss_params[
                                                                                                   "beta"],
                                                                                               gamma=self.loss_params[
                                                                                                   "gamma"]).to(
                self.device)

        else:
            print(f"!! ERROR!! loss function:{self.config['loss_type']}")

        end = time.time()
        print(f"setting the loss function took:{(end - start):3f}[sec]")
        print("-------------------------------------------")

    def set_optimizer(self):
        print("=== Start setting the optimizer function ===")
        start = time.time()

        if self.config['optim_type'] == "Adam":
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.config['optim_lr'],
                                        weight_decay=self.config['optim_weight_decay'])
        elif self.config['optim_type'] == "SGD":
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=self.config['optim_lr'],
                                       weight_decay=self.config['optim_weight_decay'],
                                       momentum=self.config['optim_momentum'])
        else:
            print(f"!! ERROR!! loss function:{self.config['optim_type']}")

        if self.load_trained_model:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        end = time.time()
        print(f"setting the optimizer took:{(end - start):3f}[sec]")
        print("-------------------------------------------")

    def set_DataParallel(self):
        print(f"--- set_DataParallel() ---")
        if self.config['MultiGPU'] is True:
            print(f"set_DataParallel with:{self.config['device_ids']=}")
            self.model = nn.DataParallel(self.model, device_ids=self.config['device_ids'])
        else:
            print(f"Using single device: {self.device}")

    # =======================

    # ===== Prepr methods ===========
    def prep_labels_wearer(self, label):
        # Turn the labels to a one-hot-encoding form of the labels
        label_one_hot = TF.one_hot(label.long(), num_classes=self.num_classes)
        return label_one_hot

    def prep_labels_places(self, label):
        labels_list = []

        for headboxes in label:
            new_label = torch.zeros((self.config["video_frame_h"], self.config["video_frame_w"]))

            for headbox in headboxes:
                crop_headbox = (headbox[0] - 200, headbox[1], headbox[2] - 200, headbox[3])
                downsampled = tuple(
                    int(x // self.config['downsample_ratio']) for x in crop_headbox)  # 3 -> self.downsample_ratio
                new_label[downsampled[0]: downsampled[2] + 1, downsampled[1]: downsampled[3] + 1] = 1

            # Turn the labels to a one-hot-encoding form of the labels
            new_label_one_hot = TF.one_hot(new_label.long(), num_classes=self.num_classes)
            new_label_one_hot = new_label_one_hot.permute(2, 0, 1)

            labels_list.append(new_label_one_hot)

        return torch.stack(labels_list, dim=0).long()

    def prep_audio(self, audio_data):
        return_data = 0

        if self.config["feature_type"] == "stft":
            batch = audio_data.shape[0]
            return_data = torch.tensor((), device=self.device)
            for i in range(0, batch):
                mic_stft = torch.stft(input=audio_data[i],
                                      n_fft=self.n_fft,
                                      hop_length=self.n_fft // 2,  # default: floor(n_fft / 4)
                                      win_length=self.n_fft,  # default: n_fft
                                      window=self.win,  # default: box
                                      center=True,  # default: True
                                      pad_mode='reflect',  # default: reflect
                                      normalized=False,  # default: False
                                      onesided=None,  # default: True (for real input&win)
                                      return_complex=True).unsqueeze(0)
                return_data = torch.cat((return_data, mic_stft), 0)

        elif self.config["feature_type"] == "mel":
            print(f"!ERROR! -- {self.config['feature_type']} NOT SUPPORTED YET")
        else:
            print(f"!ERROR! -- {self.config['feature_type']} NOT SUPPORTED")

        return return_data

    def prep_video(self, data_video):
        return data_video

    def prep_data(self, data_dict):
        if self.av_key == "a":
            data_dict['audio'] = self.prep_audio(data_dict['audio'])
        elif self.av_key == "v":
            data_dict['video'] = self.prep_video(data_dict['video'])
        elif self.av_key == "av":
            data_dict['audio'] = self.prep_audio(data_dict['audio'])
            data_dict['video'] = self.prep_video(data_dict['video'])
        else:
            print(f"!ERROR! : {self.av_key=}")

        return data_dict

    # ------------------------------

    def get_classification_report(self, labels_true, pred_logits, mode, index=0):
        # Get classification report for each classification task (wearer, num_active , etc)
        for key in labels_true.keys():
            print(f"--- Get classification report for mode:{mode} and task: {key} ---")
            self.get_classification_report_single_task(labels_true[key], pred_logits[key], mode, task=key, index=index)
            print("=========================================================")

    def get_classification_report_single_task(self, labels_true, pred_logits, mode, task, index=0):
        """
        labels_true: tensor with all the true labels of the classification task
        pred_logits: tensor with all the predicted logits of the classification task
        mode: train / val / test (str)
        task: wearer / num_active (str). describes the classification task
        index: an epoch number
        """
        # --- Get a numpy array for the predicted logits, predicted labels, and true labels --
        pred_logits_np = pred_logits.detach().cpu().numpy()
        labels_pred_np = (torch.max(torch.exp(pred_logits), 1)[1]).detach().data.cpu().numpy()
        labels_true_np = labels_true.data.cpu().numpy()
        # ------------------

        # ---- create folder for plots -------------
        save_path = f"{self.dir_fig}/{mode}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # -----------------------------------------
        num_classes = 0
        if task == "wearer":
            num_classes = 2
        elif task == "speaker_places":
            num_classes = self.num_classes
        else:
            print(f"!ERROR! - set num classes for: {task}")

        # === Print classification_report ===
        print("-- classification_report ---")
        target_names = []
        for i in range(num_classes):
            target_names.append(f"class {i}")
        report_labels = list(range(0, num_classes))
        labels_true_np = np.argmax(labels_true_np, axis=1)
        report = classification_report(labels_true_np.flatten(), labels_pred_np.flatten(),
                                       labels=report_labels, target_names=target_names, zero_division=1.0)
        print(report)
        text_file = open(f"{save_path}/class_report_{task}_{index}.txt", "w")
        n = text_file.write(report)
        text_file.close()
        print("----------------------------")
        # ------------------

        # =======   Build and save confusion matrix ===========
        """
        Confusion matrix whose i-th row and j-th column entry indicates the number of samples
         with true label being i-th class and predicted label being j-th class.
        """
        labels = [1, 0]
        # print(f"{labels=}")
        cm1 = confusion_matrix(labels_true_np.flatten(), labels_pred_np.flatten(), labels=labels, normalize="true")
        cm2 = confusion_matrix(labels_true_np.flatten(), labels_pred_np.flatten(), labels=labels, normalize="pred")
        cm3 = confusion_matrix(labels_true_np.flatten(), labels_pred_np.flatten(), labels=labels, normalize="all")
        cm4 = confusion_matrix(labels_true_np.flatten(), labels_pred_np.flatten(), labels=labels, normalize=None)

        # print and save all cm's
        # print(f"CM for 'true':\n{cm1}")
        plt.figure(figsize=(12, 7))
        sn.heatmap(cm1, annot=True)
        plt.suptitle(f"confusion_matrix_true_{task}_{index}")
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.savefig(f"{save_path}/confusion_matrix_true_{task}_{index}.png")
        plt.show()
        plt.close()

        # print(f"CM for 'pred':\n{cm2}")
        plt.figure(figsize=(12, 7))
        sn.heatmap(cm2, annot=True)
        plt.suptitle(f"confusion_matrix_pred_{task}_{index}")
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.savefig(f"{save_path}/confusion_matrix_pred_{task}_{index}.png")
        plt.show()
        plt.close()

        # print(f"CM for 'all':\n{cm3}")
        plt.figure(figsize=(12, 7))
        sn.heatmap(cm3, annot=True)
        plt.suptitle(f"confusion_matrix_all_{task}_{index}")
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.savefig(f"{save_path}/confusion_matrix_all_{task}_{index}.png")
        plt.show()
        plt.close()

        print(f"CM for 'None':\n{cm4}")
        plt.figure(figsize=(12, 7))
        sn.heatmap(cm4, annot=True)
        plt.suptitle(f"confusion_matrix_None_{task}_{index}")
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.savefig(f"{save_path}/confusion_matrix_None_{task}_{index}.png")
        plt.show()
        plt.close()
        # -------------------------------------------------

    def save_loss_acc_iou_f1_precision_recall_plot(self):

        for key in self.loss_all_train.keys():
            # Plot all the loss from train val and test
            plt.figure()
            plt.plot(self.loss_all_train[key], linewidth=3, color='blue', label=f"Loss train")
            plt.plot(self.loss_all_val[key], linewidth=3, color='orange', label=f"Loss val")
            plt.plot(self.loss_all_test[key], linewidth=3, color='red', label=f"Loss test")
            plt.legend()
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(f"Loss '{key}' - Train vs Val vs Test", fontsize=12)
            plt.grid(True)

            fig_name = f"Loss_{key}"
            plt.savefig(f"{self.dir_fig}/{fig_name}")
            plt.close()

        for key in self.acc_all_train.keys():
            # Plot all the acc from train val and test
            plt.figure()
            plt.plot(self.acc_all_train[key], linewidth=3, color='blue', label='Acc train')
            plt.plot(self.acc_all_val[key], linewidth=3, color='orange', label='Acc val')
            plt.plot(self.acc_all_test[key], linewidth=3, color='red', label='Acc test')
            plt.legend()
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(f"Acc '{key}' - Train vs Val vs test", fontsize=12)
            plt.grid(True)

            fig_name = f"Accuracy_{key}"
            plt.savefig(f"{self.dir_fig}/{fig_name}")
            plt.close()

        for key in self.iou_all_train.keys():
            # Plot all the acc from train val and test
            plt.figure()
            plt.plot(self.iou_all_train[key], linewidth=3, color='blue', label='IOU train')
            plt.plot(self.iou_all_val[key], linewidth=3, color='orange', label='IOU val')
            plt.plot(self.iou_all_test[key], linewidth=3, color='red', label='IOU test')
            plt.legend()
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(f"IOU '{key}' - Train vs Val vs test", fontsize=12)
            plt.grid(True)

            fig_name = f"IOU_{key}"
            plt.savefig(f"{self.dir_fig}/{fig_name}")
            plt.close()

        for key in self.f1_all_train.keys():
            # Plot all the acc from train val and test
            plt.figure()
            plt.plot(self.f1_all_train[key], linewidth=3, color='blue', label='F1 train')
            plt.plot(self.f1_all_val[key], linewidth=3, color='orange', label='F1 val')
            plt.plot(self.f1_all_test[key], linewidth=3, color='red', label='F1 test')
            plt.legend()
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(f"F1 '{key}' - Train vs Val vs test", fontsize=12)
            plt.grid(True)

            fig_name = f"F1_{key}"
            plt.savefig(f"{self.dir_fig}/{fig_name}")
            plt.close()

        for key in self.recall_all_train.keys():
            # Plot all therecall from train val and test
            plt.figure()
            plt.plot(self.recall_all_train[key], linewidth=3, color='blue', label='Recall train')
            x = self.recall_all_val[key]
            plt.plot(self.recall_all_val[key], linewidth=3, color='orange', label='Recall val')
            plt.plot(self.recall_all_test[key], linewidth=3, color='red', label='Recall test')
            plt.legend()
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(f"Recall '{key}' - Train vs Val vs test", fontsize=12)
            plt.grid(True)

            fig_name = f"Recall_{key}"
            plt.savefig(f"{self.dir_fig}/{fig_name}")
            plt.close()

        for key in self.precision_all_train.keys():
            # Plot all the precision from train val and test
            plt.figure()
            plt.plot(self.precision_all_train[key], linewidth=3, color='blue', label='Precision train')
            plt.plot(self.precision_all_val[key], linewidth=3, color='orange', label='Precision val')
            plt.plot(self.precision_all_test[key], linewidth=3, color='red', label='Precision test')
            plt.legend()
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(f"Precision '{key}' - Train vs Val vs test", fontsize=12)
            plt.grid(True)

            fig_name = f"Precision_{key}"
            plt.savefig(f"{self.dir_fig}/{fig_name}")
            plt.close()

        for key in self.tversky_all_train.keys():
            # Plot all the tversky from train val and test
            plt.figure()
            plt.plot(self.tversky_all_train[key], linewidth=3, color='blue', label='Tversky train')
            plt.plot(self.tversky_all_val[key], linewidth=3, color='orange', label='Tversky val')
            plt.plot(self.tversky_all_test[key], linewidth=3, color='red', label='Tversky test')
            plt.legend()
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(f"Tversky '{key}' - Train vs Val vs test", fontsize=12)
            plt.grid(True)

            fig_name = f"Tversky_{key}"
            plt.savefig(f"{self.dir_fig}/{fig_name}")
            plt.close()

        for mode in self.cm_dict.keys():
            for task in self.cm_dict[mode].keys():
                # Plot all the CM items
                plt.figure()
                for cm_item in self.cm_dict[mode][task].keys():
                    plt.plot(self.cm_dict[mode][task][cm_item], linewidth=3, label=f'{cm_item}')
                plt.legend()
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(f"CM items of {task} {mode}", fontsize=12)
                plt.grid(True)

                fig_name = f"CM_items_{task}_{mode}"
                plt.savefig(f"{self.dir_fig}/{fig_name}")
                plt.close()

        for mode in self.cm_dict.keys():
            for task in self.cm_dict[mode].keys():
                # Plot all the CM items
                plt.figure()
                for cm_item in ['TP', 'FN']:
                    plt.plot(self.cm_dict[mode][task][cm_item], linewidth=3,
                             label=f'{cm_item}')
                plt.legend()
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(f"TP vs FN of {task} {mode}", fontsize=12)
                plt.grid(True)

                fig_name = f"TP_FN_{task}_{mode}"
                plt.savefig(f"{self.dir_fig}/{fig_name}")
                plt.close()

        for mode in self.cm_dict.keys():
            for task in self.cm_dict[mode].keys():
                # Plot all the CM items
                plt.figure()
                for cm_item in ['TN', 'FP']:
                    plt.plot(self.cm_dict[mode][task][cm_item], linewidth=3,
                             label=f'{cm_item}')
                plt.legend()
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(f"TN vs FP of {task} {mode}", fontsize=12)
                plt.grid(True)

                fig_name = f"TN_FP_{task}_{mode}"
                plt.savefig(f"{self.dir_fig}/{fig_name}")
                plt.close()

    def epoch_loop(self, epoch_idx, mode="train"):
        start_epoch = time.time()

        # Choose the dataloader between train/val/test
        if mode == "train":
            dataloader = self.dataloader_train
        elif mode == "val":
            dataloader = self.dataloader_val
        elif mode == "test":
            dataloader = self.dataloader_test

        for batch_idx, all_data in enumerate(dataloader):
            if (PRINT_FACTOR is not None) and (batch_idx % PRINT_FACTOR == 0):
                print(f"batch index:{batch_idx}")

                current_time = datetime.datetime.now().time()
                # print(current_time.strftime("%H:%M:%S"))

            start_batch = time.time()

            # ---  Extract the relevant data from the data from dataloader ---
            data_dict, labels = all_data
            labels_wearer = labels['is_wearer_active']
            labels_places = labels['places_in_picture']
            labels_places = [eval(x) for x in labels_places]
            # ----------------------------------------------------------------

            # Move data to GPU:
            for key in data_dict.keys():
                data_dict[key] = data_dict[key].to(self.device)
            # ------------------

            # ---- Prep the labels: ---
            labels_wearer = self.prep_labels_wearer(labels_wearer).to(self.device)
            labels_places = self.prep_labels_places(labels_places).to(self.device)
            # ----------------------------------------------------------------

            # ---- Prep Data for the model ---
            data_dict = self.prep_data(data_dict)
            # --------------------------------------------

            # Get the predictions from the model
            self.optimizer.zero_grad()
            pred_logits_wearer, pred_logits_speaker_places = self.model(
                input_data_audio=data_dict.get('audio', None),  # Get audio data, defaulting to None if key not found
                input_data_video=data_dict.get('video', None),  # Get video data, defaulting to None if key not found
            )

            # Save the logits from the model for later
            # all_pred_logits['wearer'] = torch.cat((all_pred_logits['wearer'], pred_logits_wearer.detach()), 0)
            # all_pred_logits['speaker_places'] = torch.cat((all_pred_logits['speaker_places'], pred_logits_speaker_places.detach()), 0)

            # Get the loss for the model's output
            pred_logits_wearer = pred_logits_wearer.float()
            pred_logits_speaker_places = pred_logits_speaker_places.float()

            labels_wearer = labels_wearer.float()
            labels_places = labels_places.float()


            loss_wearer = self.loss_fn_wearer(pred_logits_wearer, labels_wearer)

            #  (4, 2, 360, 640)  -> (4, 360, 640, 2)
            pred_logits_speaker_places = pred_logits_speaker_places.permute(0, 2, 3, 1)
            labels_places = labels_places.permute(0, 2, 3, 1)

            loss_speaker_segmentation = self.loss_fn_speaker_segmentation(
                pred_logits_speaker_places.reshape(-1, self.num_classes),
                labels_places.reshape(-1, self.num_classes))

            pred_logits_speaker_places = pred_logits_speaker_places.permute(0, 3, 1, 2).to("cpu")
            labels_places = labels_places.permute(0, 3, 1, 2).to("cpu")

            # Train only wearer VAD for given number of epochs
            if (self.num_epochs_wearer is not None) and (epoch_idx < self.num_epochs_wearer):
                loss_total = self.config['loss_mix_weights']['loss_wearer'] * loss_wearer
            else:
                loss_total = self.config['loss_mix_weights']['loss_wearer'] * loss_wearer + \
                             self.config['loss_mix_weights']['loss_speaker_segmentation'] * loss_speaker_segmentation

            if batch_idx % loss_print_factor == 0:
                if self.config['break_flag']:
                    batch_num = min(len(dataloader), self.config['batch_before_stop'][mode])
                else:
                    batch_num = len(dataloader)
                print(
                    f"{mode}: Epoch-{epoch_idx + 1}/{self.epoch + self.num_epochs}, Batch-{batch_idx + 1}/{batch_num}: LOSS = {loss_total}")

            # Optimizer step, updating the model's weights
            if mode == "train":
                loss_total.backward()
                self.optimizer.step()

            # Save the losses
            self.epoch_loss['wearer'].append(loss_wearer)   # loss_wearer.item() #.item()
            self.epoch_loss['speaker_places'].append(loss_speaker_segmentation.item()) #.item()
            self.epoch_loss['total'].append(loss_total.item()) #.item()

            speaker_segmentation_labels_pred_np = (torch.max(torch.exp(pred_logits_speaker_places.detach()), 1)[1]).detach().data.cpu().numpy()
            speaker_segmentation_labels_true_np = (torch.max(torch.exp(labels_places.detach()), 1)[1]).data.cpu().numpy()

            wearer_labels_pred_np = (torch.max(torch.exp(pred_logits_wearer.detach()), 1)[1]).detach().data.cpu().numpy()
            wearer_labels_true_np = (torch.max(torch.exp(labels_wearer.detach()), 1)[1]).data.cpu().numpy()

            alpha = self.config['loss_params_dict']['ftl']["alpha"]
            beta = self.config['loss_params_dict']['ftl']["beta"]

            speaker_segmentation_cm = confusion_matrix(speaker_segmentation_labels_true_np.flatten(),
                                                       speaker_segmentation_labels_pred_np.flatten(),
                                                       labels=[1, 0],
                                                       normalize=None)
            # print(f"{speaker_segmentation_cm=}")
            tp, fn, fp, tn = speaker_segmentation_cm.ravel()
            self.epoch_cm['speaker_places']['TP'].append(tp)
            self.epoch_cm['speaker_places']['FP'].append(fp)
            self.epoch_cm['speaker_places']['TN'].append(tn)
            self.epoch_cm['speaker_places']['FN'].append(fn)

            speaker_segmentation_metrics = calc_metrics(TN=tn, FP=fp, FN=fn, TP=tp, alpha=alpha, beta=beta)
            # print(f"{speaker_segmentation_metrics=}")

            wearer_cm = confusion_matrix(wearer_labels_true_np.flatten(),
                                         wearer_labels_pred_np.flatten(),
                                         labels=[1, 0],
                                         normalize=None)
            # print(f"{wearer_cm=}")
            tp, fn, fp, tn = wearer_cm.ravel()
            self.epoch_cm['wearer']['TP'].append(tp)
            self.epoch_cm['wearer']['FP'].append(fp)
            self.epoch_cm['wearer']['TN'].append(tn)
            self.epoch_cm['wearer']['FN'].append(fn)

            wearer_metrics = calc_metrics(TN=tn, FP=fp, FN=fn, TP=tp, alpha=alpha, beta=beta)
            # print(f"{wearer_metrics=}")

            self.update_epoch_dicts(wearer_metrics, speaker_segmentation_metrics)

            end_batch = time.time()

            if batch_idx == 100 and self.epoch == 0:
                self.print_end_time(start_epoch)

            if (self.config['break_flag'] is True) and (batch_idx == self.config['batch_before_stop'][mode]):
                print(f" ==== NOTE!!! there is a break in train after {batch_idx} batches!! ====")

                # ====== Classification report batch =======
                if not self.config["only_wearer_mode"]:

                    print(f"\n\n--- Get classification report for mode:{mode} and task: speaker_places ---")
                    self.get_classification_report_single_task(labels_true=labels_places,
                                                               pred_logits=pred_logits_speaker_places.detach(),
                                                               mode=mode,
                                                               task="speaker_places",
                                                               index=batch_idx + 1)

                print(f"\n\n--- Get classification report for mode:{mode} and task: wearer ---")
                self.get_classification_report_single_task(labels_true=labels_wearer,
                                                           pred_logits=pred_logits_wearer.detach(),
                                                           mode=mode,
                                                           task="wearer",
                                                           index=batch_idx + 1)
                # --------------------------------------
                break

            # ------ finish 'enumerate(self.loader_train)' loop

        # return the mean of the loss and accuracy
        self.mean_epoch()

        # ------------------

    def init_epoch_dicts(self):
        self.epoch_cm = {'wearer': {"TP": [], "TN": [], "FP": [], "FN": []},
                         'speaker_places': {"TP": [], "TN": [], "FP": [], "FN": []}}
        self.epoch_loss = {'wearer': [], 'speaker_places': [], 'total': []}
        self.epoch_acc = {'wearer': [], 'speaker_places': []}
        self.epoch_f1 = {'speaker_places': [], 'wearer': []}
        self.epoch_iou = {'speaker_places': [], 'wearer': []}
        self.epoch_precision = {'speaker_places': [], 'wearer': []}
        self.epoch_recall = {'speaker_places': [], 'wearer': []}
        self.epoch_tversky = {'speaker_places': [], 'wearer': []}

    def update_epoch_dicts(self, wearer_metrics, speaker_segmentation_metrics):
        self.epoch_acc['wearer'].append(wearer_metrics['accuracy'])
        self.epoch_f1['wearer'].append(wearer_metrics['f1'])
        self.epoch_iou['wearer'].append(wearer_metrics['iou'])
        self.epoch_recall['wearer'].append(wearer_metrics['recall'])
        self.epoch_precision['wearer'].append(wearer_metrics['precision'])
        self.epoch_tversky['wearer'].append(wearer_metrics['tversky'])

        self.epoch_acc['speaker_places'].append(speaker_segmentation_metrics['accuracy'])
        self.epoch_f1['speaker_places'].append(speaker_segmentation_metrics['f1'])
        self.epoch_iou['speaker_places'].append(speaker_segmentation_metrics['iou'])
        self.epoch_recall['speaker_places'].append(speaker_segmentation_metrics['recall'])
        self.epoch_precision['speaker_places'].append(speaker_segmentation_metrics['precision'])
        self.epoch_tversky['speaker_places'].append(speaker_segmentation_metrics['tversky'])

    def mean_epoch(self):
        for key in self.epoch_loss.keys():
            self.epoch_loss[key] = np.mean(np.array(torch.tensor(self.epoch_loss[key]))).item() #.item()

        for key in self.epoch_acc.keys():
            self.epoch_acc[key] = np.mean(np.array(torch.tensor(self.epoch_acc[key]))).item()

        for key in self.epoch_f1.keys():
            self.epoch_f1[key] = np.mean(np.array(torch.tensor(self.epoch_f1[key]))).item()

        for key in self.epoch_iou.keys():
            self.epoch_iou[key] = np.mean(np.array(torch.tensor(self.epoch_iou[key]))).item()

        for key in self.epoch_precision.keys():
            self.epoch_precision[key] = np.mean(np.array(torch.tensor(self.epoch_precision[key]))).item()

        for key in self.epoch_recall.keys():
            self.epoch_recall[key] = np.mean(np.array(torch.tensor(self.epoch_recall[key]))).item()

        for key in self.epoch_tversky.keys():
            self.epoch_tversky[key] = np.mean(np.array(torch.tensor(self.epoch_tversky[key]))).item()

        for key1 in self.epoch_cm.keys():  # speaker_places, wearer
            for key2 in self.epoch_cm[key1].keys():  # TP, TN, FP, FN
                self.epoch_cm[key1][key2] = np.mean(np.array(torch.tensor(self.epoch_cm[key1][key2]))).item()

    # --- Training methods -----
    def single_epoch_train(self, epoch_idx=0):
        # Single epoch training

        print(f"start train_epoch: {epoch_idx + 1}")

        self.init_epoch_dicts()

        # Set model to train mode
        self.model.to(self.device)
        self.model.train()
        self.epoch_loop(epoch_idx, mode="train")

    def print_end_time(self, start_epoch):
        hundred_batches_time_s = time.time() - start_epoch
        mean_batch_time_s = hundred_batches_time_s / 100

        examples_num = len(self.dataloader_train) + len(self.dataloader_val) + len(self.dataloader_test)
        examples_before_brake = self.config['batch_before_stop']['train'] + self.config['batch_before_stop']['val'] + \
                                self.config['batch_before_stop']['test']

        if self.config['break_flag']:
            examples_num = min(examples_num, examples_before_brake)

        epoch_time_s = mean_batch_time_s * examples_num

        hours = int(epoch_time_s // 3600)
        minutes = int((epoch_time_s % 3600) // 60)
        remaining_seconds = int(epoch_time_s % 60)

        current_time = datetime.datetime.now().time()

        # Calculate the end time
        end_datetime = datetime.datetime.combine(datetime.date.today(), current_time) + datetime.timedelta(
            hours=hours, minutes=minutes, seconds=remaining_seconds)
        end_time = end_datetime.time()

        print("*******************************************")
        print("The calculation of one epoch will be finished in about {:02d}:{:02d}:{:02d}".format(hours, minutes,
                                                                                                   remaining_seconds))
        print("The running will end at {}".format(end_time.strftime('%H:%M:%S')))
        print("*******************************************")

    def single_epoch_val_test(self, mode_val_test, epoch_idx=0):
        # Single epoch val/test

        print(f"start single_epoch_val_test for: {mode_val_test}, epoch:{epoch_idx + 1}")
        self.init_epoch_dicts()

        # Set model to eval mode
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            self.epoch_loop(epoch_idx, mode=mode_val_test)

    # ------------------------------

    def load_loss_acc_lists(self):
        # ---- Losses -------
        self.loss_dict = self.checkpoint['loss_dict']
        # print(f"@@@@{self.loss_dict=}")
        self.loss_all_train = self.loss_dict['loss_all_train']
        self.loss_all_val = self.loss_dict['loss_all_val']
        self.loss_all_test = self.loss_dict['loss_all_test']

        # ---- Accuracy ---------
        self.acc_dict = self.checkpoint['acc_dict']
        # print(f"@@@@{self.acc_dict=}")
        self.acc_all_train = self.acc_dict['acc_all_train']
        self.acc_all_val = self.acc_dict['acc_all_val']
        self.acc_all_test = self.acc_dict['acc_all_test']

        # ---- F1 Score ---------
        self.f1_dict = self.checkpoint['f1_dict']
        self.f1_all_train = self.f1_dict['f1_all_train']
        self.f1_all_val = self.f1_dict['f1_all_val']
        self.f1_all_test = self.f1_dict['f1_all_test']
        # -------------------

        # ---- IOU ---------
        self.iou_dict = self.checkpoint['iou_dict']
        self.iou_all_train = self.iou_dict['iou_all_train']
        self.iou_all_val = self.iou_dict['iou_all_val']
        self.iou_all_test = self.iou_dict['iou_all_test']
        # -------------------

        # ---- Recall ---------
        self.recall_dict = self.checkpoint['recall_dict']
        self.recall_all_train = self.recall_dict['recall_all_train']
        self.recall_all_val = self.recall_dict['recall_all_val']
        self.recall_all_test = self.recall_dict['recall_all_test']
        # -------------------

        # ---- Precision ---------
        self.precision_dict = self.checkpoint['precision_dict']
        self.precision_all_train = self.precision_dict['precision_all_train']
        self.precision_all_val = self.precision_dict['precision_all_val']
        self.precision_all_test = self.precision_dict['precision_all_test']

        # ---- Tversky ---------
        self.tversky_dict = self.checkpoint['tversky_dict']
        self.tversky_all_train = self.tversky_dict['tversky_all_train']
        self.tversky_all_val = self.tversky_dict['tversky_all_val']
        self.tversky_all_test = self.tversky_dict['tversky_all_test']

        # ---- CM ---------
        self.cm_dict = self.checkpoint['cm_dict']
        self.cm_all_train = self.cm_dict['cm_all_train']
        self.cm_all_val = self.cm_dict['cm_all_val']
        self.cm_all_test = self.cm_dict['cm_all_test']

    def init_loss_acc_lists(self):
        # ---- Losses -------
        self.loss_all_train = {'wearer': [], 'speaker_places': [], 'total': []}
        self.loss_all_val = {'wearer': [], 'speaker_places': [], 'total': []}
        self.loss_all_test = {'wearer': [], 'speaker_places': [], 'total': []}
        self.loss_dict = {'loss_all_train': self.loss_all_train,
                          'loss_all_val': self.loss_all_val,
                          'loss_all_test': self.loss_all_test}
        # -------------------

        # ---- Accuracy ---------
        self.acc_all_train = {'wearer': [], 'speaker_places': []}
        self.acc_all_val = {'wearer': [], 'speaker_places': []}
        self.acc_all_test = {'wearer': [], 'speaker_places': []}
        self.acc_dict = {'acc_all_train': self.acc_all_train,
                         'acc_all_val': self.acc_all_val,
                         'acc_all_test': self.acc_all_test}
        # -------------------

        # ---- F1 Score ---------
        self.f1_all_train = {'speaker_places': [], 'wearer': []}
        self.f1_all_val = {'speaker_places': [], 'wearer': []}
        self.f1_all_test = {'speaker_places': [], 'wearer': []}
        self.f1_dict = {'f1_all_train': self.f1_all_train,
                        'f1_all_val': self.f1_all_val,
                        'f1_all_test': self.f1_all_test}
        # -------------------

        # ---- IOU ---------
        self.iou_all_train = {'speaker_places': [], 'wearer': []}
        self.iou_all_val = {'speaker_places': [], 'wearer': []}
        self.iou_all_test = {'speaker_places': [], 'wearer': []}
        self.iou_dict = {'iou_all_train': self.iou_all_train,
                         'iou_all_val': self.iou_all_val,
                         'iou_all_test': self.iou_all_test}
        # -------------------

        # ---- Recall ---------
        self.recall_all_train = {'speaker_places': [], 'wearer': []}
        self.recall_all_val = {'speaker_places': [], 'wearer': []}
        self.recall_all_test = {'speaker_places': [], 'wearer': []}
        self.recall_dict = {'recall_all_train': self.recall_all_train,
                            'recall_all_val': self.recall_all_val,
                            'recall_all_test': self.recall_all_test}
        # -------------------

        # ---- Precision ---------
        self.precision_all_train = {'speaker_places': [], 'wearer': []}
        self.precision_all_val = {'speaker_places': [], 'wearer': []}
        self.precision_all_test = {'speaker_places': [], 'wearer': []}
        self.precision_dict = {'precision_all_train': self.precision_all_train,
                               'precision_all_val': self.precision_all_val,
                               'precision_all_test': self.precision_all_test}
        # -------------------

        # ---- Tversky ---------
        self.tversky_all_train = {'speaker_places': [], 'wearer': []}  # 'wearer': [],
        self.tversky_all_val = {'speaker_places': [], 'wearer': []}  # 'wearer': [],
        self.tversky_all_test = {'speaker_places': [], 'wearer': []}  # 'wearer': [],
        self.tversky_dict = {'tversky_all_train': self.tversky_all_train,
                             'tversky_all_val': self.tversky_all_val,
                             'tversky_all_test': self.tversky_all_test}
        # -------------------

        # ---- CM ---------
        self.cm_all_train = {'speaker_places': {'TP': [], 'FP': [], 'TN': [], 'FN': []},
                             'wearer': {'TP': [], 'FP': [], 'TN': [], 'FN': []}}
        self.cm_all_val = {'speaker_places': {'TP': [], 'FP': [], 'TN': [], 'FN': []},
                           'wearer': {'TP': [], 'FP': [], 'TN': [], 'FN': []}}
        self.cm_all_test = {'speaker_places': {'TP': [], 'FP': [], 'TN': [], 'FN': []},
                            'wearer': {'TP': [], 'FP': [], 'TN': [], 'FN': []}}
        self.cm_dict = {'cm_all_train': self.cm_all_train,
                        'cm_all_val': self.cm_all_val,
                        'cm_all_test': self.cm_all_test}
        # -------------------

    def init_bests_dicts(self):
        if not self.load_trained_model:
            self.best_val_acc_wearer = {'epoch': 0, 'acc': 0}
            self.best_val_f1_wearer = {'epoch': 0, 'f1': 0}
            self.best_val_iou_wearer = {'epoch': 0, 'iou': 0}
            self.best_val_recall_wearer = {'epoch': 0, 'recall': 0}
            self.best_val_precision_wearer = {'epoch': 0, 'precision': 0}

            self.best_val_acc_speaker_segmentation = {'epoch': 0, 'acc': 0}
            self.best_val_f1_speaker_segmentation = {'epoch': 0, 'f1': 0}
            self.best_val_iou_speaker_segmentation = {'epoch': 0, 'iou': 0}
            self.best_val_recall_speaker_segmentation = {'epoch': 0, 'recall': 0}
            self.best_val_precision_speaker_segmentation = {'epoch': 0, 'precision': 0}

            self.bests_dict = {'best_val_acc_wearer': self.best_val_acc_wearer,
                               'best_val_f1_wearer': self.best_val_f1_wearer,
                               'best_val_iou_wearer': self.best_val_iou_wearer,
                               'best_val_recall_wearer': self.best_val_recall_wearer,
                               'best_val_precision_wearer': self.best_val_precision_wearer,

                               'best_val_acc_speaker_segmentation': self.best_val_acc_speaker_segmentation,
                               'best_val_f1_speaker_segmentation': self.best_val_f1_speaker_segmentation,
                               'best_val_iou_speaker_segmentation': self.best_val_iou_speaker_segmentation,
                               'best_val_recall_speaker_segmentation': self.best_val_recall_speaker_segmentation,
                               'best_val_precision_speaker_segmentation': self.best_val_precision_speaker_segmentation}

        else:
            self.bests_dict = self.checkpoint['bests_dict']

            self.best_val_acc_wearer = self.bests_dict['best_val_acc_wearer']
            self.best_val_f1_wearer = self.bests_dict['best_val_f1_wearer']
            self.best_val_iou_wearer = self.bests_dict['best_val_iou_wearer']
            self.best_val_recall_wearer = self.bests_dict['best_val_recall_wearer']
            self.best_val_precision_wearer = self.bests_dict['best_val_precision_wearer']

            self.best_val_acc_speaker_segmentation = self.bests_dict['best_val_acc_speaker_segmentation']
            self.best_val_f1_speaker_segmentation = self.bests_dict['best_val_f1_speaker_segmentation']
            self.best_val_iou_speaker_segmentation = self.bests_dict['best_val_iou_speaker_segmentation']
            self.best_val_recall_speaker_segmentation = self.bests_dict['best_val_recall_speaker_segmentation']
            self.best_val_precision_speaker_segmentation = self.bests_dict['best_val_precision_speaker_segmentation']

            # ====== Save plot - losses and accuracy ====
            self.save_loss_acc_iou_f1_precision_recall_plot()

    def append_new_epoch(self, mode):
        for key in self.epoch_loss:
            self.loss_dict[f'loss_all_{mode}'][key].append(self.epoch_loss[key])

        for key in self.epoch_acc:
            self.acc_dict[f'acc_all_{mode}'][key].append(self.epoch_acc[key])

        for key in self.epoch_f1:
            self.f1_dict[f'f1_all_{mode}'][key].append(self.epoch_f1[key])

        for key in self.epoch_iou:
            self.iou_dict[f'iou_all_{mode}'][key].append(self.epoch_iou[key])

        for key in self.epoch_recall:
            self.recall_dict[f'recall_all_{mode}'][key].append(self.epoch_recall[key])

        for key in self.epoch_precision:
            self.precision_dict[f'precision_all_{mode}'][key].append(self.epoch_precision[key])

        for key in self.epoch_tversky:
            self.tversky_dict[f'tversky_all_{mode}'][key].append(self.epoch_tversky[key])

        for key1 in self.epoch_cm.keys():  # speaker_places, wearer
            for key2 in self.epoch_cm[key1].keys():  # TP, TN, FP, FN
                self.cm_dict[f'cm_all_{mode}'][key1][key2].append(self.epoch_cm[key1][key2])

    def update_bests(self, epoch):

        if self.epoch_acc['wearer'] > self.best_val_acc_wearer['acc']:
            self.best_val_acc_wearer['acc'] = self.epoch_acc['wearer']
            self.best_val_acc_wearer['epoch'] = epoch

        if self.epoch_acc['speaker_places'] > self.best_val_acc_speaker_segmentation['acc']:
            self.best_val_acc_speaker_segmentation['acc'] = self.epoch_acc['speaker_places']
            self.best_val_acc_speaker_segmentation['epoch'] = epoch

        if self.epoch_f1['wearer'] > self.best_val_f1_wearer['f1']:
            self.best_val_f1_wearer['f1'] = self.epoch_f1['wearer']
            self.best_val_f1_wearer['epoch'] = epoch

        if self.epoch_f1['speaker_places'] > self.best_val_f1_speaker_segmentation['f1']:
            self.best_val_f1_speaker_segmentation['f1'] = self.epoch_f1['speaker_places']
            self.best_val_f1_speaker_segmentation['epoch'] = epoch

        if self.epoch_iou['speaker_places'] > self.best_val_iou_speaker_segmentation['iou']:
            self.best_val_iou_speaker_segmentation['iou'] = self.epoch_iou['speaker_places']
            self.best_val_iou_speaker_segmentation['epoch'] = epoch

        if self.epoch_iou['wearer'] > self.best_val_iou_wearer['iou']:
            self.best_val_iou_wearer['iou'] = self.epoch_iou['wearer']
            self.best_val_iou_wearer['epoch'] = epoch

        if self.epoch_recall['speaker_places'] > self.best_val_recall_speaker_segmentation['recall']:
            self.best_val_recall_speaker_segmentation['recall'] = self.epoch_recall['speaker_places']
            self.best_val_recall_speaker_segmentation['epoch'] = epoch

        if self.epoch_recall['wearer'] > self.best_val_recall_wearer['recall']:
            self.best_val_recall_wearer['recall'] = self.epoch_recall['wearer']
            self.best_val_recall_wearer['epoch'] = epoch

        if self.epoch_precision['speaker_places'] > self.best_val_precision_speaker_segmentation['precision']:
            self.best_val_precision_speaker_segmentation['precision'] = self.epoch_precision['speaker_places']
            self.best_val_precision_speaker_segmentation['epoch'] = epoch

        if self.epoch_precision['wearer'] > self.best_val_precision_wearer['precision']:
            self.best_val_precision_wearer['precision'] = self.epoch_precision['wearer']
            self.best_val_precision_wearer['epoch'] = epoch
        # ---------------------

    def full_train(self):
        # ----- inti. lists ---
        if not self.load_trained_model:
            self.init_loss_acc_lists()
        else:
            self.load_loss_acc_lists()
        # -----------------

        self.init_bests_dicts()

        print(f" ==== Start training loop =======")
        for epoch in range(self.epoch, self.epoch + self.num_epochs):
            start_time = time.time()
            """
            *************************************  
            For each of the following methods 'single_epoch_train' and 'single_epoch_val_test' the return is:
            epoch_loss: dict' with: {'wearer', 'speaker_places','total'} each is list with the losses in this epoch-per batch
            epoch_acc : dict' with: {'wearer', 'speaker_places'} each is a list with the accuracy in the epoch per-batch
            all_labels_true: dict' with: {'wearer', 'speaker_places'} each is a tensor with all the true labels in the epoch
            all_pred_logits: dict' with: {'wearer', 'speaker_places'} each is a tensor with all the logits in the epoch
            *************************************
            """
            # ----- Train ------
            print(f"--- *** Start Train the model for epoch={epoch + 1}*** -------")
            start_train = time.time()

            self.single_epoch_train(epoch_idx=epoch)  # all_labels_true_epoch_train, all_pred_logits_epoch_train,

            end_train = time.time()
            epoch_mins_train, epoch_secs_train = epoch_time(start=start_train, end=end_train)
            print(f'\n--- train Time: {epoch_mins_train}m {epoch_secs_train:.2f}s')

            self.append_new_epoch(mode='train')
            # ------------------


            # ----- Val ------
            print(f"--- *** Start Val the model for epoch={epoch + 1}*** -------")
            start_val = time.time()

            self.single_epoch_val_test(mode_val_test="val", epoch_idx=epoch)  # all_labels_true_epoch_val, all_pred_logits_epoch_val,

            end_val = time.time()
            epoch_mins_val, epoch_secs_val = epoch_time(start_val, end_val)
            print(f'--- val Time: {epoch_mins_val}m {epoch_secs_val:.2f}s')

            # Update best val acc:
            self.update_bests(epoch=epoch)

            self.append_new_epoch(mode='val')
            # ------------------


            # ----- Test ------
            print(f"--- *** Start Testing the model for epoch={epoch + 1}*** -------")
            start_test = time.time()

            self.single_epoch_val_test(mode_val_test="test", epoch_idx=epoch)  # all_labels_true_epoch_test, all_pred_logits_epoch_test,

            end_test = time.time()
            epoch_mins_test, epoch_secs_test = epoch_time(start_test, end_test)

            self.append_new_epoch(mode='test')
            # ------------------


            # Print training times
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch - Total train,Val,Test : {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
            print(f'\ttrain Time: {epoch_mins_train}m {epoch_secs_train}s')
            print(f'\tval   Time: {epoch_mins_val}m   {epoch_secs_val}s')
            print(f'\ttest  Time: {epoch_mins_test}m  {epoch_secs_test}s')

            # Print losses in epoch
            loss_print_factor = 1

            # --- Save a model checkpoint -----
            if (self.config['save_model'] is True) and ((epoch + 1) % self.config['save_every'] == 0):
                print(f" --- Start Saving model checkpoint for epoch: {epoch + 1}")
                filename = f"checkpoint_epoch_{epoch + 1}"
                torch.save({'epoch': epoch,
                            'model': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            # 'loss': self.total_loss_epoch_test,
                            'save_name': self.save_name,
                            'config': self.config,
                            'bests_dict': self.bests_dict,
                            'loss_dict': self.loss_dict,
                            'acc_dict': self.acc_dict,
                            'iou_dict': self.iou_dict,
                            'f1_dict': self.f1_dict,
                            'recall_dict': self.recall_dict,
                            'precision_dict': self.precision_dict,
                            'tversky_dict': self.tversky_dict,
                            'cm_dict': self.cm_dict},
                           f"{self.dir_models}/{filename}.tar")  # PATH

                # ====== Save plot - losses and accuracy ====
                self.save_loss_acc_iou_f1_precision_recall_plot()

            # if self.config['test_one_batch_mode']:
            #     self.save_loss_acc_iou_f1_precision_recall_plot()

                print('Finished saving model - Checkpoint Saved')
            # --- finish saving the best-acc-so-far model

            # ====== Classification report =======
            # self.get_classification_report(labels_true=all_labels_true_epoch_train,
            #                                pred_logits=all_pred_logits_epoch_train,
            #                                mode="train",
            #                                index=epoch + 1)
            #
            # self.get_classification_report(labels_true=all_labels_true_epoch_val,
            #                                pred_logits=all_pred_logits_epoch_val,
            #                                mode="val",
            #                                index=epoch + 1)
            #
            # self.get_classification_report(labels_true=all_labels_true_epoch_test,
            #                                pred_logits=all_pred_logits_epoch_test,
            #                                mode="test",
            #                                index=epoch + 1)
            # --------------------------------------

        self.epoch += 1
        # ------------------------------------
        # ====== END for loop of training the model ==========

        print(f"Best val accuracy - 'wearer' - during training:{self.best_val_acc_wearer['acc'] * 100:.2f}%"
              f"\t For epoch: {self.best_val_acc_wearer['epoch'] + 1}")

        print(
            f"Best val F1 score - 'wearer' - during training:{self.best_val_f1_wearer['f1'] * 100:.2f}%"
            f"\t For epoch: {self.best_val_f1_wearer['epoch'] + 1}")

        print(
            f"Best val IOU - 'wearer' - during training:{self.best_val_iou_wearer['iou'] * 100:.2f}%"
            f"\t For epoch: {self.best_val_iou_wearer['epoch'] + 1}")

        print(
            f"Best val Recall score - 'wearer' - during training:{self.best_val_recall_wearer['recall'] * 100:.2f}%"
            f"\t For epoch: {self.best_val_recall_wearer['epoch'] + 1}")

        print(
            f"Best val Precision - 'wearer' - during training:{self.best_val_precision_wearer['precision'] * 100:.2f}%"
            f"\t For epoch: {self.best_val_precision_wearer['epoch'] + 1}")

        print("------------------------------------")

        print(
            f"Best val accuracy - 'speaker_segmentation' - during training:{self.best_val_acc_speaker_segmentation['acc'] * 100:.2f}%"
            f"\t For epoch: {self.best_val_acc_speaker_segmentation['epoch'] + 1}")

        print(
            f"Best val F1 score - 'speaker_segmentation' - during training:{self.best_val_f1_speaker_segmentation['f1'] * 100:.2f}%"
            f"\t For epoch: {self.best_val_f1_speaker_segmentation['epoch'] + 1}")

        print(
            f"Best val IOU - 'speaker_segmentation' - during training:{self.best_val_iou_speaker_segmentation['iou'] * 100:.2f}%"
            f"\t For epoch: {self.best_val_iou_speaker_segmentation['epoch'] + 1}")

        print(
            f"Best val Recall score - 'speaker_segmentation' - during training:{self.best_val_recall_speaker_segmentation['recall'] * 100:.2f}%"
            f"\t For epoch: {self.best_val_recall_speaker_segmentation['epoch'] + 1}")

        print(
            f"Best val Precision - 'speaker_segmentation' - during training:{self.best_val_precision_speaker_segmentation['precision'] * 100:.2f}%"
            f"\t For epoch: {self.best_val_precision_speaker_segmentation['epoch'] + 1}")

        # ----- Saving the last trained-model ------
        print(" ---- Start Saving Final model ----")
        filename = f"FinalModel"
        torch.save({'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # 'loss': self.total_loss_epoch_test,
                    'save_name': self.save_name,
                    'config': self.config,
                    'bests_dict': self.bests_dict,
                    'loss_dict': self.loss_dict,
                    'acc_dict': self.acc_dict,
                    'iou_dict': self.iou_dict,
                    'f1_dict': self.f1_dict,
                    'recall_dict': self.recall_dict,
                    'precision_dict': self.precision_dict,
                    'tversky_dict': self.tversky_dict,
                    'cm_dict': self.cm_dict},
                   f"{self.dir_models}/{filename}.tar")  # PATH
        print('Finished saving model - Checkpoint Saved')
        # -------------------------------------------

        # ====== Save plot - losses and accuracy ====
        self.save_loss_acc_iou_f1_precision_recall_plot()
        # self.save_loss_precision_plot()
        # self.save_loss_recall_plot()
        # --------------------------------------------

    def train_model(self):
        print("\n\n ======================================")
        print(f"Start training, with num epochs:{self.num_epochs}")
        print(f"---------------------------------------")

        self.full_train()

    def print_model_summary(self):
        input_size_video = [self.config['batch_train'],
                            self.data_param_video['n_frames'], self.data_param_video['n_channels'],
                            self.data_param_video['frame_h'], self.data_param_video['frame_w']]
        input_size_audio = [self.config['batch_train'], self.data_param_audio['n_mics'],
                            self.data_param_audio['n_fft'], self.data_param_audio['n_time']]

        col_names = ["input_size", "output_size", "num_params"]
        with torch.cuda.device(self.device):
            summary(self.model,
                    input_size=[input_size_audio, input_size_video],
                    col_names=col_names)  # set input_size to tuple for [audio, video]


def get_parameters_from_user(device_ids,
                             MultiGPU,
                             break_flag,
                             B,
                             num_epochs,
                             unet_size,
                             loss_type,
                             loss_params_dict,  # weight_speaking_class,
                             color_mode,
                             train_on_speakers_table,
                             only_wearer_mode):
    # print("In get_parameters_from_user")
    if len(device_ids) > 1 and MultiGPU == True:
        cuda_num = device_ids[0]
    else:
        cuda_num = int(input(f"Enter cuda_num (default is {device_ids[0]}):"))
    break_flag = input(f"Enter break_flag (default is {break_flag}):")
    break_flag = True if break_flag == "True" else False
    print(f"{break_flag=}")
    only_wearer_mode = input(f"Enter only_wearer_mode (default is {only_wearer_mode}):")
    only_wearer_mode = True if only_wearer_mode == "True" else False
    print(f"{only_wearer_mode=}")
    # B = int(input(f"Enter B (default is {B}):"))
    num_epochs = int(input(f"Enter num_epochs (default is {num_epochs}):"))
    # unet_size = int(input(f"Enter unet_size (512/128/64 default is {unet_size}):"))
    loss_type = input(f"Enter loss_type ['bce', 'dc', 'ftl'] (default is {loss_type}):")
    for param in loss_params_dict[loss_type]:
        new_param_val = input(f"Enter {param} (default is {loss_params_dict[loss_type][param]}):")
        if new_param_val == "None":
            new_param_val = None
        else:
            new_param_val = float(new_param_val)
        loss_params_dict[loss_type][param] = new_param_val
    color_mode = input(f"Enter color_mode ['RGB', 'grayscale'] (default is {color_mode}):")

    train_on_speakers_table = input(f"Enter train_on_speakers_table (default is {train_on_speakers_table}):")
    train_on_speakers_table = True if train_on_speakers_table == "True" else False

    return cuda_num, break_flag, B, num_epochs, unet_size, loss_type, loss_params_dict, color_mode, train_on_speakers_table, only_wearer_mode


def get_config():
    MultiGPU = False  # True/False for using DataParallel
    break_flag = True  # False  # False=don't use break, True=use break

    # --- Choose the model to be trained ----
    choose_model = "av_model"  # "vit_csd_av_v1", vit_csd_audio , vit_csd_av_v1
    # ---------------------------------------

    # ===== Choose the different models parameters: ====
    feature_type = "stft"  # stft , mel , inst_RTF

    ########################################
    # --- Dataset -----
    data_name = "easycom"
    num_classes = 2
    num_audio_channels = 6  # facebook:6
    video_fps = 20  # The video frame per second
    context_frames = [3, 3]
    av_key = "av"  # a, v , av (audio/video/audio-video)
    label_col_name = ["is_wearer_active", "places_in_picture"]
    balanced_dataset = False
    train_on_speakers_table = False  # speakers_train_table.csv
    color_mode = "RGB"  # "RGB", "grayscale"
    visualize_mode = False
    dir_visualize_csv_table = None

    # ===== Data processing parameters =====
    # Audio:
    sr = 16000  # the data will be resampled for this sr.
    num_fft = 512  # will result in (n_fft//2+1) tensors
    n_fft_effective = (num_fft // 2) + 1

    # Video
    downsample_ratio = 6  # 3/ 6
    video_frame_original_h = 1080
    video_frame_original_w = 1920
    video_frame_h = video_frame_original_h // downsample_ratio
    video_frame_w = video_frame_original_w // downsample_ratio

    # -------------------------------------------

    # ---- Loss Function ---------
    loss_mix_weights = {'loss_wearer': 1, 'loss_speaker_segmentation': 0}
    loss_type = "bce"  # "bce", "dc"-Dice Coefficient, "ftl"-Focal Tversky Loss
    pos_weight_speaker_segmentation = 100  # [1, 2, 5, 10, 100, 1000, 10000]
    pos_weight_wearer = 3
    loss_params_bce = {"weight": None,
                       "pos_weight_speaker_segmentation": pos_weight_speaker_segmentation,
                       "pos_weight_wearer": pos_weight_wearer}
    loss_params_ftl = {"classes": None, "alpha": 0.5, "beta": 0.5, "gamma": 1.0}
    loss_params_dc = {"classes": None}
    loss_params_dict = {"bce": loss_params_bce,
                        "ftl": loss_params_ftl,
                        "dc": loss_params_dc}
    loss_weight = None

    # --------------------
    # =========================

    # === Choose optimizer ===
    optim_type = "Adam"  # Adam, SGD
    optim_lr = 1e-6
    optim_weight_decay = 1e-9
    optim_momentum = 0  # for SGD: default=0
    # ------------------------

    # ===== Training parameter ======
    save_model = True  # save during train ; It always saves the last model on the last epoch
    save_every = 1  # save a model every given number of epoches
    test_one_batch_mode = False
    num_epochs = 15  # 15  # number of total training epochs
    num_epochs_wearer = None  # number of epochs to train only the VAD-wearer classification. None|5

    av_combination_level = "before_unet"  # "before_unet", "in_unet_bottleneck"
    only_wearer_mode = False
    unet_size = 512  # 512/128/64
    batch_before_stop = {'train': 1000, 'val': 500, 'test': 500}

    B = 16
    batch_train = B
    batch_val = B
    batch_test = B

    if color_mode == "grayscale":
        video_n_channels = 1
    else:
        video_n_channels = 3

    num_to_val = None  # None(=val all), or number of examples to val
    num_workers = 10  # usually I use 8, after running the script saw 10 is faster

    loss_params = loss_params_dict[loss_type]
    # --------------------------------------------------------------

    config = {
        "device_ids": device_ids,
        "cuda_num": cuda_num,
        "MultiGPU": MultiGPU,
        "n_GPU": len(device_ids),

        "choose_model": choose_model,
        "feature_type": feature_type,

        # ----- Dataset ---
        "data_name": data_name,
        "num_classes": num_classes,
        "num_audio_channels": num_audio_channels,
        "video_fps": video_fps,
        "context_frames": context_frames,
        "av_key": av_key,
        "balanced_dataset": balanced_dataset,
        "label_col_name": label_col_name,
        "color_mode": color_mode,
        "train_on_speakers_table": train_on_speakers_table,

        "visualize_mode": visualize_mode,
        "dir_visualize_csv_table": dir_visualize_csv_table,
        "test_one_batch_mode": test_one_batch_mode,

        # --- loss ---
        "loss_mix_weights": loss_mix_weights,
        "loss_type": loss_type,
        "loss_params": loss_params,
        "loss_weight": loss_weight,
        "loss_params_dict": loss_params_dict,

        # --- Optimizer ---
        "optim_type": optim_type,
        "optim_lr": optim_lr,
        "optim_weight_decay": optim_weight_decay,
        "optim_momentum": optim_momentum,

        # --- Data processing parameters -----
        "sr": sr,
        "num_fft": num_fft,
        "n_fft_effective": n_fft_effective,

        "video_n_channels": video_n_channels,
        "video_frame_h": video_frame_h,
        "video_frame_w": video_frame_w,
        "downsample_ratio": downsample_ratio,

        # --- Training params ---
        "break_flag": break_flag,
        "save_model": save_model,
        "save_every": save_every,
        "num_epochs": num_epochs,
        "num_epochs_wearer": num_epochs_wearer,
        "batch_train": batch_train,
        "batch_val": batch_val,
        "batch_test": batch_test,
        "num_to_val": num_to_val,
        "num_workers": num_workers,

        'av_combination_level': av_combination_level,
        'only_wearer_mode': only_wearer_mode,
        'unet_size': unet_size,
        'batch_before_stop': batch_before_stop
    }
    return config


if __name__ == "__main__":
    # hyper-parameters
    device_ids = [0, 1, 2, 3]
    cuda_num = device_ids[0]  # device must be the first device_ids

    load_trained_model = False

    if not load_trained_model:
        config = get_config()
        pprint.pprint(config, sort_dicts=False)
        print("=====================================")

        trainer = Trainer(config, load_trained_model)  # , wandb_run_name)

    else:  # load_trained_model

        dir_root = "/dsi/scratch/from_netapp/users/elnatan_k"
        results_path = f"{dir_root}/new_results"
        model_name = "2023_09_14__16_38_LossType_bce_weight_None_pos_weight_speaker_segmentation_0.5_pos_weight_wearer_0.5_unet_size_512_before_unet_only_wearer_mode_n_epochs_10_B_16_color_mode_RGB"
        model_path = f"{results_path}/{model_name}/model"

        checkpoint = torch.load(f'{model_path}/FinalModel.tar')
        config = checkpoint['config']
        trainer = Trainer(config, load_trained_model, checkpoint)

    trainer.train_model()
