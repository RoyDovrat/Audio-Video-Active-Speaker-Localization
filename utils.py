import datetime
import ast
import os
import shutil
from filecmp import dircmp

import torch
from torchmetrics import JaccardIndex
from sklearn.metrics import precision_score, recall_score
from torchmetrics.classification import BinaryF1Score
import time
import pandas as pd
from segmentation_models_pytorch.losses import TverskyLoss
from sklearn.metrics import confusion_matrix
import pprint


def choose_cuda(cuda_num):
    if cuda_num == "cpu" or cuda_num == -1:
        device = "cpu"
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if 0 <= cuda_num <= device_count - 1:  # devieces starts from '0'
            device = torch.device(f"cuda:{cuda_num}")
        else:
            print(f"Cuda Num:{cuda_num} NOT found, choosing cuda:0")
            device = torch.device(f"cuda:{0}")
    else:
        device = torch.device("cpu")

    print("*******************************************")
    print(f" ****** running on device: {device} ******")
    print("*******************************************")
    return device


def calc_accuracy(TN, FP, FN, TP):
    return (TP + TN)/(TN + FP + FN + TP)


def calc_f1(FP, FN, TP, zero_division=1.0):
    if 2*TP + FP + FN == 0:
        return zero_division
    return 2*TP/(2*TP + FP + FN)


def calc_precision(FP, TP, zero_division=1.0):
    if TP + FP == 0:
        return zero_division
    return TP/(TP + FP)


def calc_recall(FN, TP, zero_division=1.0):
    if TP + FN == 0:
        return zero_division
    return TP/(TP + FN)


def calc_iou(FP, FN, TP, zero_division=1.0):
    if TP + FP + FN == 0:
        return zero_division
    return TP/(TP + FP + FN)


def calc_tversky(FP, FN, TP, alpha, beta, zero_division=1.0):
    if TP + alpha*FP + beta*FN == 0:
        return zero_division
    return TP/(TP + alpha*FP + beta*FN)


def calc_metrics(TN, FP, FN, TP, alpha=0.5, beta=0.5):
    return {"accuracy": calc_accuracy(TN, FP, FN, TP),
            "f1": calc_f1(FP, FN, TP),
            "precision": calc_precision(FP, TP),
            "recall": calc_recall(FN, TP),
            "iou": calc_iou(FP, FN, TP),
            "tversky": calc_tversky(FP, FN, TP, alpha, beta)}


def calculate_accuracy(y_pred, y):
    """
    calc the accuracy of the predictions between y_pred and y.
    for example: for 40% returns 0.40
    """
    # Convert predicted probabilities to class labels
    _, y_pred_labels = torch.max(y_pred, dim=1)
    # Convert true labels to class labels
    _, y_true_labels = torch.max(y, dim=1)
    # Calculate accuracy
    accuracy = torch.mean((y_pred_labels == y_true_labels).float())
    return accuracy


# def calculate_dice(y_pred, y):
#     # Convert predicted probabilities to class labels
#     _, y_pred_labels = torch.max(y_pred, dim=1)
#     # Convert true labels to class labels
#     _, y_true_labels = torch.max(y, dim=1)
#     # dice_score = (2 * (y_pred_labels * y_true_labels).sum()) / (
#     #         (y_pred_labels + y_true_labels).sum() + 1e-8
#     # )
#     F1Score = BinaryF1Score(y_pred_labels.flatten(), y_true_labels.flatten())
#     #print(f"{dice_score=}")
#     return F1Score  #dice_score

# calculate_f1_score
def calculate_dice(y_pred, y):  # Set the threshold to an appropriate value
    y_pred_probs = torch.softmax(y_pred, dim=1)  # Convert logits to probabilities
    _, y_pred_labels = torch.max(y_pred_probs, dim=1)
    _, y_true_labels = torch.max(y, dim=1)
    f1 = BinaryF1Score(task="binary", num_classes=2)
    f1_score = f1(y_pred_labels.flatten(), y_true_labels.flatten())
    # print(f"{f1_score=}")
    return f1_score


def calculate_iou(y_pred, y):
    jaccard = JaccardIndex(task='binary',
                           num_classes=2)  # task: Literal['binary', 'multiclass', 'multilabel'], threshold: float= 0.5, num_classes: Optional[int] = None, num_labels: Optional[int] = None, average: Optional[Literal['micro', 'macro', 'weighted', 'none']] = 'macro', ignore_index: Optional[int] = None, validate_args: bool= True, **kwargs:
    return jaccard(y_pred, y)


def calculate_precision(y_pred, y):
    # Flatten the tensors and convert to binary values
    # y_pred_flat = torch.round(y_pred).flatten()
    # y_true_flat = y.flatten()

    # Convert predicted probabilities to class labels
    _, y_pred_labels = torch.max(y_pred, dim=1)
    # Convert true labels to class labels
    _, y_true_labels = torch.max(y, dim=1)

    # num_y_pred_labels_ones = torch.sum(y_pred_labels == 1)
    # print("Number of ones in y_pred_labels:", num_y_pred_labels_ones.item())
    # num_y_true_labels_ones = torch.sum(y_true_labels == 1)
    # print("Number of ones in y_true_labels:", num_y_true_labels_ones.item())

    # Calculate precision
    precision = precision_score(y_pred_labels.flatten(), y_true_labels.flatten(), zero_division=1)
    # print(f"{precision=}")
    return precision


def calculate_recall(y_pred, y):
    # Convert predicted probabilities to class labels
    _, y_pred_labels = torch.max(y_pred, dim=1)
    # Convert true labels to class labels
    _, y_true_labels = torch.max(y, dim=1)

    # Calculate precision
    recall = recall_score(y_pred_labels.flatten(), y_true_labels.flatten(), average='micro')
    return recall


def calculate_speaker_segmentation_accuracy(y_pred, y):  #TODO: is this function fits to our project
    """
    calc the accuracy of the predictions between y_pred and y.
    for example: for 40% returns 0.40
    """
    dice_score = 0
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y[:, 0].view_as(top_pred)).sum()    #TODO: y[:, 0] or y[:, 1]??
    acc = correct.float() / y.shape[0]
    dice_score += (2 * (top_pred * y).sum()) / (    #TODO: top_pred or y_pred?
            (top_pred + y).sum() + 1e-8
    )
    #print(f"Dice score: {dice_score / len(loader)}") #TODO: fix score
    #print(f"{dice_score=}")
    return acc


def get_config():
    # hyper-parameters
    device_ids = [2, 3]  # [0 ,1, 2, 3]#[0, 1, 2]  # [2, 5, 6, 7]   # choose cuda number for deploying the model
    cuda_num = device_ids[0]  # device must be the first device_ids
    MultiGPU = False  # True/False for using DataParallel
    break_flag = False  # False=don't use break, True=use break

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
    label_col_name = ["is_wearer_active", "places_in_picture"]  # ["speakers_num", "is_wearer_active"]
    balanced_dataset = False

    # ===== Data processing parameters =====
    # Audio:
    sr = 16000  # the data will be resampled for this sr.
    num_fft = 512  # will result in (n_fft//2+1) tensors
    n_fft_effective = (num_fft // 2) + 1

    # Video
    color_mode = "grayscale"  # "grayscale"/"RGB"
    if color_mode == "grayscale":
        channels_num = 1
    else:
        channels_num = 3
    video_n_channels = channels_num
    downsample_ratio = 3
    video_frame_original_h = 1080
    video_frame_original_w = 1920
    video_frame_h = video_frame_original_h//downsample_ratio
    video_frame_w = video_frame_original_w//downsample_ratio
    # -------------------------------------------

    # ---- Loss Function ---------
    loss_mix_weights = {'loss_wearer': 0, 'loss_speaker_segmentation': 1}  # weight between the losses
    loss_type = "bce"  # "ce"  #bce: binaryCrossEntropyLoss ce: pytorch's CrossEntropyLoss, cs: cost sensitive losses
    # LS           = 0.1   # ce's label_smoothing: set =0.0 for NO smoothing , need to be a float in range [0.0,1.0]
    # cs_base_loss = "ls"  # cs's base loss:'ce' CrossEntropyLoss, 'ls' label smoothing, 'gls': gaussian label smoothing
    # cs_lambd     = 15    # for CS loss, this is the weight between base CE-loss and the CS-loss

    loss_weight = None  # None / ndarray defined below - !DON'T change here!
    M_avg = None  # None / ndarray defined below - !DON'T change here!
    flag_loss_W = None  # None / ndarray defined below - !DON'T change here!

    # --------------------
    # =========================

    # === Choose optimizer ===
    optim_type = "Adam"  # Adam, SGD
    optim_lr = 1e-6
    optim_weight_decay = 1e-9
    optim_momentum = 0  # for SGD: default=0
    # ------------------------

    # ===== Training parameter ======
    save_model = True  # save during train ; It always saves the last model on the last epoch  #!!!
    save_every = 1  # save a model every given number of epoches  #!!!
    num_epochs = 5  # 15  # number of total training epochs
    num_epochs_wearer = None  # number of epochs to train only the VAD-wearer classification. None|5

    av_combination_level = "in_unet_bottleneck"  # "before_unet", "in_unet_bottleneck"
    unet_size = 512  # 512/128/64  "small"  # "small"-128, "big"-512
    batch_before_stop = 100  # 200/500

    B = 4  # originally was 128
    # cuda_num, break_flag, B, num_epochs, unet_size = get_parameters_from_user(device_ids,
    #                                                                           MultiGPU,
    #                                                                           break_flag,
    #                                                                           B,
    #                                                                           num_epochs,
    #                                                                           unet_size)
    batch_train = B
    batch_val = B
    batch_test = B
    num_to_val = None  # None(=val all), or number of examples to val
    num_workers = 10  # usually I use 8, after running the script saw 10 is faster

    # --------------------------------------------------------------

    # use W&B for tracking the results
    # wandb.login()
    config = {
        "device_ids": device_ids,
        "cuda_num": cuda_num,
        "MultiGPU": MultiGPU,
        "n_GPU": len(device_ids),

        "choose_model": choose_model,

        # # # --- model vit CSD Transformer hyperparams
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

        # --- loss ---
        "loss_mix_weights": loss_mix_weights,
        "loss_type": loss_type,
        # "LS"              : LS,
        # "cs_base_loss"    : cs_base_loss,
        # "cs_lambd"        : cs_lambd,
        "M_avg": M_avg,
        "loss_weight": loss_weight,

        # --- Optimizer ---
        "optim_type": optim_type,
        "optim_lr": optim_lr,
        "optim_weight_decay": optim_weight_decay,
        "optim_momentum": optim_momentum,

        # --- Data processing parameters -----
        "sr": sr,
        "num_fft": num_fft,
        "n_fft_effective": n_fft_effective,
        # "flag_log": flag_log,

        "video_n_channels": video_n_channels,
        "video_frame_h": video_frame_h,
        "video_frame_w": video_frame_w,

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
        'unet_size': unet_size,
        'batch_before_stop': batch_before_stop
    }
    return config


def get_config1():
    # hyper-parameters
    device_ids = [0, 1, 2, 3]  # [0 ,1, 2, 3]#[0, 1, 2]  # [2, 5, 6, 7]   # choose cuda number for deploying the model
    cuda_num = device_ids[0]  # device must be the first device_ids
    MultiGPU = False  # True/False for using DataParallel
    break_flag = True  #False  # False=don't use break, True=use break

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
    context_frames = [3, 3] #TODO: to fit for smaller context
    av_key = "av"  # a, v , av (audio/video/audio-video)
    label_col_name = ["is_wearer_active", "places_in_picture"]  # ["speakers_num", "is_wearer_active"]
    balanced_dataset = False  # TODO: this is not supported yet in the easycom dataset
    color_mode = "grayscale"  # "RGB", "grayscale"

    # ===== Data processing parameters =====
    # Audio:
    sr = 16000  # the data will be resampled for this sr.
    num_fft = 512  # will result in (n_fft//2+1) tensors
    n_fft_effective = (num_fft // 2) + 1

    # Video
    if color_mode == "grayscale":
        video_n_channels = 1
    else:
        video_n_channels = 3

    video_frame_h = 360
    video_frame_w = 640
    # -------------------------------------------

    # ---- Loss Function ---------
    loss_mix_weights = {'loss_wearer': 0, 'loss_speaker_segmentation': 1}  # weight between the losses
    loss_type = "ftl"  # "bce", "dc"-Dice Coefficient, "ftl"-Focal Tversky Loss       #TODO: test this loss types to solve the imbalancing
    #  "ce"  #bce: binaryCrossEntropyLoss ce: pytorch's CrossEntropyLoss, cs: cost sensitive losses
    weight_not_speaking_class = 1000 #[1, 2, 5, 10, 100, 1000]  #TODO: why in "weight_not_speaking_class" - not?
    # LS           = 0.1   # ce's label_smoothing: set =0.0 for NO smoothing , need to be a float in range [0.0,1.0]
    # cs_base_loss = "ls"  # cs's base loss:'ce' CrossEntropyLoss, 'ls' label smoothing, 'gls': gaussian label smoothing
    # cs_lambd     = 15    # for CS loss, this is the weight between base CE-loss and the CS-loss

    loss_weight = None  # None / ndarray defined below - !DON'T change here!
    M_avg = None  # None / ndarray defined below - !DON'T change here!
    flag_loss_W = None  # None / ndarray defined below - !DON'T change here!
    if loss_type == "bce":  # regular binaryCrossEntropyLoss()
        # for CrossEntropyLoss() we can use 'loss_weight' - a vector of weights for each class
        # loss_weight: can be None or weight according to the number of example of each class
        if data_name == "easycom":
            """
            There are total of 381828 frames in the dataset, 
            EasyCom num of examples each class: {0: 116254, 1: 222332, 2: 38280, 3: 4859, 4: 103}
            EasyCom num of examples for 3 classes:{0: 116,254, 1: 222,332, 2+3+4: 38280+4859+103=43,242}
            loss_weight_i = 1 - (#Class_i / #tot)
            loss_weight for EasyCom with 3 classes will be: ~[0.77257, 0.36445, 0.863]
            """
            N_frames = 381828
            loss_weight = [1.0 - (116254 / N_frames),  # TODO: what's happening here? loss_weight
                           1.0 - (222332 / N_frames),
                           1.0 - (43242 / N_frames)]  # None , or see above
            loss_weight = None
            flag_loss_W = False  # used only as string in the folder/run name
        else:
            print(f"dataset: {data_name} NOT supported yet")

    # --------------------
    # =========================

    # === Choose optimizer ===
    optim_type = "Adam"  # Adam, SGD
    optim_lr = 1e-6
    optim_weight_decay = 1e-9
    optim_momentum = 0  # for SGD: default=0
    # ------------------------

    # ===== Training parameter ======
    save_model = True  # save during train ; It always saves the last model on the last epoch  #!!!
    save_every = 1  # save a model every given number of epoches  #!!!
    num_epochs = 5  # 15  # number of total training epochs
    num_epochs_wearer = None  # number of epochs to train only the VAD-wearer classification. None|5

    av_combination_level = "in_unet_bottleneck"  # "before_unet", "in_unet_bottleneck"
    unet_size = 512  # 512/128/64  "small"  # "small"-128, "big"-512
    batch_before_stop = 100  # 200/500

    B = 4  # originally was 128
    # cuda_num, break_flag, B, num_epochs, unet_size, loss_type = get_parameters_from_user(device_ids,
    #                                                                                      MultiGPU,
    #                                                                                      break_flag,
    #                                                                                      B,
    #                                                                                      num_epochs,
    #                                                                                      unet_size,
    #                                                                                      loss_type)
    batch_train = B
    batch_val = B
    batch_test = B
    num_to_val = None  # None(=val all), or number of examples to val
    num_workers = 10  # usually I use 8, after running the script saw 10 is faster



    # --------------------------------------------------------------

    # use W&B for tracking the results
    # wandb.login()
    config = {
        "device_ids": device_ids,
        "cuda_num": cuda_num,
        "MultiGPU": MultiGPU,
        "n_GPU": len(device_ids),

        "choose_model": choose_model,

        # # # --- model vit CSD Transformer hyperparams
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

        # --- loss ---
        "loss_mix_weights": loss_mix_weights,
        "loss_type": loss_type,
        "weight_not_speaking_class": weight_not_speaking_class,
        # "LS"              : LS,
        # "cs_base_loss"    : cs_base_loss,
        # "cs_lambd"        : cs_lambd,
        "M_avg": M_avg,
        "loss_weight": loss_weight,

        # --- Optimizer ---
        "optim_type": optim_type,
        "optim_lr": optim_lr,
        "optim_weight_decay": optim_weight_decay,
        "optim_momentum": optim_momentum,

        # --- Data processing parameters -----
        "sr": sr,
        "num_fft": num_fft,
        "n_fft_effective": n_fft_effective,
        # "flag_log": flag_log,

        "video_n_channels": video_n_channels,
        "video_frame_h": video_frame_h,
        "video_frame_w": video_frame_w,

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
        'unet_size': unet_size,
        'batch_before_stop': batch_before_stop
    }
    return config


def test_calculate_precision():
    y_pred = torch.tensor([[0.9, 0.1], [0.7, 0.3]])
    y_true = torch.tensor([[1, 0], [0, 1]])

    # Calculate precision using the function
    precision = calculate_precision(y_pred, y_true)

    # Expected output: Precision: 0.75
    print("Precision:", precision)

    y_pred = torch.tensor([[0.6, 0.4], [0.8, 0.2]])
    y_true = torch.tensor([[1, 1], [0, 0]])

    # Calculate precision using the function
    precision = calculate_precision(y_pred, y_true)

    # Expected output: Precision: 0.5
    print("Precision:", precision)

    y_pred = torch.tensor([[0.2, 0.8, 0.4], [0.9, 0.3, 0.7]])
    y_true = torch.tensor([[0, 1, 1], [1, 0, 1]])

    # Calculate precision using the function
    precision = calculate_precision(y_pred, y_true)

    # Expected output: Precision: 0.8333333333333334
    print("Precision:", precision)


def test_calculate_accuracy():
    # Test case 1: y_pred and y are both tensors of shape (4, 2)
    y_pred = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1]])
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    expected_accuracy = 0.75

    accuracy = calculate_accuracy(y_pred, y_true)
    assert accuracy == expected_accuracy
    print("Test case 1 passed")

    # Test case 2: y_pred and y are both tensors of shape (3, 2)
    y_pred = torch.tensor([[0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
    y_true = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    expected_accuracy = 0.333333333333333333

    accuracy = calculate_accuracy(y_pred, y_true)
    print(f"{accuracy=}")
    assert accuracy == expected_accuracy
    print("Test case 2 passed")

    # Test case 3: y_pred and y are both tensors of shape (2, 2)
    y_pred = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
    y_true = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    expected_accuracy = 0.5

    accuracy = calculate_accuracy(y_pred, y_true)
    assert accuracy == expected_accuracy
    print("Test case 3 passed")

    print("All test cases passed!")


def func():
    # Your function's code here
    for _ in range(10**10):
        pass


def epoch_time(start, end):
    elapsed_time = end - start
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    return minutes, seconds


def test_epoch_time():
    start_time = time.time()  # Record the start time
    func()  # Call the function whose time you want to measure
    end_time = time.time()  # Record the end time

    minutes, seconds = epoch_time(start_time, end_time)

    print(f"Time taken by func(): {minutes} minutes {seconds:.2f} seconds")


def create_visualize_train_table():
    # File paths
    input_csv_path = "/dsi/scratch/from_netapp/users/elnatan_k/Table.csv"
    output_csv_path = "/dsi/scratch/from_netapp/users/elnatan_k/visualize_train.csv"

    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    # Select the first 200 rows
    subset_df = df.head(200)

    # Save the subset as a new CSV file
    subset_df.to_csv(output_csv_path, index=False)

    print("CSV file saved successfully.")


def create_one_batch_train_table(batch_size):
    # File paths
    input_csv_path = "/dsi/scratch/from_netapp/users/elnatan_k/Table.csv"
    output_csv_path = "/dsi/scratch/from_netapp/users/elnatan_k/one_batch_train.csv"

    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    # Select the first 'batch_size' rows
    subset_df = df.head(batch_size)

    # Save the subset as a new CSV file
    subset_df.to_csv(output_csv_path, index=False)

    print("CSV file saved successfully.")


def create_frames_with_speakers_table():
    # File paths
    input_csv_path = "/dsi/scratch/from_netapp/users/elnatan_k/Train_table.csv"
    output_csv_path = "/dsi/scratch/from_netapp/users/elnatan_k/speakers_train_table.csv"

    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    # Select the rows that the attribute "speakers_in_picture_IDs" isn't an empty list
    subset_df = df[df['speakers_in_picture_IDs'].apply(lambda x: len(eval(x)) > 0)]

    # Save the subset as a new CSV file
    subset_df.to_csv(output_csv_path, index=False)

    print("speakers_train_table CSV file saved successfully.")


def set_trainer_path1():   #TODO: fix this, there is space!
    # set saving name
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d")  # only date  #TODO: save by the minute!!!
    if config['choose_model'] == "av_model":
        str_model_details = f"{config['av_combination_level']}"
    if config['loss_type'] == "bce":
        str_model_details += f"bce_weight_{config['weight_not_speaking_class']}"
    else:
        str_model_details = f""
    # TODO: fix the names!
    save_name = f"{dt_string}_{config['data_name']}_"\
                 f"{config['choose_model']}_LossType_"\
                 f"{config['loss_type']}_{str_model_details}_"\
                 f"n_epochs_{config['num_epochs']}_"\
                 f"B_{config['batch_train']}_"\
                 f"unet_{config['unet_size']}"
    print(f"save_name:\n{save_name}")


def set_trainer_path():
    # set saving name
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y_%m_%d__%H_%m") # Include year, month, day, hour, and minute
    if config['choose_model'] == "av_model":
        str_model_details = f"{config['av_combination_level']}"
    if config['loss_type'] == "bce":
        str_model_details += f"bce_weight_{config['weight_not_speaking_class']}"
    else:
        str_model_details = f""
    save_name = f"{dt_string}_{config['data_name']}_" \
                f"{config['choose_model']}_LossType_" \
                f"{config['loss_type']}_{str_model_details}_" \
                f"n_epochs_{config['num_epochs']}_" \
                f"B_{config['batch_train']}_" \
                f"unet_{config['unet_size']}"
    print(f"save_name:\n{save_name}")


def count_speaker_pixels(input_csv_path):

    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    total_area = 0

    # Convert string representations to actual lists
    df["places_in_picture"] = df["places_in_picture"].apply(ast.literal_eval)

    for coordinates in df["places_in_picture"]:
        for rectangle in coordinates:
            x1, y1, x2, y2 = rectangle
            width = x2 - x1
            height = y2 - y1
            area = width * height
            total_area += area

    speaker_pixels_num = total_area
    pixels_num_in_frame = 1080*1920
    frame_num = df.shape[0]
    not_speaker_pixels_num = pixels_num_in_frame*frame_num - speaker_pixels_num
    return speaker_pixels_num, not_speaker_pixels_num


def count_wearer_active_frames(input_csv_path):
    df = pd.read_csv(input_csv_path)

    # Count the number of rows where is_wearer_active == 1
    active_count = len(df[df['is_wearer_active'] == 1])

    # Count the number of rows where is_wearer_active == 0
    inactive_count = len(df[df['is_wearer_active'] == 0])

    return active_count, inactive_count

def compare_directories(dir1, dir2):
    dcmp = dircmp(dir1, dir2)

    def print_diff(dcmp):
        for name in dcmp.left_only:
            print(f"Only in {dcmp.left}: {os.path.join(dcmp.left, name)}")
        for name in dcmp.right_only:
            print(f"Only in {dcmp.right}: {os.path.join(dcmp.right, name)}")
        for name in dcmp.diff_files:
            print(f"Differing file found: {os.path.join(dcmp.left, name)} and {os.path.join(dcmp.right, name)}")
        for sub_dcmp in dcmp.subdirs.values():
            print_diff(sub_dcmp)

    print_diff(dcmp)



def compare_directory_structure(dir1, dir2):
    def get_directory_structure(directory):
        structure = {}
        for root, dirs, files in os.walk(directory):
            rel_root = os.path.relpath(root, directory)
            structure[rel_root] = (dirs, files)
        return structure

    structure1 = get_directory_structure(dir1)
    structure2 = get_directory_structure(dir2)

    def compare_directory_structure(dir1, dir2):
        def compare_folders(folder1, folder2):
            items1 = set(os.listdir(folder1))
            items2 = set(os.listdir(folder2))

            for item in items1 - items2:
                print(f"Only in {dir1}: {os.path.join(folder1, item)}")
            for item in items2 - items1:
                print(f"Only in {dir2}: {os.path.join(folder2, item)}")

            common_items = items1 & items2
            for item in common_items:
                path1 = os.path.join(folder1, item)
                path2 = os.path.join(folder2, item)
                if os.path.isdir(path1) and os.path.isdir(path2):
                    print(f"Comparing directory: {path1} and {path2}")
                    compare_folders(path1, path2)
                elif os.path.isfile(path1) and os.path.isfile(path2):
                    continue
                else:
                    print(f"Different types: {path1} and {path2}")

        print(f"Comparing {dir1} and {dir2}")
        compare_folders(dir1, dir2)


def check_cm():
    cm = confusion_matrix(y_true=[1, 1], y_pred=[1, 1], labels=[1, 0])
    tp, fn, fp, tn = cm.ravel()
    print(cm)
    print(f"{tp=}, {fn=}, {fp=}, {tn=}")


def print_checkpoint(model_name):
    dir_root = "/dsi/scratch/from_netapp/users/elnatan_k"
    results_path = f"{dir_root}/new_results"  # trainer.dir_results #"/dsi/scratch/from_netapp/users/elnatan_k/results/"
    # "2023_09_14__10_32_LossType_bce_weight_None_pos_weight_speaker_segmentation_3.0_pos_weight_wearer_3.0_unet_size_512_before_unetonly_wearer_mode_n_epochs_5_B_16_color_mode_RGB"
    # run a lot of time - 21_35__24_08_2023_easycom_av_model_LossType_dc__n_epochs_5_B_4_unet_512
    model_path = f"{results_path}/{model_name}/model"
    print(f"{model_path=}")
    checkpoint = torch.load(f'{model_path}/checkpoint_epoch_4.tar')  # checkpoint_epoch_3.tar  FinalModel.tar
    # pprint.pprint(checkpoint, sort_dicts=False)
    print(checkpoint.keys())
    print(f"{checkpoint['acc_dict']['acc_all_test']['wearer']}")
    print(f"{checkpoint['recall_dict']['recall_all_test']['wearer']}")
    print(f"{checkpoint['precision_dict']['precision_all_test']['wearer']}")
    print(f"{checkpoint['f1_dict']['f1_all_test']['wearer']}")

    print(f"wearer test metric first and final results:")
    for epoch in [1, 4]:
        print(f"Epoch {epoch}:")
        for metric in ['acc', 'recall', 'precision', 'f1']:
            print(f"{metric}: {checkpoint[f'{metric}_dict'][f'{metric}_all_test']['wearer'][epoch - 1]}")


    # for key in ['acc_dict', 'f1_dict', 'recall_dict', 'precision_dict']: # 'bests_dict', 'loss_dict', 'iou_dict', 'tversky_dict', 'cm_dict'
    #     print(f"{key=}, {checkpoint[key]=}")
    #     print(f"{checkpoint[key][f'{key}_all_test']=}")
    #     print(f"wearer {key} :\n{checkpoint[key][f'{key}_all_test']['wearer']}\n")
    config = checkpoint['config']

def delete_empty_dirs():
    print("In delete_empty_dirs")
    # Define the root directory path
    root_directory = "/dsi/scratch/from_netapp/users/elnatan_k/results"  #new_results

    # Function to check if a directory is empty
    def is_directory_empty(directory):
        return not any(os.listdir(directory))

    # Iterate over all directories in the root directory
    deleted_dir_counter = 0
    dir_num = 0
    for dirpath, dirnames, filenames in os.walk(root_directory, topdown=False):
        for dirname in dirnames:
            # print(f"{dirname=}")
            dir_to_check = os.path.join(dirpath, dirname)
            if dirname == "model" and is_directory_empty(dir_to_check):
                deleted_dir_counter += 1
                print(f"{deleted_dir_counter}) ({dir_num}) Deleting empty directory in {dirpath}")
                shutil.rmtree(dirpath)
    print("Done")

if __name__ == "__main__":
    # config = get_config1()
    # test_calculate_precision()
    # test_calculate_accuracy()
    # test_epoch_time()
    #create_visualize_train_table()
    # set_trainer_path()
    # create_frames_with_speakers_table()

    # input_csv_path = "/dsi/scratch/from_netapp/users/elnatan_k/Train_table.csv"
    # speaker_pixels_num, not_speaker_pixels_num = count_speaker_pixels(input_csv_path)
    # print(f"{speaker_pixels_num=}, {not_speaker_pixels_num=}")
    # ratio = not_speaker_pixels_num / speaker_pixels_num
    # print(f"The ratio is {ratio}")
    #
    # input_csv_path = "/dsi/scratch/from_netapp/users/elnatan_k/speakers_train_table.csv"
    # speaker_pixels_num, not_speaker_pixels_num = count_speaker_pixels(input_csv_path)
    # print(f"{speaker_pixels_num=}, {not_speaker_pixels_num=}")
    # ratio = not_speaker_pixels_num / speaker_pixels_num
    # print(f"The ratio is {ratio}")

    for table_name in ['Train_table', 'Validation_table', 'Test_table']:
        input_csv_path = f"/dsi/scratch/from_netapp/users/elnatan_k/{table_name}.csv"
        wearer_active_frames_num, not_wearer_active_frames_num = count_wearer_active_frames(input_csv_path)
        print(f"In {table_name}:")
        print(f"{wearer_active_frames_num=}, {not_wearer_active_frames_num=}")
        ratio = not_wearer_active_frames_num / wearer_active_frames_num
        print(f"The ratio is {ratio}")

    # Example usage
    # dir1_path = '/dsi/scratch/from_netapp/users/roy_do/Video_3_ratio'
    # dir2_path = '/dsi/scratch/from_netapp/users/roy_do/Video_6_ratio'
    #compare_directories(dir1_path, dir1_path)
    # print("Starts compare_directory_structure")
    # compare_directory_structure(dir1_path, dir2_path)
    # batch_size = 16
    # create_one_batch_train_table(batch_size)

    # check_cm()
    # model_name = "2023_09_27__19_56_LossType_bce_weight_None_pos_weight_speaker_segmentation_100_pos_weight_wearer_3_unet_size_512_before_unet_n_epochs_15_B_16_color_mode_RGB"  # "2023_09_14__16_38_LossType_bce_weight_None_pos_weight_speaker_segmentation_0.5_pos_weight_wearer_0.5_unet_size_512_before_unet_only_wearer_mode_n_epochs_10_B_16_color_mode_RGB"
    # print_checkpoint(model_name)
    delete_empty_dirs()



















