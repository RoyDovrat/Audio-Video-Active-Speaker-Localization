import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
from torchvision.models import ResNet18_Weights
from models import UNET
import sys
import utils

EPS = sys.float_info.epsilon

class av_model(nn.Module):
    def __init__(self,
                 data_param_audio,
                 data_param_video,
                 num_classes=2,
                 flag_log=True,
                 device="cpu",
                 size_fitting="reduce_video",
                 av_combination_level="before_unet",
                 unet_size=512,
                 only_wearer_mode=False):

        """
        Args:
            input params:
            n_mics (int): number of input mics
            n_fft (int): number of fft bins
            n_time (int): number of time bins

            Model type params:
            flag_log (bool): using log10()  on the input
            multi_channel (bool): using only single channel, or all the available input channels

            feature_type: "stft"

            Classifier params:
            num_classes (int): number of classes for classification head


        """
        super().__init__()

        self.data_param_audio = data_param_audio
        self.data_param_video = data_param_video

        self.device = device

        #  --- Audio  ----
        # audio-data properties
        self.audio_n_mics = data_param_audio['n_mics']
        self.audio_n_fft  = data_param_audio['n_fft']
        self.audio_fs     = data_param_audio['fs']

        # ---- Video ---
        # video-data properties
        self.video_n_frames   = data_param_video['n_frames']
        self.video_n_channels = data_param_video['n_channels']
        self.video_frame_h    = data_param_video['frame_h']
        self.video_frame_w    = data_param_video['frame_w']
        self.video_downsample_ratio = data_param_video['downsample_ratio']

        self.c = 40
        if self.video_downsample_ratio == 6:
            self.c = 20

        # --------------------------------------------------------------

        self.size_fitting = size_fitting
        self.av_combination_level = av_combination_level
        self.only_wearer_mode = only_wearer_mode
        self.unet_size = unet_size

        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the first layer of the model
        if self.video_n_frames < 7:
            kernel_size = 3
        else:
            kernel_size = 7
        self.resnet.conv1 = nn.Conv2d(self.audio_n_mics, 64, kernel_size=kernel_size)
        self.fc1 = nn.Linear(512, 2)

        if self.av_combination_level == "before_unet":
            self.fc_before_video = nn.Linear(512, self.video_frame_h*self.video_frame_w)   #fc2
        elif self.av_combination_level == "in_unet_bottleneck":
            # todo: why 40? #TODO: calculate
            self.fc_before_video = nn.Linear(512, self.data_param_audio['n_time']*self.c)

        in_channels = self.video_n_frames
        self.conv3d = torch.nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1)

        if self.av_combination_level == "before_unet":
            in_channels = self.video_n_channels + 1
        elif self.av_combination_level == "in_unet_bottleneck":
            in_channels = self.video_n_channels

        features = []
        feature = unet_size
        for i in range(4):
            features.insert(0, feature)
            feature = feature // 2
        #print(f"@@@@@@@@@@@@@@@{features=}")

        self.unet = UNET(in_channels=in_channels,
                         out_channels=2,
                         features=features,
                         av_combination_level=self.av_combination_level)
        # --------------------

        # General hyperparameters for the model
        self.flag_log = flag_log

        self.num_classes = num_classes

        # -----------------------

    # ----- Perp methods --------
    def prep_audio(self, input_data_audio):
        """
        Prep the audio input features to the model
        """

        input_data_audio = input_data_audio.abs()

        if self.flag_log is True:
            input_data_audio = torch.log10(input_data_audio + EPS)

        return input_data_audio

    def prep_video(self, input_data_video):
        return input_data_video

    def concat_av_by_duplicate_audio(self, audio, video):
        # Create an empty list to store the concatenated tensors
        concatenated_tensors = []

        # Iterate over the first dimension (batch) of the tensors
        for i in range(audio.size(0)):
            audio_slice = audio[i, :, :]
            video_slice = video[i, :, :, :, :]

            # Reshape tensor1 to have the same shape as tensor2
            audio_slice_reshaped = audio_slice.unsqueeze(0).unsqueeze(0).repeat(video.size(1), 1, 1, 1)
            #print(f"{audio_slice_reshaped.shape=}")

            # Concatenate the tensors along the second dimension (dimension 1)
            concatenated_tensor = torch.cat([audio_slice_reshaped, video_slice], dim=1)

            # Append the concatenated tensor to the list
            concatenated_tensors.append(concatenated_tensor)

        # Create a new tensor by stacking the concatenated tensors along the first dimension
        concatenated_av_tensors = torch.stack(concatenated_tensors, dim=0)
        return concatenated_av_tensors

    def concat_av_by_reduce_video(self, audio, video):
        # Create an empty list to store the concatenated tensors
        concatenated_tensors = []

        # Iterate over the first dimension (batch) of the tensors
        for i in range(audio.size(0)):
            audio_slice = audio[i, :, :]
            video_slice = video[i, :, :, :, :]

            audio_slice = audio_slice.unsqueeze(0)
            # print(f"{audio_slice.shape=}")

            video_slice = self.conv3d(video_slice)
            # print(f"{video_slice.shape=}")
            video_slice = video_slice.view(self.data_param_video['n_channels'], self.video_frame_h, self.video_frame_w)
            # print(f"After reshape: {video_slice.shape=}")

            av_slice_tensor = torch.cat((audio_slice, video_slice), dim=0)
            # print(f"{av_slice_tensor.shape=}")

            # Append the concatenated tensor to the list
            concatenated_tensors.append(av_slice_tensor)

        # Create a new tensor by stacking the concatenated tensors along the first dimension
        concatenated_av_tensors = torch.stack(concatenated_tensors, dim=0)
        return concatenated_av_tensors

    def audio_representation(self, audio):

        audio = self.resnet.conv1(audio)
        audio = self.resnet.bn1(audio)
        audio = self.resnet.relu(audio)
        audio = self.resnet.maxpool(audio)
        audio = self.resnet.layer1(audio)
        audio = self.resnet.layer2(audio)
        audio = self.resnet.layer3(audio)
        audio = self.resnet.layer4(audio)
        audio = self.resnet.avgpool(audio)
        audio = audio.view(audio.size(0), -1)  # Reshape audio tensor
        return audio

    def forward_features(self, input_data_audio, input_data_video):
        """
        forward_features get the 2 input data (audio and video)
        and returns the feature vector for the classification head
        """
        B = input_data_audio.shape[0]

        audio = self.audio_representation(input_data_audio)

        vad_wearer = self.fc1(audio)

        if self.only_wearer_mode:
            places_preds = torch.zeros(torch.Size([16, 2, 180, 320])).to(self.device)
            return vad_wearer, places_preds

        if self.av_combination_level == "before_unet":
            audio = self.fc_before_video(audio).view(B, self.video_frame_h, self.video_frame_w)
            # print(f"After fc_before_video and reshape: {audio.shape}")

            if self.size_fitting == "dup_audio":
                # duplicate audio 7 times to fit the video dimensions
                # Output dimensions: (B, 7, 4, H, W) (7 depends on the context frames)
                av_tens = self.concat_av_by_duplicate_audio(audio=audio, video=input_data_video)
            elif self.size_fitting == "reduce_video":
                # Output dimensions: (B, 1, 4, H, W)
                av_tens = self.concat_av_by_reduce_video(audio=audio, video=input_data_video)

            places_preds = self.unet(av_tens).view(B, self.unet.out_channels, self.video_frame_h, self.video_frame_w)

        elif self.av_combination_level == 'in_unet_bottleneck':
            print("in_unet_bottleneck")
            video = self.conv3d(input_data_video)
            video = video.view(B, self.data_param_video['n_channels'], self.video_frame_h, self.video_frame_w)
            audio = self.fc_before_video(audio).view(B, self.data_param_audio['n_time'], self.c)  # 40
            places_preds = self.unet(video, audio)

        return vad_wearer, places_preds

    def forward(self, input_data_audio=None, input_data_video=None):
        # Prep the different inputs to the model
        input_data_audio = self.prep_audio(input_data_audio)
        input_data_video = self.prep_video(input_data_video)

        # Get the feature vector before classification
        places_preds, vad_wearer_preds = self.forward_features(input_data_audio, input_data_video)

        return places_preds, vad_wearer_preds


if __name__ == '__main__':
    device_ids = [3]  # [2, 5, 6, 7]   # choose cuda number for deploying the model
    cuda_num = device_ids[0]  # device must be the first device_ids
    device = utils.choose_cuda(cuda_num)

    # ------- Input data param's -------
    batch = 4
    # --- Audio:
    data_param_audio = {
        'n_mics': 6,
        'n_fft': 257,
        'n_time': 22,
        'fs': 16000,
    }

    # --- Video:
    color_mode = "grayscale"  # "grayscale"/"RGB"
    if color_mode == "grayscale":
        video_n_channels = 1
    else:
        video_n_channels = 3
    data_param_video = {
        'n_frames': 7,  # 1 + [3,3] context frames
        'n_channels': video_n_channels,  # RGB
        'frame_h': 360,
        'frame_w': 640,
    }
    # ---------------------------

    # General
    num_classes = 2  # Number of classes for the CSD head
    flag_log = True
    # ---------------------------------------

    av_combination_level = "before_unet"  #" in_unet_bottleneck", "before_unet"

    # Define the model
    model = av_model(data_param_audio=data_param_audio,
                     data_param_video=data_param_video,
                     num_classes=num_classes,
                     flag_log=flag_log,
                     device=device,
                     size_fitting="reduce_video",
                     av_combination_level=av_combination_level).to(device)

    # Define the input data
    input_size_video = [batch,
                        data_param_video['n_frames'], data_param_video['n_channels'],
                        data_param_video['frame_h'], data_param_video['frame_w']]
    test_input_video = torch.rand(input_size_video).to(device)

    input_size_audio = [batch, data_param_audio['n_mics'], data_param_audio['n_fft'], data_param_audio['n_time']]
    test_input_audio = torch.rand(input_size_audio, dtype=torch.cfloat).to(device)  # complex data

    # Get logits from model
    model.eval()
    logits_wearer, logits_num_active = model(input_data_audio=test_input_audio, input_data_video=test_input_video)

    # Print input and output shape
    # print(f"input shape is:{test_input_audio.shape}")
    # print(f"input shape is:{test_input_video.shape}")
    # print(f"logits_wearer shape is:{logits_wearer.shape}")
    # print(f"logits_wearer shape is:{logits_num_active.shape}")
    # =====================================

    # --- Print model summary ---
    print(f"\n")
    col_names = ["input_size", "output_size", "num_params"]
    summary(model, input_size=[input_size_audio, input_size_video], col_names=col_names)  # set input_size to tuple for [audio, video]
    """
    Total #param is:

    """
    # =====================================
