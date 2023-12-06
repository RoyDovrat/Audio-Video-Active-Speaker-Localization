import pickle
import os
import torch
import torchaudio
import sys

problematic_files = ["28-00-634.wav", "10-00-423.wav", "21-00-563.wav", "11-00-446.wav", "09-00-429.wav",
                     "30-00-709.wav", "09-00-390.wav", "06-00-365.wav", "04-00-342.wav", "12-00-454.wav",
                     "15-00-480.wav", "17-00-478.wav", "11-00-420.wav", "06-00-370.wav", "28-00-653.wav",
                     "14-00-459.wav"]

EPS = sys.float_info.epsilon
FRAME_NUM = 1200

class PrepAudio:
    def __init__(self, dir_origin, dir_output, frame_num=1200, sr=16000, n_fft=512):

        self.dir_origin = dir_origin
        self.dir_audio = f"{dir_origin}/Glasses_Microphone_Array_Audio"  # audio_sessions_source_path
        self.dir_Speech_Transcriptions = f"{dir_origin}/Speech_Transcriptions"

        self.dir_output = f"{dir_output}/Audio"
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        self.sr = sr
        self.len_split = 0.05  # in seconds
        self.frame_num = frame_num
        self.n_fft = n_fft
        self.win = torch.hann_window(window_length=self.n_fft)  # , device=device

        self.table = []

    def prep_data(self):
        _, sub_folders_sessions, _ = next(os.walk(self.dir_audio))
        print(f"all sessions:\n {sorted(sub_folders_sessions)}")
        for session in sorted(sub_folders_sessions):
            self.prep_session(session=session)

    def prep_session(self, session):
        print(f"********************** {session} ****************************")

        dir_session = f"{self.dir_audio}/{session}"
        dir_session_output = f"{self.dir_output}/{session}"
        if not os.path.exists(dir_session_output):
            os.makedirs(dir_session_output)

        _, _, file_list = next(os.walk(dir_session))
        for file in sorted(file_list):
            self.prep_file(dir_session, dir_session_output, file)

    def prep_file(self, dir_session, dir_session_output, file):
        # Remove the .wav
        file_dir_name, _ = os.path.splitext(file)
        print(f"--------------- file {file.split('.')[0]} ----------------")

        file_output_path = f"{dir_session_output}/{file_dir_name}"
        if not os.path.exists(file_output_path):
            os.makedirs(file_output_path)

        # Open the WAV file
        file_path = os.path.join(dir_session, file)
        all_wavs, origin_sample_rate = torchaudio.load(file_path, normalize=False)

        downsampled = self.downsample_file(all_wavs, origin_sample_rate)
        downsampled_normalized = self.normalize(downsampled)
        segments = self.slice_the_file(downsampled_normalized)
        self.save_segments(segments, file_output_path)
        self.save_segments(segments, file_output_path)

    def downsample_file(self, signal, origin_sample_rate):
        # Convert the signal to a tensor
        signal = torch.tensor(signal)

        # Create a Resample transform with the target sample rate of 18000
        resample = torchaudio.transforms.Resample(origin_sample_rate, self.sr)
        # Apply the transform to the signal
        signal = signal.to(torch.float32)
        downsampled_signal = resample(signal)
        return downsampled_signal

    def normalize(self, all_wavs):
        # Norm by the max abs value
        max_value = torch.max(torch.abs(all_wavs))
        return 0.99 * all_wavs / (max_value + EPS)

    def slice_the_file(self, file):
        segments = file.unfold(dimension=1, size=int(self.sr * self.len_split), step=int(self.sr * self.len_split))
        return torch.permute(segments, (1, 0, 2))

    def save_segments(self, segments, file_output_path):
        segments = segments.numpy()

        path_to_check_dir = "/dsi/scratch/from_netapp/users/elnatan_k/check_dir"

        for i, segment in enumerate(segments):
            audio_temp_segment_path = os.path.join(path_to_check_dir, f"seg{i}.pickle")
            with open(audio_temp_segment_path, 'wb') as handle:
                pickle.dump(segment, handle)
            audio_segment_path = os.path.join(file_output_path, f"seg{i}.pickle")
            self.save_audio_smaller_pickle_instead_of_original(audio_old_segment_path=audio_temp_segment_path,
                                                               audio_new_segment_path=audio_segment_path)

    def save_audio_smaller_pickle_instead_of_original(self, audio_old_segment_path, audio_new_segment_path):
        with open(audio_old_segment_path, 'rb') as handle:
            data = pickle.load(handle)
        os.remove(audio_old_segment_path)

        torch_seg = torch.from_numpy(data)

        # saving
        with open(audio_new_segment_path, 'wb') as handle:
            torch_seg = torch_seg.detach().cpu()
            pickle.dump(torch_seg, handle)


if __name__ == "__main__":
    dir_origin = "/dsi/gannot-lab/datasets2/EasyCom_fb_AR_Glasses/Main"
    dir_Speech_Transcriptions = f"{dir_origin}/Speech_Transcriptions"
    dir_output = "/dsi/scratch/from_netapp/users/elnatan_k"

    prep = PrepAudio(dir_origin, dir_output)
    prep.prep_data()





