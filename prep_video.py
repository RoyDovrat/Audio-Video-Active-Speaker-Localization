import pickle
import os
import json
import torch
import torchvision
import sys

problematic_files = ["28-00-634.wav", "10-00-423.wav", "21-00-563.wav", "11-00-446.wav", "09-00-429.wav",
                     "30-00-709.wav", "09-00-390.wav", "06-00-365.wav", "04-00-342.wav", "12-00-454.wav",
                     "15-00-480.wav", "17-00-478.wav", "11-00-420.wav", "06-00-370.wav", "28-00-653.wav",
                     "14-00-459.wav"]

EPS = sys.float_info.epsilon
FRAME_NUM = 1200


class PrepVideo:
    def __init__(self,
                 dir_origin,
                 dir_output,
                 downsampled_ratio=3,
                 frame_num=1200,
                 sr=16000,
                 H=1080,
                 W=1920):

        self.dir_origin = dir_origin
        self.dir_video = f"{dir_origin}/Video_Compressed"  # video_sessions_source_path
        self.dir_Speech_Transcriptions = f"{dir_origin}/Speech_Transcriptions"

        self.dir_output = f"{dir_output}/Video_{downsampled_ratio}_ratio"   # /Video"
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        self.sr = sr
        self.len_split = 0.05  # in seconds
        self.frame_num = frame_num

        self.H = H
        self.W = W
        self.downsampled_ratio = downsampled_ratio


    def prep_data(self):
        _, sub_folders_sessions, _ = next(os.walk(self.dir_video))
        print(f"all sessions:\n {sorted(sub_folders_sessions)}")

        for session in sorted(sub_folders_sessions):
            self.prep_session(session=session)

    def prep_session(self, session):
        if session != 'Session_12': #'Session_4', 'Session_6', 'Session_7', 'Session_12':
            return
        print(f"********************** {session} ****************************")

        dir_session_output = f"{self.dir_output}/{session}"
        dir_session = f"{self.dir_video}/{session}"
        if not os.path.exists(dir_session):
            os.makedirs(dir_session)

        _, _, file_list = next(os.walk(dir_session))
        for file in sorted(file_list):
            self.prep_file(dir_session, dir_session_output, file)

    def prep_file(self, dir_session, dir_session_output, file):
        # Remove the .wav
        file_dir_name, _ = os.path.splitext(file)
        print(f"--------------- file {file.split('.')[0]} ----------------")

        if file_dir_name != '01-00-273':
            return
        else:
            print(f"In {file_dir_name}!")


        file_output_path = f"{dir_session_output}/{file_dir_name}"
        if not os.path.exists(file_output_path):
            os.makedirs(file_output_path)

        # Open the MP4 file
        file_path = os.path.join(dir_session, file)
        print("session=", dir_session.split('/')[-1])
        video, audio, fps = torchvision.io.read_video(file_path)
        video = self.cut_frames_side(video)
        new_H, new_W = self.H//self.downsampled_ratio, self.W//self.downsampled_ratio
        downsampled = self.downsample_video(video, new_H=new_H, new_W=new_W)
        self.save_frames(downsampled, file_output_path)

    def cut_frames_side(self, video):
        return video[:, :, -1920:, :]

    def downsample_video(self, video, new_H=360, new_W=640):
        video = torch.permute(video, (0, 3, 1, 2))
        new_size = (new_H, new_W)
        downsampled = torchvision.transforms.Resize(size=new_size, antialias=None)(video)
        downsampled = torch.permute(downsampled, (0, 2, 3, 1))
        return downsampled

    def save_frames(self, video, file_output_path):
        # slice the data
        frames_num, _, _, _ = video.shape
        video = video.numpy()
        path_to_check_dir = "/dsi/scratch/from_netapp/users/elnatan_k/check_dir"

        for i in range(frames_num):
            segment = video[i]
            temp_segment_path = os.path.join(path_to_check_dir, f"seg{i}.pickle")
            with open(temp_segment_path, 'wb') as handle:
                pickle.dump(segment, handle)
            segment_path = os.path.join(file_output_path, f"seg{i}.pickle")
            self.save_smaller_pickle_instead_of_original(path=temp_segment_path,  new_name=segment_path)

    def save_smaller_pickle_instead_of_original(self, path, new_name):
        with open(path, 'rb') as handle:
            data = pickle.load(handle)

        os.remove(path)

        torch_seg = torch.from_numpy(data)
        # saving
        with open(new_name, 'wb') as handle:
            torch_seg = torch_seg.detach().cpu()
            pickle.dump(torch_seg, handle)

    def speakersInFrame(self, session, file):
        json_file_name = file.split('.')[0] + ".json"
        print(f"{json_file_name=}")
        Speech_Transcriptions_file = f"{dir_Speech_Transcriptions}/{session}/{json_file_name}"
        f = open(Speech_Transcriptions_file, 'r')
        cols = ["frame_num", "speakers_num"]
        rows = []

        # returns JSON object as a dictionary
        wav_file_name = json_file_name.split('.')[0] + ".wav"
        print(f"{wav_file_name=}")
        if wav_file_name in problematic_files:
            return "ERROR"
        data = json.load(f)

        SpeakersInFrame = [set() for i in range(FRAME_NUM)]

        for i in range(FRAME_NUM):
            for line in data[:]:
                if line['Start_Frame'] <= i <= line['End_Frame']:
                    SpeakersInFrame[i].add(line['Participant_ID'])
            rows.append({"frame_num": i, "speakers_num": len(SpeakersInFrame[i])})
        f.close()

        SpeakersNumInFrame = [len(SpeakersInFrame[i]) for i in range(FRAME_NUM)]
        return SpeakersNumInFrame


if __name__ == "__main__":
    dir_origin = "/dsi/gannot-lab2/datasets2/EasyCom_fb_AR_Glasses/Main"
    dir_Speech_Transcriptions = f"/{dir_origin}/Speech_Transcriptions"
    dir_output = "/dsi/scratch/from_netapp/users/roy_do"

    downsampled_ratio = 6  # 6
    prep = PrepVideo(dir_origin, dir_output, downsampled_ratio)
    prep.prep_data()

