from abc import ABC, abstractmethod
import cv2
import glob
import os


def get_frames_loader(name):
    if name.lower() == "videoloader":
        return VideoLoader
    elif name.lower() == "sequenceloader":
        return SequenceLoader
    else:
        raise ValueError(f"No such video loader: {name}")

class Loader(ABC):
    @abstractmethod
    def read_first_frame(self):
        pass

    @abstractmethod
    def get_next_frame(self):
        pass

    
class VideoLoader(Loader):
    def __init__(self, video_path):
        self.video_path = video_path
        if not os.path.isfile(video_path):
            raise ValueError(f"video path: {video_path} does not exist")
        self.cap = cv2.VideoCapture(video_path)
        self.width  = int(self.cap.get(3)) // 4 # width
        self.height = int(self.cap.get(4)) // 4 # height
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        
    def read_first_frame(self):
        # Read first frame
        success, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # quit if unable to read the video file
        if not success:
            raise Exception('Failed to read video first frame')
        return frame
            
    def get_next_frame(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                self.cap.release()
#                 output.release()
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame


class SequenceLoader(Loader):
    def __init__(self, sequence_folder):
        self.sequence_folder = sequence_folder
        if not os.path.isdir(sequence_folder):
            raise ValueError(f"sequence folder {sequence_folder} is not valid")
        self.frames = sorted([file for file in glob.glob(f'{sequence_folder}/*.jpg')])
        frame = cv2.imread(self.frames[0])
        self.height, self.width, _ = frame.shape
        self.fps = 30
        
    def read_frame(self,index):
        frame = cv2.imread(self.frames[index])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def read_first_frame(self):
        return self.read_frame(0)
            
    def get_next_frame(self):
        for i in range(1, len(self.frames)):
            yield self.read_frame(i)
            

