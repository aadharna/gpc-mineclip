import torch
import itertools
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from mineclip import MineCLIP
import wandb

class VideoDataset(Dataset):
    def __init__(self,data):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)

def load_mineclip(device):

    # Initialize MineClip
    cfg = OmegaConf.load("clip_conf_simple.yaml")
    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)
    model = MineCLIP(**cfg).to(device)
    model.load_ckpt(ckpt.path,strict=True)
    model.eval()
    return model

def example_read_video(video_object, start=0, end=None):
    transform = T.Resize((160,256))
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )
        

    video_frames = torch.empty(0)
    video_pts = []
    video_object.set_current_stream("video")
    frames = []
    for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
        frames.append(transform(frame['data']))
        video_pts.append(frame['pts'])
    if len(frames) > 0:
        video_frames = torch.stack(frames, 0)

    return video_frames, video_pts, video_object.get_metadata()

def init_wandb(task_id,sample_rate=16):
    wandb.init(project="curriculum-from-recordings", entity="gcapable")
    wandb.config = {
    "task": task_id,
    "sample_rate": sample_rate
    }