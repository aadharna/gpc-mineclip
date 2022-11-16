import gym
from omegaconf import OmegaConf
import torch
from mineclip import MineCLIP
import torchvision.transforms as T
import torchvision.transforms as T

class MinedojoWrapper(gym.Wrapper):
    def __init__(self, env,prompts):
        super().__init__(env)
        cfg = OmegaConf.load("clip_conf_simple.yaml")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        OmegaConf.set_struct(cfg, False)
        ckpt = cfg.pop("ckpt")
        OmegaConf.set_struct(cfg, True)
        self.model = MineCLIP(**cfg).to(self.device)
        weights = torch.load(ckpt['path'])
        self.model.load_state_dict(weights,strict=False)
        self.prompt_feats = self.model.encode_text(prompts)
        self.transform = T.Resize((160,256))
        
    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        frame = next_state['rgb'].copy()
        frame = torch.from_numpy(frame).to(self.device)
        frame = self.transform(frame)
        frame = torch.unsqueeze(torch.unsqueeze(frame,dim=0),dim=0)
        image_feats = self.model.forward_image_features(frame)
        video_feats = self.model.forward_video_features(image_feats)
        reward, _ = self.model(video_feats, text_tokens=self.prompt_feats, is_video_features=True)
        return next_state, reward, done, info