import gym
from omegaconf import OmegaConf
import torch
from mineclip import MineCLIP
import torchvision.transforms as T
import torchvision.transforms as T

class MineClipWrapper(gym.Wrapper):
    def __init__(self, env,prompts):
        super().__init__(env)
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize MineClip
        cfg = OmegaConf.load("clip_conf_simple.yaml")
        OmegaConf.set_struct(cfg, False)
        ckpt = cfg.pop("ckpt")
        OmegaConf.set_struct(cfg, True)
        self.model = MineCLIP(**cfg).to(self.device)
        weights = torch.load(ckpt['path'])
        self.model.load_state_dict(weights,strict=False)
        self.model.eval()

        # Calculate Initial features for prompts and dummy frames 
        with torch.no_grad():
            self.prompt_feats = self.model.encode_text(prompts)
            initial_frames = torch.zeros((1, 16, 3, 160, 256),dtype=torch.int64, device=self.device)
            self.initial_feats =  self.model.forward_image_features(initial_frames)
            self.image_feats = self.initial_feats.clone().detach()

        #self.transform = T.Resize((160,256)) # Transformation to downsample image
        self.previous_reward = 0
        self.pi = 0
        
    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        with torch.no_grad():
            frame = next_state['rgb'].copy()
            frame = torch.from_numpy(frame).to(self.device)
            #frame = self.transform(frame)
            frame = torch.unsqueeze(torch.unsqueeze(frame,dim=0),dim=0)
            img_feats = self.model.forward_image_features(frame)
            self.image_feats = torch.cat((self.image_feats[:,1:,:],img_feats),dim=1)
            video_feats = self.model.forward_video_features(self.image_feats)
            prompt = torch.unsqueeze(self.prompt_feats[self.pi],dim=0)
            reward, _ = self.model(video_feats, text_tokens=prompt, is_video_features=True)
            delta_reward = reward- self.previous_reward
            self.previous_reward = reward
        return next_state, delta_reward.item(), done, info

    def change_prompt(self, index=None):
        if index is None:
            self.pi +=1
        else:
            self.pi = index

    def reset(self):
        self.image_feats = self.initial_feats.clone().detach()
        self.previous_reward = 0
        self.env.reset()

