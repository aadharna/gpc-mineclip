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


class MonitorAndSwitchRewardFn(gym.Wrapper):
    def __init__(self, env, window_size=25, subtask_solved_threshold=0.95):
        super().__init__(env)
        assert(isinstance(env, MineClipWrapper))
        self.env = env
        self.subtask_solved_threshold = subtask_solved_threshold
        self.running_average_class = StreamingMovingAverage(window_size)
        self.running_average = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # sets the self.running_average variable
        self.running_average = self.running_average_class.process(reward)

        if self.running_average >= self.subtask_solved_threshold:
            try:
                self.env.change_prompt()
            except IndexError:
                # we have exhausted all the tasks in the prompt list
                done = True

        return next_state, reward, done, info

    def reset(self):
        return self.env.reset()

    def get_running_average(self):
        return self.running_average


# calculate a running average without using np.mean every time on a list
class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)
