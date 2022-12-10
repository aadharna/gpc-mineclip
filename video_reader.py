import os
import torch
import torchvision
from torch.utils.data import DataLoader
import wandb
import yaml
from video_utils import (VideoDataset,
                         init_wandb,
                         load_mineclip,
                         example_read_video)

STREAM = 'video'

if __name__ == '__main__':
    import argparse
    # get the task name from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mineclip')
    parser.add_argument('--weight_type', type=str, default='attn')
    args = parser.parse_args()
    TASK = args.task

    with open("tasks_conf.yaml", "r") as stream:
        try:
            conf = yaml.safe_load(stream)[TASK]
        except yaml.YAMLError as exc:
            print(exc)

    PATH = os.path.join(os.getcwd(), 'recordings', '{}.mp4'.format(conf['name']))
    prompts = conf['prompts']

    init_wandb(task_id=conf['name'], weight_type=args.weight_type)
    prompt_ids = ["Sub-Task {}".format(x+1) for x in range(len(prompts))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video = torchvision.io.VideoReader(PATH, STREAM)
    vf, info, meta = example_read_video(video)
    dataset = VideoDataset(vf)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False
    )
    model = load_mineclip(device)
    with torch.no_grad():
        prompt_feats = model.encode_text(prompts)
    for data in loader:
        data = torch.unsqueeze(data, dim=0).to(device)
        with torch.no_grad():
            reward, _ = model(data, text_tokens=prompt_feats, is_video_features=False)
            reward /= torch.exp(model.clip_model.logit_scale)
        log_data = dict(zip(prompt_ids, reward[0].cpu().numpy()))
        wandb.log(log_data)
