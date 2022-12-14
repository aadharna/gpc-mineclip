import gym
import torch
import torch.nn as nn
import numpy as np

import minedojo
from wrappers import MineClipWrapper
from kl_class import ActionSmoothingLoss

from mineagent import MultiCategoricalActor, SimpleFeatureFusion
from mineagent import features as F
# from mineagent.batch import Batch
from tianshou.data import Batch

import hydra
import wandb
from buffer import ReplayMemory
# import moviepy, imageio # Not used in script but used in background by wandb for logging videos, do pip install moviepy imageio

from collections import namedtuple


DEBUG = False

class Critic(nn.Module):
    def __init__(self, feature_net, **kwargs):
        super().__init__()
        self.feature_net = feature_net
        hidden_depth = kwargs.pop("hidden_depth")
        hidden_dim = kwargs.pop("hidden_dim")
        self.value_net = nn.Sequential(
            nn.Linear(feature_net.output_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_depth - 1)],
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs):
        x, _ = self.feature_net(obs)
        return self.value_net(x)
        

class ActorCritic(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        feature_net_kwargs = cfg.feature_net_kwargs
        feature_net = {}
        for k, v in feature_net_kwargs.items():
            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(F, cls)
            feature_net[k] = cls(**v, device=device)
        feature_fusion_kwargs = cfg.feature_fusion
        feature_net = SimpleFeatureFusion(
            feature_net, **feature_fusion_kwargs, device=device
        )
        self.actor = MultiCategoricalActor(
            feature_net,
            action_dim=[3, 3, 4, 25, 25, 8],
            device=device,
            **cfg.actor,
        ).to(device)
        self.critic = Critic(feature_net, **cfg.critic).to(device)
    
    def get_value(self, obs):
        return self.critic(obs.obs)
        
    def get_action_and_value(self, obs, action=None, deterministic=False):
        # import ipdb; ipdb.set_trace()
        action_dims = self.actor._action_dim
        logits = self.actor(obs.obs)[0] # torch.Size([1, 68])
        probs = self.actor.dist_fn(logits)
        if action == None:
            action = probs.sample()
        if deterministic:
            action = probs.mode()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs.obs), logits
    
def convert_action_space(action):
    """
    Convert raw action 
    """
    
def preprocess_obs(obs, info, prev_action_logits, device):
    """
    Convert raw env obs to agent obs
    """
    
    B = 1
    
    compass = torch.cat((
        torch.sin(torch.tensor(obs["location_stats"]["yaw"])), 
        torch.sin(torch.tensor(obs["location_stats"]["pitch"])),
        torch.cos(torch.tensor(obs["location_stats"]["yaw"])),
        torch.cos(torch.tensor(obs["location_stats"]["pitch"]))
    ))
    gps = obs["location_stats"]["pos"] #(3, )
    voxels = obs["voxels"]['block_meta'] #(3, 3, 3)
    biome_id = obs["location_stats"]["biome_id"] #(1, )
    prompt = info["prompt"]
    image = info["img_feats"]
    prev_action = prev_action_logits
    if DEBUG:
        import ipdb; ipdb.set_trace()
    
    obs = {
        # "compass": torch.tensor(compass.reshape(B, 4), device=device),
        "compass": compass.clone().detach().reshape(B,  4),
        "gps": torch.tensor(gps.reshape(B, 3), device=device),
        "voxels": torch.tensor(
            voxels.reshape(B, 3 * 3 * 3), dtype=torch.long, device=device
        ), 
        "biome_id": torch.tensor(
            biome_id.reshape(B, ), dtype=torch.long, device=device
        ), 
        "prev_action": torch.tensor(prev_action.reshape(B, 68), device=device),  
        # "prompt": torch.tensor(prompt.reshape(B, 512), device=device), 
        "prompt": prompt.clone().detach().reshape(B, 512),
        "image": image.clone().detach().reshape(B, 512)
    }
    
    return Batch(obs=obs)
    
def transform_action(action):
    """
    Map agent action to env action.
    """
    assert action.ndim == 2
    action = action[0]
    action = action.cpu().numpy()
    if action[-1] != 0 or action[-1] != 1 or action[-1] != 3:
        action[-1] = 0
    action = np.concatenate([action, np.array([0, 0])])
    return action
        

@hydra.main(config_name="ppo_conf", config_path=".", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg.experiment.batch_size = cfg.experiment.n_envs * cfg.experiment.rollout_length
    
    run = wandb.init(
        project=cfg.experiment.wandb_project_name, 
        entity="gcapable", 
        config=cfg
        )
    
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    
    # Set up the environment TODO: vectorize to have n_envs > 1
    env = minedojo.make(
        task_id="harvest_wool_with_shears_and_sheep",
        image_size=(160, 256),
        world_seed=123,
        seed=42,
    )

    prompts = [
        "find a sheep",
        "right-click on the sheep with your hand to interact with it",
        "when the sheep's health bar appears, wait for it to turn white",
        "left-click on the sheep to shear it",
        "collect the wool that appears",
    ]
    env = MineClipWrapper(env, prompts)
    action_space = gym.spaces.MultiDiscrete((3,3,4,25,25,8))
    actionSmoothingLoss = ActionSmoothingLoss(act_space= action_space,
                                              reduction='batchmean',
                                              log_target=True)
    
    # Set up the agent
    agent = ActorCritic(cfg, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.experiment.learning_rate)
    # TODO update capacity from hardcode
    memory = ReplayMemory(capacity=100000,si_counter=5)
    Rollout = namedtuple('Rollout',
                        ('states', 'actions'))
    
    # Set up storage
    # TODO: see if we can use tensor here with Batch
    # obs_buf = ReplayBuffer(size=cfg.experiment.rollout_length)
    obs = np.empty((cfg.experiment.rollout_length, cfg.experiment.n_envs), dtype=object) # rollout_length x 1
    # FIXME: check dim of transformed action
    actions = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs, 6)).to(device)
    prev_action_logits = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs, 68)).to(device)
    logprobs = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs)).to(device)
    rewards = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs)).to(device)
    dones = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs)).to(device)
    values = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs)).to(device)

    # Training
    timestep = 0
    env.reset()
    action = env.action_space.no_op()
    next_obs, reward, next_done, info = env.step(env.action_space.no_op())
    next_obs = preprocess_obs(next_obs, info, torch.zeros((1, 68)), device)
    if DEBUG: 
        import ipdb; ipdb.set_trace()
    num_updates = int(cfg.experiment.total_timesteps // cfg.experiment.batch_size)
    
    for i in range(num_updates):
        if cfg.experiment.anneal_lr:
            frac = 1.0 - (i - 1.0) / num_updates
            lrnow = frac * cfg.experiment.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        frame_buffer = []
        SAVE_GIF = num_updates % cfg.experiment.gif_interval == 0
        if SAVE_GIF:
            # initialize image buffer for gif
            frame_buffer = []
        
        for step in range(0, cfg.experiment.rollout_length):
            timestep += 1 * cfg.experiment.n_envs
            # obs[step] = next_obs
            obs[step]= (next_obs)
            dones[step] = next_done
            
            # Collect rollout
            with torch.no_grad(): 
                # import ipdb; ipdb.set_trace()
                action, logprob, ent, value, logits = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            # import ipdb; ipdb.set_trace()
            prev_action_logits[step] = logits
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, done, info = env.step(transform_action(action))
            if SAVE_GIF:
                frame_buffer.append(next_obs["rgb"])
                
            if timestep % cfg.experiment.log_interval == 0:
                wandb.log({"reward": reward}, step=timestep)
                wandb.log({"prompt": env.prompts[env.pi]}, step=timestep)
                if abs(reward) > cfg.experiment.obs_save_threshold:
                    images = wandb.Image(np.transpose(next_obs["rgb"], (1, 2, 0)), caption=f"reward spike: {reward}")
                    wandb.log({"examples": images})
                              
            next_obs = preprocess_obs(next_obs, info, logits, device)
            rewards[step] = torch.tensor(reward, device=device)
        
        if SAVE_GIF:
            # wandb.log(
            #     {f"rollout_gif": wandb.Video(np.transpose(np.array(frame_buffer), (0, 2, 3, 1)), fps=30, format="gif")},
            #     step=timestep)
            wandb.log({"video": wandb.Video(np.array(frame_buffer), fps=30)}, step=timestep)
            
        # Compute returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.experiment.rollout_length)):
                if t == cfg.experiment.rollout_length - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.experiment.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.experiment.gamma * cfg.experiment.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        b_obs = obs.reshape(-1)
        if DEBUG: 
            import ipdb; ipdb.set_trace()
        b_actions = actions.reshape(-1, 6)
        b_prev_action_logits = prev_action_logits.reshape(-1, 68)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if rewards.sum().item()  > memory.average_reward:
            memory.push(b_obs, b_actions.long())
        memory.average_reward = (memory.average_reward*i + rewards.sum().item())/(i+1)
        
        b_inds = np.arange(cfg.experiment.batch_size)
        clipfracs = []
        for epoch in range(cfg.experiment.n_train_epochs):
            np.random.shuffle(b_inds)
            minibatch_size = cfg.experiment.batch_size // cfg.experiment.n_minibatches
            for start in range(0, cfg.experiment.batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue, logits = agent.get_action_and_value(Batch(b_obs[mb_inds]), b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs = [((ratio - 1.0).abs() > cfg.experiment.clip_coef).float().mean().item()]
                    
                mb_advantages = b_advantages[mb_inds]
                if cfg.experiment.normalize_advantage:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -ratio * mb_advantages
                pg_loss2 = -torch.clamp(ratio, 1.0 - cfg.experiment.clip_coef, 1.0 + cfg.experiment.clip_coef) * mb_advantages
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.flatten()
                if cfg.experiment.clip_value_loss:
                    # Clip value to reduce variability during critic training
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.experiment.clip_coef,
                        cfg.experiment.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                # Action Smoothing Loss
                a_loss = 0
                if cfg.experiment.action_smoothing:
                    for mb_ind, action_logit in zip(mb_inds, b_prev_action_logits[mb_inds]):
                        start_window = max(0, mb_ind - cfg.experiment.action_smoothing_window)
                        # take the preceding n actions, take all the logits for these actions
                        if mb_ind == 0:
                            preceding_action_logits = b_prev_action_logits[0, :]
                        else:
                            preceding_action_logits = b_prev_action_logits[start_window:mb_ind, :]
                        a_loss += actionSmoothingLoss(action_logit, preceding_action_logits)
                a_loss /= len(mb_inds)

                loss = pg_loss - entropy_loss * cfg.experiment.entropy_coef + v_loss * cfg.experiment.value_loss_coef - a_loss * cfg.experiment.action_smoothing_coef
                # Optimizer step
                optimizer.zero_grad()
                loss.backward()
                # global gradient clipping
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.experiment.max_grad_norm)
                optimizer.step()
            
            # if cfg.experiment.target_kl is not None:
            #     if approx_kl > cfg.experiment.target_kl:
            #         break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Log to wandb  
        wandb.log({"learnging_rate": optimizer.param_groups[0]['lr']}, step=timestep)
        wandb.log({"loss.policy_loss": pg_loss.item()}, step=timestep)
        wandb.log({"loss.value_loss": v_loss.item()}, step=timestep)
        wandb.log({"loss.action_smoothing_loss": a_loss.item()}, step=timestep)
        wandb.log({"loss.entropy": entropy_loss.item()}, step=timestep)
        wandb.log({"loss.approx_kl": approx_kl.item()}, step=timestep)
        wandb.log({"loss.old_approx_kl": old_approx_kl.item()}, step=timestep)
        wandb.log({"loss.total_loss": loss.item()}, step=timestep)
        wandb.log({"loss.clipfrac": np.mean(clipfracs)}, step=timestep)
        wandb.log({"loss.explained_var": explained_var}, step=timestep)

        # Self-Imitation Learning
        rollout = memory.sample(batch_size=1)
        batch = Rollout(*zip(*rollout))
        b_obs = batch.states[0]
        b_actions = batch.actions[0]

        b_inds = np.arange(cfg.experiment.batch_size)

        if (i+1)%memory.si_counter == 0 and len(memory) > 0:
            avg_loss = 0
            for epoch in range(cfg.experiment.n_train_epochs):
                np.random.shuffle(b_inds)
                minibatch_size = cfg.experiment.batch_size // cfg.experiment.n_minibatches
                for start in range(0, cfg.experiment.batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    logits = agent.actor(Batch(b_obs[mb_inds]).obs)[0] 
                    dist_fn = agent.actor.dist_fn(logits)
                    loss = dist_fn.imitation_loss(b_actions[mb_inds])             
                    # Optimizer step
                    optimizer.zero_grad()
                    loss.backward()
                    # global gradient clipping
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.experiment.max_grad_norm)
                    optimizer.step()
                    avg_loss += loss.item()
            wandb.log({"loss.imitation_loss": avg_loss/cfg.experiment.n_train_epochs}, step=timestep)
        
    env.close()
    wandb.finish()
    
if __name__ == "__main__":
    main()