import gym
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import numpy as np

import minedojo
from wrappers import MineClipWrapper

from mineclip.mineagent import features as F
from mineclip import SimpleFeatureFusion, MineAgent, MultiCategoricalActor
from mineclip.mineagent.batch import Batch

import hydra
import wandb

DEBUG = True

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
        x = self.feature_net(obs)
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
        return self.critic(obs)
        
    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        import ipdb; ipdb.set_trace()
        probs = self.actor.dist_fn(logits=logits)
        if action == None:
            action = probs.sample()
            # action = probs.mode() if deterministic
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)
    
    
def preprocess_obs(obs, info, prev_action, device):
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
    
    obs = {
        "compass": torch.tensor(compass.reshape(B, 4), device=device),
        "gps": torch.tensor(gps.reshape(B, 3), device=device),
        "voxels": torch.tensor(
            voxels.reshape(B, 3 * 3 * 3), dtype=torch.long, device=device
        ), # FIXME: cannot reshape array of size 8 into shape (1,)
        "biome_id": torch.tensor(
            biome_id.reshape(B, ), dtype=torch.long, device=device
        ), 
        # FIXME: implement separate feature network for prev_action 
        "prev_action": torch.randint(
            low=0, high=88, size=(B,), dtype=torch.long, device=device
        ),
        "prev_action": torch.tensor(prev_action.reshape(B, ), device=device), 
        "prompt": torch.tensor(prompt.reshape(B, 512), device=device), 
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
        task_id="combat_spider_plains_leather_armors_diamond_sword_shield",
        image_size=(160, 256),
        world_seed=123,
        seed=42,
    )
    prompts = [
        "find spider",
        "kill spider",
    ]
    env = MineClipWrapper(env,prompts)
    
    # Set up the agent
    agent = ActorCritic(cfg, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.experiment.learning_rate)
    
    # Set up storage
    # TODO: see if we can use tensor here with Batch
    obs = np.empty((cfg.experiment.rollout_length, cfg.experiment.n_envs, ), dtype=object)
    # FIXME: check dim of transformed action
    actions = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs, 89)).to(device)
    logprobs = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs)).to(device)
    rewards = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs)).to(device)
    dones = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs)).to(device)
    values = torch.zeros((cfg.experiment.rollout_length, cfg.experiment.n_envs)).to(device)
    
    # Training
    timestep = 0
    env.reset()
    action = env.action_space.no_op()
    next_obs, reward, next_done, info = env.step(env.action_space.no_op())
    next_obs = preprocess_obs(next_obs, info, action, device)
    if DEBUG: 
        import ipdb; ipdb.set_trace()
    num_updates = cfg.experiment.total_timesteps // cfg.experiment.batch_size
    
    for i in range(num_updates):
        if cfg.experiment.anneal_lr:
            frac = 1.0 - (i - 1.0) / num_updates
            lrnow = frac * cfg.experiment.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        for step in range(0, cfg.experiment.rollout_length):
            timestep += 1 * cfg.experiment.n_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            # Collect rollout
            with torch.no_grad(): 
                if DEBUG:
                    import ipdb; ipdb.set_trace()
                action, logprob, ent, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, done, info = env.step(transform_action(action))
            next_obs = preprocess_obs(next_obs, info, action, device)
            rewards[step] = torch.tensor(reward, device=device)
            next_obs, next_done = next_obs, torch.Tensor(done).to(device)
            
            if timestep % cfg.experiment.log_interval == 0:
                wandb.log({"reward": reward})
                
        # Compute returns
        with torch.no_grad():
            next_value = agent.get_value(next_value)
            advantages = torch.zero_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.experiment.num_steps)):
                if t == cfg.experiment.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.experiment.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.experiment.gamma * cfg.experiment.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        b_obs = obs
        if DEBUG: 
            import ipdb; ipdb.set_trace()
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        b_inds = np.arange(cfg.experiment.batch_size)
        clipfracs = []
        for epoch in range(cfg.experiment.n_train_epochs):
            np.random.shuffle(b_inds)
            minibatch_size = cfg.experiment.batch_size // cfg.experiment.n_minibatches
            for start in range(0, cfg.experiment.batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                # FIXME: b_obs is numpy array...
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions)
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
                loss = pg_loss - entropy_loss * cfg.experiment.entropy_coef + v_loss * cfg.experiment.value_loss_coef
                
                # Optimizer step
                optimizer.zero_grad()
                loss.backward()
                # global gradient clipping
                nn.utils.clip_grad.norm_(agent.parameters(), cfg.experiment.max_grad_norm)
                optimizer.step()
            
            if cfg.experiment.target_kl is not None:
                if approx_kl > cfg.experiment.target_kl:
                    break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Log to wandb        
        wandb.log("learnging_rate", optimizer.param_groups[0]['lr'], step=timestep)
        wandb.log("loss.policy_loss", pg_loss.item(), step=timestep)
        wandb.log("loss.value_loss", v_loss.item(), step=timestep)
        wandb.log("loss.entropy", entropy_loss.item(), step=timestep)
        wandb.log("loss.approx_kl", approx_kl.item(), step=timestep)
        wandb.log("loss.old_approx_kl", old_approx_kl.item(), step=timestep)
        wandb.log("loss.total_loss", loss.item(), step=timestep)
        wandb.log("loss.clipfrac", np.mean(clipfracs), step=timestep)
        wandb.log("loss.explained_var", explained_var, step=timestep)
        
    env.close()
    wandb.finish()
    
if __name__ == "__main__":
    main()