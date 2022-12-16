import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import KLDivLoss


class ActionSmoothingLoss(nn.Module):
    def __init__(self, act_space, reduction='batchmean', log_target=True):
        super(ActionSmoothingLoss, self).__init__()
        self.kl_loss = KLDivLoss(reduction=reduction,
                                 log_target=log_target)
        self.action_space = act_space
        self.sub_actions = self.action_space.nvec

    def forward(self, current_action, previous_actions, logits=True):
        """
        Args:
            current_action: probabilities for current action [A (e.g., 68)]
            previous_actions: probabilities for previous actions [W, A]
            W: window size
            A: sum(self.action_space.nvec)
        """
        current_action = torch.reshape(current_action, (sum(self.sub_actions),))
        previous_actions = torch.reshape(previous_actions, (-1, sum(self.sub_actions)))
        W, A = previous_actions.shape
        if logits:
            for j in range(len(self.sub_actions)):
                start = sum(self.sub_actions[:j])
                stop = sum(self.sub_actions[:j+1])
                previous_actions[:, start:stop] = F.log_softmax(previous_actions[:, start:stop], dim=1)
                current_action[start:stop] = F.log_softmax(current_action[start:stop], dim=0)

        loss = 0
        for i in range(W):
            for j in range(len(self.sub_actions)):
                if j == 0:
                    start = 0
                else:
                    start = sum(self.sub_actions[:j])
                end = sum(self.sub_actions[:j+1])
                current_sub_action = current_action[start:end]
                previous_sub_action = previous_actions[i, start:end]
                loss += self.kl_loss(current_sub_action, previous_sub_action)

        loss /= W
        return loss


if __name__ == "__main__":
    import gym
    import numpy as np
    action_space = gym.spaces.MultiDiscrete([3, 3, 4, 25, 25, 8])

    # ActionSmoothingLoss
    loss = ActionSmoothingLoss(action_space)

    prev_action_logits = torch.randn(1000, 1, 68)
    b_prev_action_logits = prev_action_logits.reshape(-1, sum(action_space.nvec))
    b_inds = np.arange(128)
    np.random.shuffle(b_inds)
    minibatch_size = 128 // 1
    for start in range(0, 128, minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]

        for mb_ind, action_logit in zip(mb_inds, b_prev_action_logits[mb_inds]):
            start_window = max(0, mb_ind - 10)
            # take the preceding n actions, take all the logits for these actions
            if mb_ind == 0:
                preceding_action_logits = b_prev_action_logits[0, :]
            else:
                preceding_action_logits = b_prev_action_logits[start_window:mb_ind, :]
            a_loss = loss(action_logit, preceding_action_logits)

    # loss
    print(a_loss)
