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
        W, A = previous_actions.shape
        if logits:
            for j in range(len(self.sub_actions)):
                previous_actions[:, sum(self.sub_actions[:j]):
                                    sum(self.sub_actions[:j+1])] = F.log_softmax(previous_actions[:, sum(self.sub_actions[:j]):
                                                                                                     sum(self.sub_actions[:j+1])], dim=1)
                current_action[sum(self.sub_actions[:j]):
                               sum(self.sub_actions[:j+1])] = F.log_softmax(current_action[sum(self.sub_actions[:j]):
                                                                                           sum(self.sub_actions[:j+1])], dim=0)

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
    action_space = gym.spaces.MultiDiscrete([3, 3, 4, 25, 25, 8])

    # ActionSmoothingLoss
    loss = ActionSmoothingLoss(action_space)

    # current_action
    current_action = []
    for num_actions in action_space.nvec:
        current_action.append(torch.rand(1, num_actions))
    current_action = torch.cat(current_action, dim=1).squeeze()

    # previous_actions
    previous_actions = []
    for num_actions in action_space.nvec:
        previous_actions.append(torch.rand(5, num_actions))
    previous_actions = torch.cat(previous_actions, dim=1)
    # loss
    loss_value = loss(current_action, previous_actions)
    print(loss_value)