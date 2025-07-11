import torch
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class DualSampler(Sampler):
    def __init__(self, ppo_bz, tc_bz, batch_size):
        self.ppo_sampler = SubsetRandomSampler(range(ppo_bz))
        self.tc_sampler = SubsetRandomSampler(range(tc_bz))
        self.batch_size = batch_size
    
    def __iter__(self):
        iter1 = iter(self.ppo_sampler)
        iter2 = iter(self.tc_sampler)
        
        for i in range(len(self.ppo_sampler) // self.batch_size):
            ppo_batch = []
            tc_batch = []
            for _ in range(self.batch_size):
                try:
                    ppo_batch.append(next(iter1))
                except StopIteration:
                    iter1 = iter(self.ppo_sampler)
                    ppo_batch.append(next(iter1))

                try:
                    tc_batch.append(next(iter2))
                except StopIteration:
                    iter2 = iter(self.tc_sampler)
                    tc_batch.append(next(iter2))

            yield ppo_batch, tc_batch

    def __len__(self):
        return len(self.ppo_sampler) // self.batch_size

class RolloutStorage(object):
    def __init__(self, num_env_steps, num_steps, num_processes, obs_shape, action_space, max_new_tokens):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        #hard-code to cases of max_new_tokens being smaller than 32
        self.output_ids = torch.zeros(
            num_steps, num_processes, 2*max_new_tokens).long()
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.format_rewards = torch.zeros(num_steps, num_processes, 1)

        # For thought cloning (SFT)
        # hard-code max input and label length as 1500
        self.obs_accu = torch.zeros(num_env_steps + 1, num_processes, *obs_shape)
        self.input_ids = torch.zeros(
            num_env_steps, num_processes, 1500).long()
        self.labels = torch.zeros(
            num_env_steps, num_processes, 1500).long()

        self.num_steps = num_steps
        self.num_env_steps = num_env_steps
        self.step_accu = 0
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.output_ids = self.output_ids.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.format_rewards = self.format_rewards.to(device)
        self.obs_accu = self.obs_accu.to(device)
        self.input_ids = self.input_ids.to(device)
        self.labels = self.labels.to(device)

    def insert(self, obs, output_ids, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, format_rewards, input_ids, labels):
        self.obs[self.step + 1].copy_(obs)
        self.output_ids[self.step].copy_(output_ids)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.format_rewards[self.step].copy_(format_rewards)
        self.obs_accu[self.step_accu + 1].copy_(obs)
        self.input_ids[self.step_accu].copy_(input_ids)
        self.labels[self.step_accu].copy_(labels)

        self.step = (self.step + 1) % self.num_steps
        self.step_accu = (self.step_accu + 1) % self.num_env_steps


    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
                    self.returns[step] = self.returns[step] + self.format_rewards[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
                    self.returns[step] = self.returns[step] + self.format_rewards[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
                    self.returns[step] = self.returns[step] + self.format_rewards[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]
                    self.returns[step] = self.returns[step] + self.format_rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        ppo_batch_size = num_processes * num_steps
        tc_batch_size = num_processes * self.step_accu

        sampler = DualSampler(ppo_batch_size, tc_batch_size, mini_batch_size)
        for sample in sampler:
            ppo_indices, tc_indices = sample
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[ppo_indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[ppo_indices]
            output_ids_batch = self.output_ids.view(-1,
                                              self.output_ids.size(-1))[ppo_indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[ppo_indices]
            return_batch = self.returns[:-1].view(-1, 1)[ppo_indices]
            masks_batch = self.masks[:-1].view(-1, 1)[ppo_indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[ppo_indices]
            
            tc_obs_batch = self.obs_accu[:-1].view(-1, *self.obs_accu.size()[2:])[tc_indices]
            input_ids_batch = self.input_ids.view(-1,
                                              self.input_ids.size(-1))[tc_indices]
            labels_batch = self.labels.view(-1, self.labels.size(-1))[tc_indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[ppo_indices]

            yield obs_batch, output_ids_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, tc_obs_batch, input_ids_batch, labels_batch
