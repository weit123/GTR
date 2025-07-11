import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate


class PPO():
    def __init__(self,
                 actor_critic,
                 optimizer,
                 accelerator,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer
        self.accelerator = accelerator

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        ppo_loss_epoch = 0
        sft_loss_epoch = 0
        dist_entropy_epoch = 0
        ppo_grad_step = 0
        sft_grad_step = 0
        self.actor_critic.train()
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.mini_batch_size)
            for sample in data_generator:
                with self.accelerator.accumulate(self.actor_critic):
                    ppo_grad_step += 1
                    obs_batch, output_ids_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ, tc_obs_batch, input_ids_batch, labels_batch = sample

                    obs_batch = obs_batch.to(self.actor_critic.base.device)
                    output_ids_batch = output_ids_batch.to(self.actor_critic.base.device)
                    actions_batch = actions_batch.to(self.actor_critic.base.device)
                    value_preds_batch.to(self.actor_critic.base.device)
                    return_batch = return_batch.to(self.actor_critic.base.device)
                    masks_batch.to(self.actor_critic.base.device)
                    old_action_log_probs_batch = old_action_log_probs_batch.to(self.actor_critic.base.device)
                    adv_targ = adv_targ.to(self.actor_critic.base.device)
                    tc_obs_batch = tc_obs_batch.to(self.actor_critic.base.device)
                    input_ids_batch = input_ids_batch.to(self.actor_critic.base.device)
                    labels_batch = labels_batch.to(self.actor_critic.base.device)

                    # Thought cloning SFT loss
                    if not torch.all(labels_batch == -100):
                        try:
                            sft_loss, _ = self.actor_critic.sft_forward(
                                inputs = tc_obs_batch,
                                input_ids = input_ids_batch,
                                labels = labels_batch
                            )
                            self.accelerator.backward(sft_loss)
                            sft_loss_epoch += sft_loss.item()
                            sft_grad_step += 1
                        except:
                            pass

                    # PPO Loss
                    values, action_log_probs = self.actor_critic.evaluate_actions(
                        obs_batch, output_ids_batch)
                    if torch.isnan(action_log_probs).any():
                        continue
                    old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                    adv_targ = adv_targ.to(action_log_probs.device)
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)


                    ratio = torch.exp(action_log_probs -
                                    old_action_log_probs_batch)

                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    ## adding a ratio clip, inspired by https://github.com/huggingface/trl/blob/5a233546ee48532eaeb24b89b8d0042147574688/trl/trainer/ppo_trainer.py#L1199
                    if torch.any(ratio > 10):
                        action_loss = -surr2.mean()
                    else:
                        action_loss = -torch.min(surr1, surr2).mean()
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()

                    try:
                        assert not torch.isnan(value_loss), "value_loss is nan"
                        assert not torch.isnan(action_loss), "action_loss is nan"
                    except:
                        print("value/action loss is nan")
                        continue
                    ppo_loss = value_loss * self.value_loss_coef+action_loss
                    try:
                        self.accelerator.backward(ppo_loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.actor_critic.parameters(),
                                self.max_grad_norm
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        value_loss_epoch += value_loss.item()
                        action_loss_epoch += action_loss.item()
                        ppo_loss_epoch += ppo_loss.item()
                    except:
                        continue

        # avoid division by zero
        if sft_grad_step == 0:
            sft_grad_step = 1
        if ppo_grad_step == 0:
            ppo_grad_step = 1            
        
        value_loss_epoch /= ppo_grad_step
        action_loss_epoch /= ppo_grad_step
        dist_entropy_epoch /= ppo_grad_step
        ppo_loss_epoch /= ppo_grad_step
        sft_loss_epoch /= sft_grad_step
        
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, ppo_loss_epoch, sft_loss_epoch
