from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()
print("using xformers")

import copy
import glob
import os
import time
from collections import deque

import gymnasium as gym
import gym_cards
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

import math
import random
from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor
import transformers

from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

import datetime

import warnings
warnings.filterwarnings("ignore")

from thought_evaluate import thought_eval_and_correct
from gpt4o_interface import GPT4oAgent

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    ## environment interaction device is cpu
    model_device = device

    #initialization of llava
    model_path = args.model_path
    cache_dir = args.cache_dir

    print(model_path)
    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
        if args.q8 or args.q4:
            raise ValueError("Lora model does not support 8bit or 4bit quantization")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            print("8bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )
            print("4bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    use_grad_ckpt = True
    if use_grad_ckpt:
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    base.config.max_length = 1024
    print("Model max context length:{}".format(base.config.max_length))
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor

    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)
    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    if "gym_cards" in args.env_name.lower():
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, None, device, False, 1)
    else:
        print("Environment not supported")
        exit(1)

    corrector = GPT4oAgent()

    obs = envs.reset()
    infos = envs.reset_infos
    ## Inputing Prompt here
    qs = get_prompt(args.env_name, args.action_only_prompt, infos)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

    projection_f = partial(text_projection, env_name=args.env_name)

    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS,
                             args=args)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    agent = algo.PPO(
            actor_critic,
            optimizer,
            accelerator,
            args.clip_param,
            args.ppo_epoch,
            args.mini_batch_size,
            args.value_loss_coef,
            args.entropy_coef,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_env_steps, args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, args.max_new_tokens)

    _, output_ids, action, action_log_prob, action_tokens_log_prob, _, _ = actor_critic.act(obs, INPUT_IDS = INPUT_IDS)
    print("action:{}".format(action))
    print("action_log_prob:{}".format(action_log_prob))
    print("action_tokens_log_prob:{}".format(action_tokens_log_prob))

    rollouts.obs[0].copy_(obs)
    rollouts.obs_accu[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)
    legal_action_prob = []
    episode_lengths = []
    valid_prob = []
    corrected_prob = []
    output_tolen = []
    thought_tolen = []

    if args.env_name == "gym_cards/Points24-v0":
        target_formula = "NOT DETERMINED"
    else:
        target_formula = None

    start = time.time()
    if args.use_tensorboard and accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        log_path = os.path.join(args.logdir, args.logtag, str(args.seed), now)
        logger = SummaryWriter(log_path)

    print(qs)
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    num_explore = int(args.explore_portion*num_updates)
    prev_infos = []
    for j in tqdm(range(num_updates)):
        legal_action_prob.clear()
        valid_prob.clear()
        corrected_prob.clear()
        episode_lengths.clear()
        output_tolen.clear()
        thought_tolen.clear()
        episode_len = 0
        for step in range(args.num_steps):
            qs = get_prompt(args.env_name, args.action_only_prompt, infos)
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            print(f"-------- UPDATE {j}, STEP {step} --------")

            # Sample actions
            with torch.no_grad():
                INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
                value, output_id, action, action_log_prob, action_tokens_log_prob, legal_action, raw_output_id = actor_critic.act(rollouts.obs[step], INPUT_IDS = INPUT_IDS)
            legal_action_prob.append(legal_action)
            text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())))

            output_tolen.append(len(tokenizer.encode(text_action)))

            print(f"TEXT ACTION: {text_action}")
            print(f"INFOS: {infos}")
            if args.env_name == "gym_cards/Points24-v0":
                print(f"TARGET_FORMULA: {target_formula}")

            img = rollouts.obs[step][0].cpu().numpy().astype(np.uint8)
            correct_ids, labels, format_reward, cont, valid, corrected, thought_token_length, target_formula = thought_eval_and_correct(actor_critic, corrector, tokenizer, args.env_name, text_action, img, infos, target_formula, (args.env_name == "gym_cards/Points24-v0" or args.env_name == "gym_cards/EZPoints-v0"))
            valid_prob.append(valid)
            corrected_prob.append(corrected)
            if thought_token_length != None:
                thought_tolen.append(thought_token_length)

            # preparing input_ids and labels for thought cloning SFT
            input_ids = torch.cat([INPUT_IDS, correct_ids], dim=1).to(model_device)
            padded_input_ids = torch.zeros(input_ids.size(0), 2*args.max_new_tokens).to(dtype=input_ids.dtype, device = input_ids.device)
            padded_input_ids[:, :input_ids.size(1)] = input_ids
            
            padded_labels = torch.full((1, 2*args.max_new_tokens), -100)
            padded_labels[:, INPUT_IDS.size(1): input_ids.size(1)] = labels
            padded_labels = padded_labels.to(input_ids.device)
            
            prev_infos = copy.deepcopy(infos)
            obs, reward, done, infos = envs.step(action)
            episode_len += 1

            print(f"ACTION_REWARD: {reward.item()}")

            try:
                current_formula = ''.join(str(element) for element in infos[0]['Formula'])
            except:
                current_formula = ''

            # early truncation
            if not cont or reward.item() == -1 or  \
                   (args.env_name == "gym_cards/Points24-v0" and (target_formula is None or not target_formula.startswith(current_formula))):
                print("TRUNCATED!")
                done = [True]
                reward = torch.FloatTensor([-1.0])
                prev_infos = []
                obs = envs.reset()

            if done:
                infos = envs.reset_infos
                target_formula = None
                episode_lengths.append(episode_len)
                episode_len = 0

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            running_episode_rewards += reward.flatten()
            for i, d, r in zip(range(args.num_processes), done, reward):
                if d:
                    episode_rewards.append(running_episode_rewards[i].item())
                    if running_episode_rewards[i] > 0:
                        episode_success_rate.append(1)
                    else:
                        episode_success_rate.append(0)
                    episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                    running_episode_rewards[i] = 0
            # bad_mask is a legacy implementation of the storage.py file
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, output_id, action,
                            action_log_prob, value, reward, masks, bad_masks, format_reward, padded_input_ids, padded_labels)
        print("****** iteration number:{} ******".format(j))
        print("prompt:{}".format(prompt))
        print("text_action:{}".format(text_action))
        print("current observation:{}".format(prev_infos))
        print("ground truth:{}".format(infos))
        print("action log prob:{}".format(action_log_prob))
        print("action tokens log prob:{}".format(action_tokens_log_prob))
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy, ppo_loss, sft_loss = agent.update(rollouts)
        lr_scheduler.step()

        rollouts.after_update()
        if len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_success_rate),
                        dist_entropy, value_loss, action_loss))

            if args.use_tensorboard and accelerator.is_main_process:
                if len(thought_tolen) == 0:
                    thought_tolen.append(0)
                log_data = {"iteration": j,
                        "num_timesteps": total_num_steps,
                        "FPS": int(total_num_steps / (end - start)),
                        "episode/episode_reward.mean": np.mean(episode_rewards),
                        "episode/episode_reward.median": np.median(episode_rewards),
                        "episode/episode_reward.min": np.min(episode_rewards),
                        "episode/episode_reward.max": np.max(episode_rewards),
                        "episode/episode_success_rate.mean": np.mean(episode_success_rate),
                        "episode/episode_action_tokens_log_prob.mean": np.mean(episode_action_tokens_log_prob),
                        "episode/episode_length": np.mean(episode_lengths),
                        "metric/legal_action_prob": np.mean(legal_action_prob),
                        "metric/valid_prob": np.mean(valid_prob),
                        "metric/corrected_prob": np.mean(corrected_prob),
                        "metric/output_token_length": np.mean(output_tolen),
                        "metric/thought_token_length": np.mean(thought_tolen),        
                        "distribution_entropy": dist_entropy,
                        "loss/value.loss": value_loss,
                        "loss/action.loss": action_loss,
                        "loss/ppo.loss": ppo_loss,
                        "loss/sft.loss": sft_loss,
                        "reward/format_reward.mean": rollouts.format_rewards.mean().item(),
                        "reward/reward.max": rollouts.rewards.max().item(),
                        "reward/reward.min": rollouts.rewards.min().item(),
                        "reward/reward.mean": rollouts.rewards.mean().item(),
                        "reward/reward.std": rollouts.rewards.std().item(),
                        "reward/reward.median": rollouts.rewards.median().item(),
                        "reward/return.max": rollouts.returns.max().item(),
                        "reward/return.min": rollouts.returns.min().item(),
                        "reward/return.mean": rollouts.returns.mean().item(),
                        "reward/return.std": rollouts.returns.std().item(),
                        "value/value.max": rollouts.value_preds.max().item(),
                        "value/value.min": rollouts.value_preds.min().item(),
                        "value/value.mean": rollouts.value_preds.mean().item(),
                        "value/value.std": rollouts.value_preds.std().item(),}
                for k, v in log_data.items():
                    logger.add_scalar(k, v, global_step=total_num_steps)
    
    # save final model
    if accelerator.is_main_process: 
        print("Saving final model...")
        checkpoint_path = os.path.join(args.output_dir, "final")     
        os.makedirs(checkpoint_path, exist_ok=True)
        actor_critic.value_model.base.save_pretrained(checkpoint_path)

if __name__ == "__main__":
    main()

