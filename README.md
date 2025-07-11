# [ICCV 2025] GTR: <u>G</u>uided <u>T</u>hought <u>R</u>einforcement Prevents Thought Collapse in RL-based VLM Agent Training

<a href="https://arxiv.org/abs/2503.08525"><img src="https://img.shields.io/badge/ArXiv-2503.08525-brightgreen"></a>

## Abstract
Reinforcement learning with verifiable outcome rewards (RLVR) has effectively scaled up chain-of-thought (CoT) reasoning in large language models (LLMs). Yet, its efficacy in training vision-language model (VLM) agents for goal-directed action reasoning in visual environments is less established. This work investigates this problem through extensive experiments on complex card games, such as 24 points, and embodied tasks from ALFWorld. We find that when rewards are based solely on action outcomes, RL fails to incentivize CoT reasoning in VLMs, instead leading to a phenomenon we termed thought collapse, characterized by a rapid loss of diversity in the agent's thoughts, state-irrelevant and incomplete reasoning, and subsequent invalid actions, resulting in negative rewards. To counteract thought collapse, we highlight the necessity of process guidance and propose an automated corrector that evaluates and refines the agent's reasoning at each RL step. This simple and scalable GTR (Guided Thought Reinforcement) framework trains reasoning and action simultaneously without the need for dense, per-step human labeling. Our experiments demonstrate that GTR significantly enhances the performance and generalization of the LLaVA-7B model across various visual environments, achieving 3-5 times higher task success rates compared to SoTA models with notably smaller model sizes.

## Table of Contents
- [Code Structure](#code_structure)
- [Getting Started](#getting_started)
- [Citation](#citation)



<a name="code_structure"></a>

## Code Structure

1. A slightly modified version of [LLaVA](https://github.com/haotian-liu/LLaVA), in accordance with [RL4VLM](https://github.com/RL4VLM/RL4VLM).

2. [GymCards](./gym-cards/README.md) environment for Points24 task, in accordance with [RL4VLM](https://github.com/RL4VLM/RL4VLM).

3. `GTR_gymcards`, code for training agent on GymCards tasks, including Points24.
4. `GTR_alf`, code for training agent on ALFWorld tasks.



<a name="getting_started"></a>

## Getting Started

The SFT-initialized models for each task can be found in [here](https://huggingface.co/datasets/LEVI-Project/sft-model-data/tree/main).

### Points24

1. Setup the environment

```bash
cd <path-to-this-repo>
pip install -e ../LLaVA
pip install -e ../gym-cards
pip install gymnasium[atari,accept-rom-license]
pip install stable-baselines3 nltk deepspeed sentencepiece git+https://github.com/openai/CLIP.git
pip install xformers
```

2. Setting your OpenAI API key in `gpt4o_interface.py`.
3. Run the script

```bash
cd scripts
bash run_p24.sh
```

For multi-GPU training, you may change the `num_processes` in `config_zero2.yaml`.

### ALFWorld

1. Setup the environment

```bash
cd <path-to-this-repo>
conda env create -f alf_conda.yml
conda activate vrenv-alf
pip install -e ../LLaVA
pip install -e ../gym-cards
pip install git+https://github.com/openai/CLIP.git
pip install numpy==1.23.5
pip install protobuf==3.20.3
pip install pydantic==1.10.14
pip install pydantic-core==2.16.3
pip install nltk
pip uninstall frozenlist gradio murmurhash preshed spacy srsly thinc weasel aiosignal annotated-types blis catalogue cloudpathlib cymem
export ALFWORLD_DATA=<storage_path>
alfworld-download
```

​	You may test the installation by running:

```bash
alfworld-play-thor
```

2. Setting your OpenAI API key in `gpt4o_interface.py`.

3. Run the script

```bash
cd scripts
bash run_alf.sh
```

​	We strongly recommend only using 1 GPU to prevent NCCL time out during the synchronization.



<a name="citation"></a>

## Citation

If you find our work useful, please kindly cite:

```
@article{wei2025gtr,
  title={GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training},
  author={Wei, Tong and Yang, Yijun and Xing, Junliang and Shi, Yuanchun and Lu, Zongqing and Ye, Deheng},
  journal={arXiv preprint arXiv:2503.08525},
  year={2025}
}
```



## Acknowledgement

<a id="acknowledgement"></a>
Our code adopts the basic environment setting and RL framework from [RL4VLM](https://github.com/RL4VLM/RL4VLM), which uses [LLaVA](https://github.com/haotian-liu/LLaVA) as a backbone and [PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) as RL algorithm implementation.