# GTR: <u>G</u>uided <u>T</u>hought <u>R</u>einforcement Prevents Thought Collapse in RL-based VLM Agent Training

<a href="https://arxiv.org/abs/2503.08525"><img src="https://img.shields.io/badge/ArXiv-2503.08525-brightgreen"></a>



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