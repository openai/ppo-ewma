**Status:** Archive (code is provided as-is, no updates expected)

# PPO-EWMA

#### [ [Paper] ](https://arxiv.org/abs/2110.00641)

This is code for training agents using PPO-EWMA and PPG-EWMA, introduced in the paper [Batch size-invariance for policy optimization](https://arxiv.org/abs/2110.00641) ([citation](#citation)). It is based on the code for [Phasic Policy Gradient](https://github.com/openai/phasic-policy-gradient).

## Installation

Supported platforms: MacOS and Ubuntu, Python 3.7

Installation using [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

```
git clone https://github.com/openai/ppo-ewma.git
conda env update --name ppo-ewma --file ppo-ewma/environment.yml
conda activate ppo-ewma
pip install -e ppo-ewma
```

Alternatively, install the dependencies from [`environment.yml`](environment.yml) manually.

## Visualize results

Results are stored in blob storage at `https://openaipublic.blob.core.windows.net/rl-batch-size-invariance/`, and can be visualized as in the paper using [this Colab notebook](https://colab.research.google.com/github/openai/ppo-ewma/blob/master/notebooks/rl_batch_size_invariance.ipynb).

## Citation

Please cite using the following BibTeX entry:
```
@article{hilton2021batch,
  title={Batch size-invariance for policy optimization},
  author={Hilton, Jacob and Cobbe, Karl and Schulman, John},
  journal={arXiv preprint arXiv:2110.00641},
  year={2021}
}
```
