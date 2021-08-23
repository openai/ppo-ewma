"""
Mostly copied from ppo.py but with some extra options added that are relevant to phasic
"""

import numpy as np
import torch as th
from queue import Queue
from mpi4py import MPI
from functools import partial
from .tree_util import tree_map, tree_multimap
from . import torch_util as tu
from .log_save_helper import LogSaveHelper
from .minibatch_optimize import minibatch_optimize
from .roller import Roller
from .reward_normalizer import RewardNormalizer

import math
from . import logger

INPUT_KEYS = {"ob", "ac", "first", "logp", "rec_logp", "vtarg", "adv", "state_in"}

def compute_gae(
    *,
    vpred: "(th.Tensor[1, float]) value predictions",
    reward: "(th.Tensor[1, float]) rewards",
    first: "(th.Tensor[1, bool]) mark beginning of episodes",
    γ: "(float)",
    λ: "(float)"
):
    orig_device = vpred.device
    assert orig_device == reward.device == first.device
    vpred, reward, first = (x.cpu() for x in (vpred, reward, first))
    first = first.to(dtype=th.float32)
    assert first.dim() == 2
    nenv, nstep = reward.shape
    assert vpred.shape == first.shape == (nenv, nstep + 1)
    adv = th.zeros(nenv, nstep, dtype=th.float32)
    lastgaelam = 0
    for t in reversed(range(nstep)):
        notlast = 1.0 - first[:, t + 1]
        nextvalue = vpred[:, t + 1]
        # notlast: whether next timestep is from the same episode
        delta = reward[:, t] + notlast * γ * nextvalue - vpred[:, t]
        adv[:, t] = lastgaelam = delta + notlast * γ * λ * lastgaelam
    vtarg = vpred[:, :-1] + adv
    return adv.to(device=orig_device), vtarg.to(device=orig_device)

def log_vf_stats(comm, **kwargs):
    logger.logkv(
        "VFStats/EV", tu.explained_variance(kwargs["vpred"], kwargs["vtarg"], comm)
    )
    for key in ["vpred", "vtarg", "adv"]:
        logger.logkv_mean(f"VFStats/{key.capitalize()}Mean", kwargs[key].mean())
        logger.logkv_mean(f"VFStats/{key.capitalize()}Std", kwargs[key].std())

def compute_advantage(model, seg, γ, λ, comm=None, adv_moments=None):
    comm = comm or MPI.COMM_WORLD
    finalob, finalfirst = seg["finalob"], seg["finalfirst"]
    vpredfinal = model.v(finalob, finalfirst, seg["finalstate"])
    reward = seg["reward"]
    logger.logkv("Misc/FrameRewMean", reward.mean())
    adv, vtarg = compute_gae(
        γ=γ,
        λ=λ,
        reward=reward,
        vpred=th.cat([seg["vpred"], vpredfinal[:, None]], dim=1),
        first=th.cat([seg["first"], finalfirst[:, None]], dim=1),
    )
    log_vf_stats(comm, adv=adv, vtarg=vtarg, vpred=seg["vpred"])
    seg["vtarg"] = vtarg
    adv_mean, adv_var = tu.mpi_moments(comm, adv)
    if adv_moments is not None:
        adv_moments.update(adv_mean, adv_var, adv.numel() * comm.size)
        adv_mean, adv_var = adv_moments.moments()
        logger.logkv_mean("VFStats/AdvEwmaMean", adv_mean)
        logger.logkv_mean("VFStats/AdvEwmaStd", math.sqrt(adv_var))
    seg["adv"] = (adv - adv_mean) / (math.sqrt(adv_var) + 1e-8)

def tree_cat(trees):
    return tree_multimap(lambda *xs: th.cat(xs, dim=0), *trees)

def recompute_logp(*, model, seg, mbsize):
    b = tu.batch_len(seg)
    with th.no_grad():
        logps = []
        for inds in th.arange(b).split(mbsize):
            mb = tu.tree_slice(seg, inds)
            pd, _, _, _ = model(mb["ob"], mb["first"], mb["state_in"])
            logp = tu.sum_nonbatch(pd.log_prob(mb["ac"]))
            logps.append(logp)
        seg["rec_logp"] = tree_cat(logps)

def compute_losses(
    model,
    model_ewma,
    ob,
    ac,
    first,
    logp,
    rec_logp,
    vtarg,
    adv,
    state_in,
    clip_param,
    vfcoef,
    entcoef,
    kl_penalty,
    imp_samp_max,
):
    losses = {}
    diags = {}
    pd, vpred, aux, _state_out = model(ob=ob, first=first, state_in=state_in)
    newlogp = tu.sum_nonbatch(pd.log_prob(ac))
    if model_ewma is not None:
        pd_ewma, _vpred_ewma, _, _state_out_ewma = model_ewma(
            ob=ob, first=first, state_in=state_in
        )
        rec_logp = tu.sum_nonbatch(pd_ewma.log_prob(ac))
    # prob ratio for KL / clipping based on a (possibly) recomputed logp
    logratio = newlogp - rec_logp
    # stale data can give rise to very large importance sampling ratios,
    # especially when using the wrong behavior policy,
    # so we need to clip them for numerical stability.
    # this can introduce bias, but by default we only clip extreme ratios
    # to minimize this effect
    logp_adj = logp
    if imp_samp_max > 0:
        logp_adj = th.max(logp, newlogp.detach() - math.log(imp_samp_max))

    # because of the large importance sampling ratios again,
    # we need to handle the ratios in log space for numerical stability
    pg_losses = -adv * th.exp(newlogp - logp_adj)
    if clip_param > 0:
        clipped_logratio = th.clamp(logratio, math.log(1.0 - clip_param), math.log(1.0 + clip_param))
        pg_losses2 = -adv * th.exp(clipped_logratio + rec_logp - logp_adj)
        pg_losses = th.max(pg_losses, pg_losses2)

    diags["entropy"] = entropy = tu.sum_nonbatch(pd.entropy()).mean()
    diags["negent"] = -entropy * entcoef
    diags["pg"] = pg_losses.mean()
    diags["pi_kl"] = kl_penalty * 0.5 * (logratio ** 2).mean()

    losses["pi"] = diags["negent"] + diags["pg"] + diags["pi_kl"]
    losses["vf"] = vfcoef * ((vpred - vtarg) ** 2).mean()

    with th.no_grad():
        if clip_param > 0:
            diags["clipfrac"] = th.logical_or(
                logratio < math.log(1.0 - clip_param),
                logratio > math.log(1.0 + clip_param),
            ).float().mean()
        diags["approxkl"] = 0.5 * (logratio ** 2).mean()
        if imp_samp_max > 0:
            diags["imp_samp_clipfrac"] = (newlogp - logp > math.log(imp_samp_max)).float().mean()

    return losses, diags

class EwmaMoments:
    """
    Calculate rolling moments using EWMAs.
    """

    def __init__(self, ewma_decay):
        self.ewma_decay = ewma_decay
        self.w = 0.0
        self.ww = 0.0 # sum of squared weights
        self.wsum = 0.0
        self.wsumsq = 0.0

    def update(self, mean, var, count, *, ddof=0):
        self.w *= self.ewma_decay
        self.ww *= self.ewma_decay ** 2
        self.wsum *= self.ewma_decay
        self.wsumsq *= self.ewma_decay
        self.w += count
        self.ww += count
        self.wsum += mean * count
        self.wsumsq += (count - ddof) * var + count * mean ** 2

    def moments(self, *, ddof=0):
        mean = self.wsum / self.w
        # unbiased weighted sample variance:
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
        var = (self.wsumsq - self.wsum ** 2 / self.w) / (self.w - ddof * self.ww / self.w)
        return mean, var

def learn(
    *,
    venv: "(VecEnv) vectorized environment",
    model: "(ppo.PpoModel)",
    model_ewma: "(ppg.EwmaModel) alternate model used for clipping or the KL penalty",
    interacts_total: "(float) total timesteps of interaction" = float("inf"),
    nstep: "(int) number of serial timesteps" = 256,
    γ: "(float) discount" = 0.99,
    λ: "(float) GAE parameter" = 0.95,
    clip_param: "(float) PPO parameter for clipping prob ratio" = 0.2,
    vfcoef: "(float) value function coefficient" = 0.5,
    entcoef: "(float) entropy coefficient" = 0.01,
    nminibatch: "(int) number of minibatches to break epoch of data into" = 4,
    n_epoch_vf: "(int) number of epochs to use when training the value function" = 1,
    n_epoch_pi: "(int) number of epochs to use when training the policy" = 1,
    lr: "(float) Adam learning rate" = 5e-4,
    beta1: "(float) Adam beta1" = 0.9,
    beta2: "(float) Adam beta2" = 0.999,
    default_loss_weights: "(dict) default_loss_weights" = {},
    store_segs: "(bool) whether or not to store segments in a buffer" = True,
    verbose: "(bool) print per-epoch loss stats" = True,
    log_save_opts: "(dict) passed into LogSaveHelper" = {},
    rnorm: "(bool) reward normalization" = True,
    kl_penalty: "(int) weight of the KL penalty, which can be used in place of clipping" = 0,
    adv_ewma_decay: "(float) EWMA decay for advantage normalization" = 0.0,
    grad_weight: "(float) relative weight of this worker's gradients" = 1,
    comm: "(MPI.Comm) MPI communicator" = None,
    callbacks: "(seq of function(dict)->bool) to run each update" = (),
    learn_state: "dict with optional keys {'opts', 'roller', 'lsh', 'reward_normalizer', 'curr_interact_count', 'seg_buf', 'segs_delayed', 'adv_moments'}" = None,
    staleness: "(int) number of iterations by which to make data artificially stale, for experimentation" = 0,
    staleness_loss: "(str) one of 'decoupled', 'behavior' or 'proximal', only used if staleness > 0" = "decoupled",
    imp_samp_max: "(float) value at which to clip importance sampling ratio" = 100.0,
):
    if comm is None:
        comm = MPI.COMM_WORLD

    learn_state = learn_state or {}
    ic_per_step = venv.num * comm.size * nstep

    opt_keys = (
        ["pi", "vf"] if (n_epoch_pi != n_epoch_vf) else ["pi"]
    )  # use separate optimizers when n_epoch_pi != n_epoch_vf
    params = list(model.parameters())
    opts = learn_state.get("opts") or {
        k: th.optim.Adam(params, lr=lr, betas=(beta1, beta2))
        for k in opt_keys
    }

    tu.sync_params(params)

    if rnorm:
        reward_normalizer = learn_state.get("reward_normalizer") or RewardNormalizer(venv.num)
    else:
        reward_normalizer = None

    def get_weight(k):
        return default_loss_weights[k] if k in default_loss_weights else 1.0

    def train_with_losses_and_opt(loss_keys, opt, **arrays):
        losses, diags = compute_losses(
            model,
            model_ewma=model_ewma,
            entcoef=entcoef,
            kl_penalty=kl_penalty,
            clip_param=clip_param,
            vfcoef=vfcoef,
            imp_samp_max=imp_samp_max,
            **arrays,
        )
        loss = sum([losses[k] * get_weight(k) for k in loss_keys])
        opt.zero_grad()
        loss.backward()
        tu.warn_no_gradient(model, "PPO")
        tu.sync_grads(params, grad_weight=grad_weight)
        diags = {k: v.detach() for (k, v) in diags.items()}
        opt.step()
        if "pi" in loss_keys and model_ewma is not None:
            model_ewma.update()
        diags.update({f"loss_{k}": v.detach() for (k, v) in losses.items()})
        return diags

    def train_pi(**arrays):
        return train_with_losses_and_opt(["pi"], opts["pi"], **arrays)

    def train_vf(**arrays):
        return train_with_losses_and_opt(["vf"], opts["vf"], **arrays)

    def train_pi_and_vf(**arrays):
        return train_with_losses_and_opt(["pi", "vf"], opts["pi"], **arrays)

    roller = learn_state.get("roller") or Roller(
        act_fn=model.act,
        venv=venv,
        initial_state=model.initial_state(venv.num),
        keep_buf=100,
        keep_non_rolling=log_save_opts.get("log_new_eps", False),
    )

    lsh = learn_state.get("lsh") or LogSaveHelper(
        ic_per_step=ic_per_step, model=model, comm=comm, **log_save_opts
    )

    callback_exit = False  # Does callback say to exit loop?

    curr_interact_count = learn_state.get("curr_interact_count") or 0
    curr_iteration = 0
    seg_buf = learn_state.get("seg_buf") or []
    segs_delayed = learn_state.get("segs_delayed") or Queue(maxsize=staleness + 1)

    adv_moments = learn_state.get("adv_moments") or EwmaMoments(adv_ewma_decay)

    while curr_interact_count < interacts_total and not callback_exit:
        seg = roller.multi_step(nstep)
        lsh.gather_roller_stats(roller)
        if staleness > 0:
            segs_delayed.put(seg)
            if not segs_delayed.full():
                continue
            seg = segs_delayed.get()
            if staleness_loss == "behavior":
                seg["rec_logp"] = seg["logp"]
            else:
                recompute_logp(model=model, seg=seg, mbsize=4)
                if staleness_loss == "proximal":
                    seg["logp"] = seg["rec_logp"]
        else:
            seg["rec_logp"] = seg["logp"]
        if rnorm:
            seg["reward"] = reward_normalizer(seg["reward"], seg["first"])
        compute_advantage(model, seg, γ, λ, comm=comm, adv_moments=adv_moments)

        if store_segs:
            seg_buf.append(tree_map(lambda x: x.cpu(), seg))

        with logger.profile_kv("optimization"):
            # when n_epoch_pi != n_epoch_vf, we perform separate policy and vf epochs with separate optimizers
            if n_epoch_pi != n_epoch_vf:
                minibatch_optimize(
                    train_vf,
                    {k: seg[k] for k in INPUT_KEYS},
                    nminibatch=nminibatch,
                    comm=comm,
                    nepoch=n_epoch_vf,
                    verbose=verbose,
                )

                train_fn = train_pi
            else:
                train_fn = train_pi_and_vf

            epoch_stats = minibatch_optimize(
                train_fn,
                {k: seg[k] for k in INPUT_KEYS},
                nminibatch=nminibatch,
                comm=comm,
                nepoch=n_epoch_pi,
                verbose=verbose,
            )
            for (k, v) in epoch_stats[-1].items():
                logger.logkv("Opt/" + k, v)

        lsh()

        curr_interact_count += ic_per_step
        curr_iteration += 1

        for callback in callbacks:
            callback_exit = callback_exit or bool(callback(locals()))

    return dict(
        opts=opts,
        roller=roller,
        lsh=lsh,
        reward_normalizer=reward_normalizer,
        curr_interact_count=curr_interact_count,
        seg_buf=seg_buf,
        segs_delayed=segs_delayed,
        adv_moments=adv_moments,
    )
