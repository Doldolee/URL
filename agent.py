"""Offline RL agents, train-only behavior fitting and fixed policy interfaces."""
from __future__ import annotations
import copy
import hashlib
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from metric import _metrics, policy_probs, probability_metrics, validate_transitions
from model import (
    BCQNet,
    CQLNet,
    DQNNet,
    FQECritic,
    FullActionPredictor,
    TemperatureScaledPredictor,
    _classes,
    full_action_proba,
)
from util import (
    NUM_ACTIONS,
    _trajectory_split,
    episode_groups,
    episode_slices,
    grouped_partition,
    validate_data,
)


# Q-learning agents

class BaseQ(ABC):
    def __init__(self,
                 num_actions,
                 state_dim,
                 device,
                 discount=0.99,
                 optimizer='Adam',
                 optimizer_parameters={},
                 use_polyak_target_update=False,
                 target_update_frequency=1,
                 tau=0.005,
                 hidden_node=0,
                 activation="-",
                 target_data = "-",
                 **kwargs):
        self.device = device
        self.num_actions = num_actions
        self.discount = discount
        self.tau = tau
        self.target_update_frequency = target_update_frequency
        self.iterations = 0
        self.target_data = target_data
        self.state_dim = state_dim
        self.note_emb_dim = kwargs.get("note_emb_dim")

        
        # Select the target-update method.
        self.maybe_update_target = self.polyak_target_update if use_polyak_target_update else self.copy_target_update

        
    @abstractmethod
    def build_network(self, state_dim, num_actions, hidden_node, activation):
        """Create and return the network; implemented by subclasses."""
        pass

    @abstractmethod
    def action(self, state):
        """Select an action for the given state; implemented by subclasses."""
        pass

    @abstractmethod
    def train(self, replay_buffer):
        """Perform a training update; implemented by subclasses."""
        pass

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        self.Q_target.load_state_dict(self.Q.state_dict())


class BCQ(BaseQ):
    def __init__(self,
                 num_actions,
                 state_dim,
                 device='cuda',
                 bcq_threshold=0.3,
                 discount=0.99,
                 optimizer='Adam',
                 optimizer_parameters={},
                 use_polyak_target_update=False,
                 target_update_frequency=1,
                 tau=0.005,
                 algorithm="-",
                 hidden_node=0,
                 activation="-",
                 **kwargs):
        self.threshold = bcq_threshold
        self.algorithm = algorithm
        super(BCQ, self).__init__(num_actions, state_dim, device, discount, optimizer,
                                            optimizer_parameters, use_polyak_target_update,
                                            target_update_frequency, tau, hidden_node, activation, **kwargs)

        # Create the network; subclasses override build_network().
        self.Q = self.build_network(state_dim, num_actions, hidden_node, activation).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)
        
        # Track the total training steps and use the first 10% for warmup.
        self.max_training_steps = kwargs.get("max_timesteps")
        self.warmup_steps = int(0.1 * self.max_training_steps)
        self.training_step_count = 0

        # Apply learning-rate warmup followed by cosine decay.
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # During warmup, increase the learning rate linearly with the training step.
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Apply cosine decay after warmup.
                progress = float(current_step - self.warmup_steps) / float(max(1, self.max_training_steps - self.warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
                
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.Q_optimizer, lr_lambda=lr_lambda)


    def build_network(self, state_dim, num_actions, hidden_node, activation):
        return BCQNet(state_dim, num_actions, hidden_node, activation)


    def action(self, state):
        with torch.no_grad():
            state = state.to(self.device)
            q, imt, i = self.Q(state)
            imt = imt.exp()
            # Normalize imt and retain actions whose normalized value exceeds the threshold.
            imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()
        # Select the action with the highest Q-value after masking.
        return int((imt * q + (1. - imt) * -1e8).argmax(1))

    def train(self, replay_buffer):
        self.Q.train()
     
        state, action, next_state, reward, done, _ = replay_buffer.sample()
      
        # Compute the target Q-value using the BCQ action mask.
        with torch.no_grad():
            q, imt, i = self.Q(next_state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)
            q, imt, i = self.Q_target(next_state)
            target_Q = reward + (1 - done) * self.discount * q.gather(1, next_action).reshape(-1, 1)

        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action)

        # Combine the Q loss with the additional imitation-branch loss.
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt, action.reshape(-1))
        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()
        # self.lr_scheduler.step()

        self.iterations += 1
        if self.iterations % self.target_update_frequency == 0:
            self.maybe_update_target()


class FixedLearningRateBCQ(BCQ):
    def __init__(self, **params):
        super().__init__(**params)
        configured_lr = params['optimizer_parameters']['lr']
        for group in self.Q_optimizer.param_groups:
            group['lr'] = configured_lr


class HeparinBCQ(FixedLearningRateBCQ):
    def train(self, replay_buffer):
        # BatchNorm buffers are part of the target: freeze them between copies.
        self.Q_target.eval()
        return super().train(replay_buffer)


class CQL(BaseQ):
    def __init__(self,
                 num_actions,
                 state_dim,
                 device='cuda',
                 discount=0.99,
                 optimizer='Adam',
                 optimizer_parameters={},
                 use_polyak_target_update=False,
                 target_update_frequency=25,
                 tau=0.005,
                 hidden_node=128,
                 activation='relu',
                 cql_alpha=1.0,
                 **kwargs):
        self.cql_alpha = cql_alpha
        super(CQL, self).__init__(num_actions,
                                          state_dim,
                                          device,
                                          discount,
                                          optimizer,
                                          optimizer_parameters,
                                          use_polyak_target_update,
                                          target_update_frequency,
                                          tau,
                                          hidden_node,
                                          activation,
                                          **kwargs)
    # Create the network; subclasses override build_network().
        self.Q = self.build_network(state_dim, num_actions, hidden_node, activation).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        # Track the total training steps and use the first 10% for warmup.
        # self.max_training_steps = kwargs.get("max_timesteps")
        # self.warmup_steps = int(0.1 * self.max_training_steps)
        # self.training_step_count = 0
        # Apply learning-rate warmup followed by cosine decay.
        # def lr_lambda(current_step):
        #     if current_step < self.warmup_steps:
        # During warmup, increase the learning rate linearly with the training step.
        #         return float(current_step) / float(max(1, self.warmup_steps))
        #     else:
        # Apply cosine decay after warmup.
        #         progress = float(current_step - self.warmup_steps) / float(max(1, self.max_training_steps - self.warmup_steps))
        #         return 0.5 * (1.0 + math.cos(math.pi * progress))
                
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.Q_optimizer, lr_lambda=lr_lambda)

    def build_network(self, state_dim, num_actions, hidden_node, activation):
        return CQLNet(state_dim, num_actions, hidden_node, activation)

    def action(self, state):
        with torch.no_grad():
            q = self.Q(state.to(self.device))
            return int(q.argmax(dim=1))

    def train(self, replay_buffer):
        self.Q.train()
        state, action, next_state, reward, done, _ = replay_buffer.sample()

        # Target Q (DQN style)
        with torch.no_grad():
            q_next = self.Q(next_state)
            next_act = q_next.argmax(dim=1, keepdim=True)

            q_next_target = self.Q_target(next_state)
            target_q_value = q_next_target.gather(1, next_act)
            target_q = reward + (1 - done) * self.discount * target_q_value

        # Current Q
        q_pred = self.Q(state)
        current_q = q_pred.gather(1, action)

        # TD loss
        td_loss = F.mse_loss(current_q, target_q)

        # Bellman error print
        # bellman_error = (target_q - current_q).abs().mean().item()
        # print(f"Bellman error: {bellman_error:.6f}")

        # CQL conservative loss
        alpha = self.cql_alpha
        lse = torch.logsumexp(q_pred / alpha, dim=1, keepdim=True) * alpha
        data_q = current_q.mean()
        cql_loss = lse.mean() - data_q

        # Total loss
        loss = td_loss + cql_loss

        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
        # self.lr_scheduler.step()

        self.iterations += 1
        if self.iterations % self.target_update_frequency == 0:
            self.maybe_update_target()


class DDQN(BaseQ):
    def __init__(self,
                 num_actions,
                 state_dim,
                 device='cuda',
                 discount=0.99,
                 optimizer='Adam',
                 optimizer_parameters={},
                 use_polyak_target_update=False,
                 target_update_frequency=25,
                 tau=0.005,
                 hidden_node=128,
                 activation='relu',
                 cql_alpha=1.0,
                 **kwargs):
        self.cql_alpha = cql_alpha
        super(DDQN, self).__init__(num_actions,
                                          state_dim,
                                          device,
                                          discount,
                                          optimizer,
                                          optimizer_parameters,
                                          use_polyak_target_update,
                                          target_update_frequency,
                                          tau,
                                          hidden_node,
                                          activation,
                                          **kwargs)
    # Create the network; subclasses override build_network().
        self.Q = self.build_network(state_dim, num_actions, hidden_node, activation).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        # Track the total training steps and use the first 10% for warmup.
        # self.max_training_steps = kwargs.get("max_timesteps")
        # self.warmup_steps = int(0.1 * self.max_training_steps)
        # self.training_step_count = 0
        # Apply learning-rate warmup followed by cosine decay.
        # def lr_lambda(current_step):
        #     if current_step < self.warmup_steps:
        # During warmup, increase the learning rate linearly with the training step.
        #         return float(current_step) / float(max(1, self.warmup_steps))
        #     else:
        # Apply cosine decay after warmup.
        #         progress = float(current_step - self.warmup_steps) / float(max(1, self.max_training_steps - self.warmup_steps))
        #         return 0.5 * (1.0 + math.cos(math.pi * progress))
                
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.Q_optimizer, lr_lambda=lr_lambda)

    def build_network(self, state_dim, num_actions, hidden_node, activation):
        return CQLNet(state_dim, num_actions, hidden_node, activation)

    def action(self, state):
        with torch.no_grad():
            q = self.Q(state.to(self.device))
            return int(q.argmax(dim=1))

    def train(self, replay_buffer):
        self.Q.train()
        state, action, next_state, reward, done, _ = replay_buffer.sample()

        # Target Q (DQN style)
        with torch.no_grad():
            q_next = self.Q(next_state)
            next_act = q_next.argmax(dim=1, keepdim=True)

            q_next_target = self.Q_target(next_state)
            target_q_value = q_next_target.gather(1, next_act)
            target_q = reward + (1 - done) * self.discount * target_q_value

        # Current Q
        q_pred = self.Q(state)
        current_q = q_pred.gather(1, action)

        # TD loss
        td_loss = F.mse_loss(current_q, target_q)

        # Total loss
        loss = td_loss

        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
        # self.lr_scheduler.step()

        self.iterations += 1
        if self.iterations % self.target_update_frequency == 0:
            self.maybe_update_target()


class DQN(BaseQ):
    def __init__(self,
                 num_actions,
                 state_dim,
                 device='cuda',
                 discount=0.99,
                 optimizer='Adam',
                 optimizer_parameters={},
                 use_polyak_target_update=False,
                 target_update_frequency=25,
                 tau=0.005,
                 hidden_node=128,
                 activation='relu',
                 cql_alpha=1.0,
                 **kwargs):
        self.cql_alpha = cql_alpha
        super(DQN, self).__init__(num_actions,
                                          state_dim,
                                          device,
                                          discount,
                                          optimizer,
                                          optimizer_parameters,
                                          use_polyak_target_update,
                                          target_update_frequency,
                                          tau,
                                          hidden_node,
                                          activation,
                                          **kwargs)
    # Create the network; subclasses override build_network().
        self.Q = self.build_network(state_dim, num_actions, hidden_node, activation).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        # Track the total training steps and use the first 10% for warmup.
        # self.max_training_steps = kwargs.get("max_timesteps")
        # self.warmup_steps = int(0.1 * self.max_training_steps)
        # self.training_step_count = 0
        # Apply learning-rate warmup followed by cosine decay.
        # def lr_lambda(current_step):
        #     if current_step < self.warmup_steps:
        # During warmup, increase the learning rate linearly with the training step.
        #         return float(current_step) / float(max(1, self.warmup_steps))
        #     else:
        # Apply cosine decay after warmup.
        #         progress = float(current_step - self.warmup_steps) / float(max(1, self.max_training_steps - self.warmup_steps))
        #         return 0.5 * (1.0 + math.cos(math.pi * progress))
                
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.Q_optimizer, lr_lambda=lr_lambda)

    def build_network(self, state_dim, num_actions, hidden_node, activation):
        return DQNNet(state_dim, num_actions, hidden_node, activation)

    def action(self, state):
        with torch.no_grad():
            q = self.Q(state.to(self.device))
            return int(q.argmax(dim=1))

    def train(self, replay_buffer):
        self.Q.train()
        state, action, next_state, reward, done, _ = replay_buffer.sample()

        # Target Q (Double DQN style)
        with torch.no_grad():
            q_next = self.Q(next_state)
            next_act = q_next.argmax(dim=1, keepdim=True)

            q_next_target = self.Q_target(next_state)
            target_q_value = q_next_target.gather(1, next_act)
            target_q = reward + (1 - done) * self.discount * target_q_value

        # Current Q
        q_pred = self.Q(state)
        current_q = q_pred.gather(1, action)

        # TD loss
        td_loss = F.mse_loss(current_q, target_q)

        # Total loss
        loss = td_loss

        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
        # self.lr_scheduler.step()

        self.iterations += 1
        if self.iterations % self.target_update_frequency == 0:
            self.maybe_update_target()


class StandardDQN(DQN):
    def train(self, replay_buffer):
        self.Q.train()
        state, action, next_state, reward, done, _ = replay_buffer.sample()
        with torch.no_grad():
            target_q = reward + (1 - done) * self.discount * self.Q_target(next_state).max(dim=1, keepdim=True).values
        current_q = self.Q(state).gather(1, action)
        loss = F.mse_loss(current_q, target_q)
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
        self.iterations += 1
        if self.iterations % self.target_update_frequency == 0:
            self.maybe_update_target()


# Evaluated policies

def pred_q_value(algorithm, policy, state):
    output = policy.Q(state)
    return output[0] if algorithm == 'BCQ' else output


def pred_action(algorithm, policy, state):
    return policy.action(state)


def frozen_policy_arrays(algorithm, policy, state, *, mode='greedy', temperature=1., batch_size=1024, device=None):
    """Freeze the actual greedy policy (including the BCQ imitation mask)."""
    device = device or next(policy.Q.parameters()).device
    policy.Q.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(state), batch_size):
            batch = torch.as_tensor(np.asarray(state[start:start+batch_size]), dtype=torch.float32, device=device)
            output = policy.Q(batch)
            q = output[0] if algorithm == 'BCQ' else output
            if algorithm == 'BCQ' and mode == 'greedy':
                imitation = output[1].exp()
                mask = imitation / imitation.max(1, keepdim=True).values > policy.threshold
                q = torch.where(mask, q, torch.full_like(q, -1e8))
            parts.append(q.cpu().numpy())
    return policy_probs(np.concatenate(parts), mode=mode, temperature=temperature)


# Constrained policy learning

def _check_probs(b, q):
    b = np.asarray(b, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if b.shape != q.shape or b.ndim != 2 or not np.isfinite(b).all() or not np.isfinite(q).all():
        raise ValueError('Behavior and Q must be finite matching matrices')
    if (b < 0).any() or not np.allclose(b.sum(1), 1, atol=1e-8, rtol=0):
        raise ValueError('Invalid behavior probabilities')
    if (np.abs(q) > 1.000001).any():
        raise ValueError('Policy tilt requires Q bounded in [-1,1]')
    return b, q


def constrained_probs(q, behavior, beta, action_counts, support_probability=.01, min_count=100):
    """Keep rare-action mass fixed; exponentially tilt within eligible actions.

    For eligible E, pi_a=b_a * m_E * exp(beta Q_a)/sum_E b exp(beta Q).
    Else pi_a=b_a. Thus rare actions retain exactly their frozen baseline mass,
    and the density ratio against THAT fitted baseline is at most exp(2 beta).
    This is an actor definition, not clipping of evaluation importance weights.
    """
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('beta must be nonnegative')
    b, q = _check_probs(behavior, q)
    counts = np.asarray(action_counts)
    if counts.shape != (b.shape[1],):
        raise ValueError('action_counts must cover every action')
    if beta == 0:
        return b.copy()
    eligible = (b >= support_probability) & (counts[None, :] >= min_count)
    mass = np.sum(np.where(eligible, b, 0.), axis=1, keepdims=True)
    weighted = np.where(eligible, b * np.exp(beta * q), 0.)
    denom = weighted.sum(axis=1, keepdims=True)
    factor = np.divide(mass, denom, out=np.ones_like(mass), where=denom > 0)
    result = np.where(eligible, weighted * factor, b)
    if not np.allclose(result.sum(1), 1., atol=1e-8, rtol=0):
        raise AssertionError('Actor mass is not preserved')
    return result


def _torch_actor(q, b, beta, counts, support_probability, min_count):
    eligible = (b >= support_probability) & (counts[None, :] >= min_count)
    mass = torch.where(eligible, b, 0.).sum(1, keepdim=True)
    weighted = torch.where(eligible, b * torch.exp(beta * q), 0.)
    denom = weighted.sum(1, keepdim=True)
    safe = torch.where(denom > 0, denom, torch.ones_like(denom))
    return torch.where(eligible, weighted * mass / safe, b)


def fit_policy(state, next_state, action, reward, done, behavior_next, groups,
               *, beta=.1, greedy=False, action_counts=None, seed=101,
               hidden_dim=128, epochs=80, min_epochs=30, patience=15,
               batch_size=4096, lr=1e-3, cql_coefficient=.1, gamma=.98,
               support_probability=.01, min_count=100, num_threads=4,
               progress_callback=None):
    """Fit with frozen BC and evolving Q tilt in the target (or greedy control).

    Evaluation later freezes the entire actor, including the selected Q model.
    """
    if not 0 <= gamma < 1 or not 1 <= min_epochs <= epochs:
        raise ValueError('Invalid discount or epoch bounds')
    if min(patience,batch_size,num_threads,hidden_dim) < 1 or lr <= 0 or cql_coefficient < 0:
        raise ValueError('Invalid optimizer configuration')
    if beta < 0 or not np.isfinite(beta):
        raise ValueError('Invalid beta')
    s, ns, a, r, d, b, _ = validate_transitions(
        state, next_state, action, reward, done, behavior_next)
    counts = np.bincount(a, minlength=b.shape[1]) if action_counts is None else np.asarray(action_counts)
    train_idx, val_idx, _, _, _ = _trajectory_split(d, r, .15, seed, groups)
    tensors = tuple(torch.from_numpy(v) for v in [s, ns, a, r, d, b])
    count_tensor = torch.from_numpy(counts)
    torch.set_num_threads(num_threads)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FQECritic(s.shape[1], b.shape[1], hidden_dim)
    target = copy.deepcopy(model).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 1)
    best, best_state, best_epoch, stale = float('inf'), None, 0, 0
    history = []
    started = time.monotonic()

    def bellman(qnext, behavior):
        if greedy:
            return qnext.max(1).values
        actor = _torch_actor(qnext, behavior, beta, count_tensor, support_probability, min_count)
        return (actor * qnext).sum(1)

    def residual(indices):
        errors = []
        model.eval()
        with torch.no_grad():
            for begin in range(0, len(indices), batch_size):
                ix = torch.from_numpy(indices[begin:begin+batch_size])
                observed = model(tensors[0][ix]).gather(1, tensors[2][ix, None]).squeeze(1)
                y = tensors[3][ix] + gamma * (1-tensors[4][ix]) * bellman(model(tensors[1][ix]), tensors[5][ix])
                errors.append((observed-y).numpy())
        error = np.concatenate(errors).astype(np.float64)
        term = d[indices] == 1
        tm = float(np.mean(error[term]**2)) if term.any() else None
        nm = float(np.mean(error[~term]**2)) if (~term).any() else None
        return dict(terminal_mse=tm, nonterminal_mse=nm,
                    balanced_mse=float(np.mean([x for x in [tm,nm] if x is not None])))

    for epoch in range(1, epochs+1):
        model.train()
        total_td = total_cql = 0.
        shuffled = rng.permutation(train_idx)
        for start in range(0, len(shuffled), batch_size):
            ix = torch.from_numpy(shuffled[start:start+batch_size])
            q = model(tensors[0][ix])
            observed = q.gather(1, tensors[2][ix, None]).squeeze(1)
            with torch.no_grad():
                y = tensors[3][ix] + gamma * (1-tensors[4][ix]) * bellman(target(tensors[1][ix]), tensors[5][ix])
            td = ((observed-y)**2).mean()
            conservative = (torch.logsumexp(q, dim=1) - observed).mean()
            loss = td + cql_coefficient * conservative
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_td += float(td.detach()) * len(ix)
            total_cql += float(conservative.detach()) * len(ix)
        metric = residual(val_idx)
        log = dict(epoch=epoch, td_mse=total_td/len(train_idx),
                   cql_penalty=total_cql/len(train_idx), validation=metric,
                   elapsed_seconds=time.monotonic()-started)
        history.append(log)
        if epoch >= min_epochs and metric['balanced_mse'] < best-1e-6:
            best, best_epoch = metric['balanced_mse'], epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        elif epoch >= min_epochs:
            stale += 1
        target.load_state_dict(model.state_dict())
        if progress_callback:
            progress_callback(log)
        if epoch >= min_epochs and stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    report = dict(seed=seed, beta=beta, greedy=greedy, hidden_dim=hidden_dim,
        selected_epoch=best_epoch, epochs_completed=len(history), history=history,
        config=dict(epochs=epochs,min_epochs=min_epochs,patience=patience,batch_size=batch_size,
        lr=lr,cql_coefficient=cql_coefficient,gamma=gamma,support_probability=support_probability,
        min_count=min_count,num_threads=num_threads),
        q_bounds=[-1,1], target_actor='greedy' if greedy else 'frozen behavior, bounded exponential Q tilt',
        train_rows=len(train_idx), validation_rows=len(val_idx),
        train_index_sha256=hashlib.sha256(train_idx.tobytes()).hexdigest(),
        validation_index_sha256=hashlib.sha256(val_idx.tobytes()).hexdigest(),
        selected_validation=residual(val_idx), elapsed_seconds=time.monotonic()-started)
    return model, report


# Behavior policy estimation

def _validate_sensitivity_partition(state, action, done, groups, num_actions, partition):
    """Validate exact caller partitions without drawing a new patient split."""
    state, action = validate_data(state, action, done, num_actions=num_actions)
    if state.shape[1] < 1 or len(state) < 1 or groups is None:
        raise ValueError("Nonempty states and verified patient groups are required")
    groups = np.asarray(groups).reshape(-1)
    if len(groups) != len(state):
        raise ValueError("Patient groups must align with rows")
    names = ("fit", "calibration", "validation")
    if set(partition) != set(names):
        raise ValueError("Partition must contain exactly fit/calibration/validation")
    row_owner = np.full(len(state), -1, dtype=np.int64)
    exact = {}
    group_sets = {}
    for owner, name in enumerate(names):
        raw = np.asarray(partition[name])
        if (raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer)
                or (raw < 0).any() or (raw >= len(state)).any()
                or len(np.unique(raw)) != len(raw)):
            raise ValueError("Invalid or duplicate " + name + " row indices")
        indices = raw.astype(np.int64)
        if (row_owner[indices] != -1).any():
            raise ValueError("Partition row indices overlap")
        row_owner[indices] = owner
        exact[name] = indices.copy()
        group_sets[name] = set(groups[indices].tolist())
    if (row_owner == -1).any():
        raise ValueError("Partition must cover every supplied row exactly once")
    for left, right in [("fit", "calibration"), ("fit", "validation"),
                        ("calibration", "validation")]:
        if group_sets[left] & group_sets[right]:
            raise ValueError("Patient groups overlap across partitions")
    for episode in episode_slices(done):
        if len(np.unique(groups[episode])) != 1:
            raise ValueError("Each episode must belong to one verified patient")
        if len(np.unique(row_owner[episode])) != 1:
            raise ValueError("An episode crosses partition boundaries")
    missing = np.setdiff1d(np.arange(num_actions), np.unique(action[exact["fit"]]))
    if len(missing):
        raise ValueError("fit patients lack action classes: " + str(missing.tolist()))
    return state, action, groups, exact


def _fit_sensitivity_temperature(estimator, state, action, num_actions):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    from model import BehaviorLogitPredictor

    raw = BehaviorLogitPredictor(estimator, num_actions)
    scores = raw.decision_function(state)
    action = np.asarray(action, dtype=np.int64)

    def loss(log_temperature):
        scaled = scores / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1)
                             - scaled[np.arange(len(scaled)), action]))

    optimum = minimize_scalar(loss, bounds=(-2., 2.), method="bounded")
    if not optimum.success or not np.isfinite(optimum.fun):
        raise ValueError("Temperature calibration failed: " + str(optimum.message))
    return BehaviorLogitPredictor(estimator, num_actions, np.exp(optimum.x)), {
        "method": "calibration-only negative log likelihood",
        "log_temperature_bounds": [-2., 2.],
        "temperature": float(np.exp(optimum.x)),
        "log_temperature": float(optimum.x),
        "calibration_log_loss": float(optimum.fun),
        "uncalibrated_calibration_log_loss": loss(0.),
        "near_search_boundary": bool(min(optimum.x+2., 2.-optimum.x) < 1e-3),
    }


def _audit_sensitivity_logistic_logits(estimator, state, action):
    """Compare sklearn logits to direct scalar-product arithmetic, without BLAS."""
    from scipy.special import logsumexp

    scaled = estimator[0].transform(state)
    classifier = estimator[-1]
    if (not np.isfinite(classifier.coef_).all()
            or not np.isfinite(classifier.intercept_).all()):
        raise ValueError("Non-finite fitted logistic parameters")
    actual = np.asarray(estimator.decision_function(state), dtype=np.float64)
    reference = np.einsum("ij,kj->ik", scaled, classifier.coef_, optimize=False)
    reference += classifier.intercept_
    if actual.ndim == 1:
        reference = reference[:, 0]
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite fitted logistic calibration logits")
    if not np.allclose(actual, reference, rtol=1e-10, atol=1e-8):
        raise ValueError("Logistic BLAS logits disagree with direct arithmetic")
    difference = float(np.max(np.abs(actual-reference)))
    if reference.ndim == 1:
        reference = np.column_stack([np.zeros(len(reference)), reference])
    columns = np.empty(len(classifier.classes_), dtype=int)
    columns[classifier.classes_] = np.arange(len(classifier.classes_))
    chosen = columns[np.asarray(action, dtype=np.int64)]
    nll = float(np.mean(logsumexp(reference, axis=1)
                        - reference[np.arange(len(reference)), chosen]))
    return {"coefficients_finite": True, "intercepts_finite": True,
        "calibration_logits_finite": True, "reference": "einsum optimize=False scalar products",
        "max_abs_logit_difference": difference, "reference_calibration_log_loss": nll,
        "calibration_logit_range": [float(actual.min()), float(actual.max())]}


def _train_sensitivity_mlp(state, action, calibration_state, calibration_action,
                           scaler, hidden_sizes, num_actions, seed, progress_callback):
    """Train CE on fit rows; calibration NLL controls early stopping only."""
    from model import NumpyMLPBehaviorPredictor
    from threadpoolctl import threadpool_limits

    fit_x = torch.from_numpy(scaler.transform(state).astype(np.float32))
    fit_y = torch.from_numpy(np.asarray(action, dtype=np.int64))
    cal_x = torch.from_numpy(scaler.transform(calibration_state).astype(np.float32))
    cal_y = np.asarray(calibration_action, dtype=np.int64)
    rng = np.random.default_rng(seed)
    history = []
    started = time.monotonic()
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), threadpool_limits(limits=1):
            torch.manual_seed(seed)
            layers = []
            previous = fit_x.shape[1]
            for hidden in hidden_sizes:
                layers.extend([torch.nn.Linear(previous, hidden), torch.nn.ReLU()])
                previous = hidden
            layers.append(torch.nn.Linear(previous, num_actions))
            network = torch.nn.Sequential(*layers)
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
            for epoch in range(1, 101):
                network.train()
                total = 0.
                for begin in range(0, len(fit_x), 512):
                    # Generate one fixed-seed permutation for the entire epoch.
                    if begin == 0:
                        order = rng.permutation(len(fit_x))
                    index = torch.from_numpy(order[begin:begin+512])
                    loss = F.cross_entropy(network(fit_x[index]), fit_y[index])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())*len(index)
                network.eval()
                losses = []
                with torch.no_grad():
                    for begin in range(0, len(cal_x), 2048):
                        logits = network(cal_x[begin:begin+2048]).double()
                        y = torch.from_numpy(cal_y[begin:begin+2048])
                        losses.append((torch.logsumexp(logits, dim=1)
                                       - logits[torch.arange(len(y)), y]).numpy())
                cal_loss = float(np.concatenate(losses).mean())
                if not np.isfinite(cal_loss):
                    raise ValueError("Non-finite MLP calibration negative log likelihood")
                record = {"epoch": epoch, "fit_cross_entropy": total/len(fit_x),
                          "calibration_log_loss": cal_loss,
                          "elapsed_seconds": time.monotonic()-started}
                history.append(record)
                if cal_loss < best_loss-1e-6:
                    best_loss, best_epoch, stale = cal_loss, epoch, 0
                    best_state = copy.deepcopy(network.state_dict())
                else:
                    stale += 1
                if progress_callback and (epoch == 1 or epoch % 5 == 0 or stale >= 10):
                    progress_callback({"event": "mlp_epoch", "hidden_sizes": list(hidden_sizes),
                                       **record})
                if stale >= 10:
                    break
            network.load_state_dict(best_state)
            linears = [layer for layer in network if isinstance(layer, torch.nn.Linear)]
            predictor = NumpyMLPBehaviorPredictor(scaler,
                [layer.weight.detach().numpy() for layer in linears],
                [layer.bias.detach().numpy() for layer in linears], num_actions)
    finally:
        torch.set_num_threads(prior_threads)
    return predictor, {"hidden_sizes": list(hidden_sizes), "activation": "relu",
        "objective": "multiclass cross entropy", "optimizer": "Adam", "learning_rate": .001,
        "batch_size": 512, "max_epochs": 100, "patience": 10, "thread_limit": 1,
        "selected_epoch": best_epoch, "epochs_completed": len(history),
        "selected_calibration_log_loss": best_loss, "history": history,
        "elapsed_seconds": time.monotonic()-started}


def fit_behavior_sensitivity_candidates(state, action, done, groups, num_actions,
                                        partition, seed=42, n_jobs=4, progress_callback=None):
    """Fit fifteen behavior sensitivity candidates on exact caller partitions.

    Returns ``(predictors, report)``. This experimental API neither selects a
    production behavior model nor changes the existing RF fitting protocol.
    Fit rows train models/scalers/priors; calibration rows tune temperature,
    count smoothing and MLP early stopping; validation rows provide diagnostics.
    The JSON-safe report contains hashes, not raw row indices or patient IDs.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from model import (BehaviorLogitPredictor, NeighborCountBehaviorPredictor,
                       ClusterCountBehaviorPredictor)

    if not isinstance(num_actions, (int, np.integer)) or num_actions < 2:
        raise ValueError("num_actions must be an integer >= 2")
    if not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    state, action, groups, exact = _validate_sensitivity_partition(
        state, action, done, groups, num_actions, partition)
    fit, cal, val = [exact[name] for name in ["fit", "calibration", "validation"]]
    started = time.monotonic()
    candidates, training, calibration, validation = {}, {}, {}, {}
    scaler = StandardScaler().fit(state[fit])
    prior = np.bincount(action[fit], minlength=num_actions).astype(float)/len(fit)
    grid = [.1, 1., 10., 100., 1000.]

    def progress(event):
        if progress_callback:
            progress_callback({**event, "total_elapsed_seconds": time.monotonic()-started})

    def register(name, predictor, params, cal_probability=None, val_probability=None):
        predictor.params = {"selected_name": name, **params,
            "standard_scaler_scope": "caller fit rows only", "num_actions": int(num_actions)}
        candidates[name] = predictor
        cp = predictor.predict_proba(state[cal]) if cal_probability is None else cal_probability
        vp = predictor.predict_proba(state[val]) if val_probability is None else val_probability
        for probability in [cp, vp]:
            if (not np.isfinite(probability).all() or (probability < 0).any()
                    or not np.allclose(probability.sum(axis=1), 1., atol=1e-8)):
                raise ValueError("Invalid candidate probabilities: " + name)
        calibration[name] = probability_metrics(action[cal], cp, num_actions=num_actions)
        validation[name] = probability_metrics(action[val], vp, num_actions=num_actions)
        progress({"event": "candidate_complete", "candidate": name,
                  "validation_log_loss": validation[name]["log_loss"]})

    def smoothing(predictor, cal_counts):
        losses = []
        for concentration in grid:
            probability = ((cal_counts+concentration*prior)
                           /(cal_counts.sum(axis=1, keepdims=True)+concentration))
            observed = probability[np.arange(len(cal)), action[cal]]
            losses.append(float(-np.log(observed).mean()))
        selected = int(np.argmin(losses))
        predictor.concentration = grid[selected]
        return {"method": "calibration-only negative log likelihood",
                "concentration_grid": grid.copy(), "calibration_log_losses": losses,
                "selected_concentration": grid[selected],
                "fit_action_prior": prior.tolist(), "selected_grid_boundary": selected in [0, len(grid)-1]}

    with threadpool_limits(limits=n_jobs):
        progress({"event": "family_start", "family": "logistic"})
        estimator = make_pipeline(scaler, LogisticRegression(C=1., solver="lbfgs",
            max_iter=1000, random_state=seed))
        # StandardScaler inside this pipeline is re-fit on these same fit rows.
        estimator.fit(state[fit], action[fit])
        register("logistic_C1_raw", BehaviorLogitPredictor(estimator, num_actions),
                 {"family": "multinomial_logistic", "C": 1., "calibration": "none"})
        tempered, temperature_report = _fit_sensitivity_temperature(
            estimator, state[cal], action[cal], num_actions)
        register("logistic_C1_temperature", tempered,
                 {"family": "multinomial_logistic", "C": 1., "calibration": "temperature",
                  "temperature": tempered.temperature})
        iterations = estimator[-1].n_iter_.astype(int).tolist()
        training["logistic_C1"] = {"solver": "lbfgs", "max_iter": 1000,
            "iterations": iterations, "iteration_limit_reached": any(x >= 1000 for x in iterations),
            "numerical_audit": _audit_sensitivity_logistic_logits(estimator, state[cal], action[cal]),
            "temperature_calibration": temperature_report}
    for hidden_sizes in [(64, 64), (128, 128), (1000,)]:
        label = "mlp_" + "x".join(map(str, hidden_sizes))
        progress({"event": "family_start", "family": label})
        estimator, mlp_report = _train_sensitivity_mlp(state[fit], action[fit],
            state[cal], action[cal], scaler, hidden_sizes, num_actions, seed, progress)
        with threadpool_limits(limits=n_jobs):
            register(label+"_raw", BehaviorLogitPredictor(estimator, num_actions),
                     {"family": "mlp", "hidden_sizes": list(hidden_sizes), "calibration": "none"})
            tempered, temperature_report = _fit_sensitivity_temperature(
                estimator, state[cal], action[cal], num_actions)
            register(label+"_temperature", tempered,
                     {"family": "mlp", "hidden_sizes": list(hidden_sizes),
                      "calibration": "temperature", "temperature": tempered.temperature})
        training[label] = {**mlp_report, "temperature_calibration": temperature_report}
    with threadpool_limits(limits=n_jobs):
        progress({"event": "family_start", "family": "nearest_neighbors"})
        transformed_fit = scaler.transform(state[fit])
        neighbors = NearestNeighbors(algorithm="auto", n_jobs=n_jobs).fit(transformed_fit)
        neighbor_candidates = {k: NeighborCountBehaviorPredictor(scaler, neighbors,
            action[fit], prior, k, query_neighbors=1000) for k in [100, 300, 1000]}
        # One bounded query per partition; prefix neighbors serve every k.
        cached = {}
        for split_name, indices in [("calibration", cal), ("validation", val)]:
            counts = {k: np.zeros((len(indices), num_actions), dtype=float) for k in neighbor_candidates}
            maximum = min(1000, len(fit))
            for begin in range(0, len(indices), 256):
                query = scaler.transform(state[indices[begin:begin+256]])
                nearest = neighbors.kneighbors(query, n_neighbors=maximum, return_distance=False)
                actions = action[fit][nearest]
                for k in neighbor_candidates:
                    actual_k = min(k, len(fit))
                    local = counts[k][begin:begin+len(query)]
                    np.add.at(local, (np.repeat(np.arange(len(query)), actual_k),
                                     actions[:, :actual_k].reshape(-1)), 1.)
                if begin == 0 or (begin//256+1) % 20 == 0 or begin+256 >= len(indices):
                    progress({"event": "neighbor_query", "split": split_name,
                              "rows_completed": min(begin+256, len(indices)), "rows": len(indices)})
            cached[split_name] = counts
        for k, predictor in neighbor_candidates.items():
            label = "knn_k"+str(k)
            smoothing_report = smoothing(predictor, cached["calibration"][k])
            register(label, predictor, {"family": "knn_empirical_actions",
                "requested_neighbors": k, "effective_neighbors": predictor.n_neighbors,
                "smoothing": "Dirichlet toward fit action prior",
                "concentration": predictor.concentration},
                predictor.probabilities_from_counts(cached["calibration"][k]),
                predictor.probabilities_from_counts(cached["validation"][k]))
            training[label] = {"neighbor_algorithm": neighbors._fit_method,
                "metric": "euclidean", "query_batch_size": 256, "n_jobs": int(n_jobs),
                "shared_query_neighbors": min(1000, len(fit)),
                "smoothing_calibration": smoothing_report}
        for requested in [50, 100, 300, 750]:
            label = "cluster_k"+str(requested)
            progress({"event": "family_start", "family": label})
            effective = min(requested, len(fit))
            clustering = MiniBatchKMeans(n_clusters=effective, random_state=seed,
                batch_size=max(1024, 3*effective), n_init=3, max_iter=100, reassignment_ratio=.01)
            labels = clustering.fit_predict(transformed_fit)
            counts = np.zeros((effective, num_actions), dtype=float)
            np.add.at(counts, (labels, action[fit]), 1.)
            predictor = ClusterCountBehaviorPredictor(scaler, clustering, counts, prior)
            cal_counts = predictor.action_counts(state[cal])
            val_counts = predictor.action_counts(state[val])
            smoothing_report = smoothing(predictor, cal_counts)
            register(label, predictor, {"family": "cluster_empirical_actions",
                "requested_clusters": requested, "effective_clusters": effective,
                "smoothing": "Dirichlet toward fit action prior", "concentration": predictor.concentration},
                predictor.probabilities_from_counts(cal_counts),
                predictor.probabilities_from_counts(val_counts))
            training[label] = {"clustering": "MiniBatchKMeans", "n_init": 3,
                "batch_size": max(1024, 3*effective), "max_iter": 100,
                "iterations": int(clustering.n_iter_), "empty_fit_clusters": int((counts.sum(1) == 0).sum()),
                "smoothing_calibration": smoothing_report}
    report = {"scope": "supplied training patients only, exact caller partition",
        "seed": int(seed), "num_actions": int(num_actions), "candidate_order": list(candidates),
        "candidate_parameters": {name: model.params for name, model in candidates.items()},
        "selection_rule": "none; caller evaluates and selects experimental sensitivity model",
        "split": {name: {"rows": len(indices), "groups": len(np.unique(groups[indices])),
            "action_counts": np.bincount(action[indices], minlength=num_actions).tolist(),
            "row_indices_sha256_int64": _index_hash(indices)} for name, indices in exact.items()},
        "scaler": {"scope": "fit rows only", "mean": scaler.mean_.tolist(),
                   "scale": scaler.scale_.tolist()},
        "fit_action_prior": prior.tolist(), "training": training,
        "calibration": calibration, "validation": validation,
        "saved_probability_floor": None, "policy_support_mask": False, "final_refit": False,
        "elapsed_seconds": time.monotonic()-started}
    return candidates, report

def _index_hash(indices):
    return hashlib.sha256(np.ascontiguousarray(indices, dtype="<i8").tobytes()).hexdigest()


def _fit_temperature(estimator, state, action):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    classes = _classes(estimator)
    scores = np.asarray(estimator.decision_function(state), dtype=np.float64)
    if scores.shape != (len(state), NUM_ACTIONS) or not np.isfinite(scores).all():
        raise ValueError("Calibration requires finite 25-class logistic logits")
    columns = np.empty(NUM_ACTIONS, dtype=np.int64)
    columns[classes] = np.arange(NUM_ACTIONS)
    chosen = columns[np.asarray(action, dtype=np.int64)]

    def loss(log_temperature):
        scaled = scores / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(scaled)), chosen]))

    optimum = minimize_scalar(loss, bounds=(-2., 2.), method="bounded")
    if not optimum.success or not np.isfinite(optimum.fun):
        raise ValueError("Temperature calibration failed: " + str(optimum.message))
    return TemperatureScaledPredictor(estimator, np.exp(optimum.x)), {
        "method": "bounded scalar calibration-only multinomial negative log likelihood",
        "log_temperature_bounds": [-2., 2.], "log_temperature": float(optimum.x),
        "temperature": float(np.exp(optimum.x)), "calibration_log_loss": float(optimum.fun),
        "uncalibrated_calibration_log_loss": loss(0.),
        "near_search_boundary": bool(min(optimum.x + 2., 2. - optimum.x) < 1e-3),
        "optimizer_success": True,
    }


def fit_behavior_candidates(state, action, done, groups, seed=42, n_jobs=4):
    """Return selected predictor, report, and all eight frozen candidate predictors.

    Only supplied policy-training rows are used. All 25 actions must occur in
    internal fit and calibration patients; missing classes raise rather than
    silently dropping candidates. Validation action counts, including absent
    actions, are reported. Exact ties choose the first declared candidate:
    RF raw, RF sigmoid, then C=.1/1/10 logistic raw/temperature.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    state, action = validate_data(state, action, done)
    if groups is None:
        raise ValueError("Verified patient groups are required for behavior retraining")
    groups = np.asarray(groups).reshape(-1)
    partition = grouped_partition(done, groups, random_seed=seed)
    for name in ["fit", "calibration"]:
        missing = np.setdiff1d(np.arange(NUM_ACTIONS), np.unique(action[partition[name]]))
        if missing.size:
            raise ValueError(name + " patients lack action classes: " + str(missing.tolist()))
    raw, calibrated, rf_report = fit_behavior(state, action, done, groups=groups,
                                             random_seed=seed, n_jobs=n_jobs)
    actual = rf_report.pop("_partition_indices")
    if any(not np.array_equal(partition[name], actual[name]) for name in partition):
        raise ValueError("RF and logistic internal patient partitions disagree")
    fit_rows, cal_rows, val_rows = (actual[name] for name in ["fit", "calibration", "validation"])
    candidates = {"rf_raw": FullActionPredictor(raw),
                  "rf_sigmoid": FullActionPredictor(calibrated)}
    for name, predictor in candidates.items():
        predictor.params = {"selected_name": name, "family": "random_forest",
                            "calibration": "sigmoid" if name == "rf_sigmoid" else "none",
                            "rf_parameters": rf_report.get("rf_parameters")}
    training = {}
    with threadpool_limits(limits=4):
        for regularization in [.1, 1., 10.]:
            label = "logistic_C" + format(regularization, "g")
            estimator = make_pipeline(StandardScaler(), LogisticRegression(
                C=regularization, solver="lbfgs", max_iter=1000, random_state=seed))
            estimator.fit(state[fit_rows], action[fit_rows])
            candidates[label + "_raw"] = FullActionPredictor(estimator)
            tempered, calibration_report = _fit_temperature(estimator, state[cal_rows], action[cal_rows])
            candidates[label + "_temperature"] = tempered
            for method in ["raw", "temperature"]:
                predictor = candidates[label + "_" + method]
                predictor.params = {"selected_name": label + "_" + method,
                    "family": "multinomial_logistic", "C": regularization,
                    "calibration": "temperature" if method == "temperature" else "none",
                    "temperature": tempered.temperature if method == "temperature" else None,
                    "solver": "lbfgs", "max_iter": 1000,
                    "standard_scaler_scope": "internal fit patients only"}
            iterations = np.asarray(estimator[-1].n_iter_).astype(int).tolist()
            training[label] = {"C": regularization, "objective": "multinomial",
                "solver": "lbfgs", "max_iter": 1000, "thread_limit": 4,
                "standard_scaler_scope": "internal fit patients only",
                "iterations": iterations, "iteration_limit_reached": any(x >= 1000 for x in iterations),
                "temperature_calibration": calibration_report}
    validation = {name: _metrics(action[val_rows], predictor.predict_proba(state[val_rows]))
                  for name, predictor in candidates.items()}
    selected_name = min(candidates, key=lambda name: validation[name]["log_loss"])
    split_report = {}
    episodes = episode_slices(done)
    for name, indices in actual.items():
        chosen = np.zeros(len(state), dtype=bool)
        chosen[indices] = True
        split_report[name] = {"rows": len(indices), "groups": len(np.unique(groups[indices])),
            "episodes": sum(bool(chosen[sl.start]) for sl in episodes),
            "action_counts": np.bincount(action[indices], minlength=NUM_ACTIONS).tolist(),
            "row_indices_sha256_int64": _index_hash(indices),
            "group_labels_sha256_int64": _index_hash(np.unique(groups[indices]))}
    report = {"scope": "supplied policy-training patients only; no outer validation or test rows",
        "seed": seed, "candidate_order": list(candidates), "selected_behavior": selected_name,
        "candidate_parameters": {name: predictor.params for name, predictor in candidates.items()},
        "selection_rule": "minimum internal patient-validation multiclass log loss; exact ties use candidate_order",
        "split": split_report, "split_fraction_by_group": {"fit": .6, "calibration": .2, "validation": .2},
        "rf_training": rf_report, "logistic_training": training, "validation": validation,
        "validation_used_for_selection": True, "final_refit": False,
        "saved_probability_floor": None, "policy_support_mask": False,
        "absent_validation_actions": np.setdiff1d(np.arange(NUM_ACTIONS), np.unique(action[val_rows])).tolist(),
        "interpretation": "Fitted behavior probabilities and internal calibration do not establish true behavior support or target-state coverage",
        "_partition_indices": actual}
    return candidates[selected_name], report, candidates


def fit_selected_behavior(state, action, done, groups, selected_name, seed=42, n_jobs=4):
    """Refit only the preselected family/C/calibration procedure on training patients.

    A temperature-selected family estimates a fresh scalar on its own internal
    calibration patients. It does not reuse the original numeric temperature or
    reselect C/family using the resampled validation data. RF keeps the existing
    fit_behavior fitting/calibration procedure; its internal raw/sigmoid choice
    is ignored in favor of the caller's fixed selected_name.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    logistic_names = {"logistic_C" + format(c, "g") + "_" + method: (c, method)
                      for c in [.1, 1., 10.] for method in ["raw", "temperature"]}
    if selected_name not in {"rf_raw", "rf_sigmoid"} | set(logistic_names):
        raise ValueError("Unknown fixed behavior candidate: " + str(selected_name))
    state, action = validate_data(state, action, done)
    if groups is None:
        raise ValueError("Verified patient groups are required for behavior retraining")
    groups = np.asarray(groups).reshape(-1)
    partition = grouped_partition(done, groups, random_seed=seed)
    for name in ["fit", "calibration"]:
        missing = np.setdiff1d(np.arange(NUM_ACTIONS), np.unique(action[partition[name]]))
        if missing.size:
            raise ValueError(name + " patients lack action classes: " + str(missing.tolist()))
    family_report = {}
    if selected_name.startswith("rf_"):
        raw, calibrated, family_report = fit_behavior(state, action, done, groups=groups,
                                                       random_seed=seed, n_jobs=n_jobs)
        actual = family_report.pop("_partition_indices")
        if any(not np.array_equal(partition[name], actual[name]) for name in partition):
            raise ValueError("RF refit patient partition differs from declared partition")
        predictor = FullActionPredictor(raw if selected_name == "rf_raw" else calibrated)
        predictor.params = {"selected_name": selected_name, "family": "random_forest",
            "calibration": "sigmoid" if selected_name == "rf_sigmoid" else "none",
            "rf_parameters": family_report.get("rf_parameters")}
    else:
        regularization, method = logistic_names[selected_name]
        with threadpool_limits(limits=4):
            estimator = make_pipeline(StandardScaler(), LogisticRegression(
                C=regularization, solver="lbfgs", max_iter=1000, random_state=seed))
            estimator.fit(state[partition["fit"]], action[partition["fit"]])
            if method == "temperature":
                predictor, calibration_report = _fit_temperature(estimator,
                    state[partition["calibration"]], action[partition["calibration"]])
            else:
                predictor, calibration_report = FullActionPredictor(estimator), None
        iterations = np.asarray(estimator[-1].n_iter_).astype(int).tolist()
        predictor.params = {"selected_name": selected_name, "family": "multinomial_logistic",
            "C": regularization, "calibration": "temperature" if method == "temperature" else "none",
            "temperature": predictor.temperature if method == "temperature" else None,
            "solver": "lbfgs", "max_iter": 1000, "standard_scaler_scope": "internal fit patients only"}
        family_report = {"iterations": iterations, "iteration_limit_reached": any(x >= 1000 for x in iterations),
                         "temperature_calibration": calibration_report, "thread_limit": 4}
    report = {"selected_behavior": selected_name, "selected_parameters": predictor.params,
        "selection_rule": "caller-fixed family/C/calibration; no candidate reselection",
        "seed": seed, "scope": "supplied policy-training patients only",
        "family_fit": family_report, "saved_probability_floor": None, "final_refit": False,
        "split": {name: {"rows": len(indices), "groups": len(np.unique(groups[indices])),
            "action_counts": np.bincount(action[indices], minlength=NUM_ACTIONS).tolist(),
            "row_indices_sha256_int64": _index_hash(indices),
            "group_labels_sha256_int64": _index_hash(np.unique(groups[indices]))}
            for name, indices in partition.items()}, "_partition_indices": partition}
    return predictor, report


def fit_behavior(state, action, done, groups=None, random_seed=42, n_jobs=4,
                 calibration_fraction=.2, validation_fraction=.2, n_estimators=200,
                 num_actions=NUM_ACTIONS, selection="validation", require_all_actions=False):
    """Return (raw RF, frozen calibrated model, report) from train arrays only.

    This callable is also usable in train-trajectory bootstrap refits.  Test
    rows/labels never enter this function.  report['_partition_indices'] holds
    row indices for reproducible reuse and is removed before JSON serialization.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    if selection not in {'validation', 'calibrated'}:
        raise ValueError('Behavior selection must be validation or calibrated')
    if not isinstance(num_actions, int) or num_actions < 1:
        raise ValueError('num_actions must be a positive integer')
    state, action = validate_data(state, action, done, num_actions=num_actions)
    partition = grouped_partition(done, groups, random_seed, calibration_fraction, validation_fraction)
    fit_rows, cal_rows, val_rows = (partition[name] for name in ["fit", "calibration", "validation"])
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=20,
        min_samples_split=2, min_samples_leaf=4, max_features="sqrt",
        random_state=random_seed, n_jobs=n_jobs, bootstrap=True, oob_score=False)
    rf.fit(state[fit_rows], action[fit_rows])
    if require_all_actions and not np.array_equal(rf.classes_, np.arange(num_actions)):
        raise ValueError('Behavior fitting partition lacks an action class')
    missing_cal = np.setdiff1d(rf.classes_, np.unique(action[cal_rows]))
    if len(missing_cal):
        raise ValueError("calibration groups lack one or more RF action classes")
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(rf), method="sigmoid",
        ensemble=False, n_jobs=n_jobs)
    calibrated.fit(state[cal_rows], action[cal_rows])
    raw_prob = full_action_proba(rf, state[val_rows], num_actions=num_actions)
    calibrated_prob = full_action_proba(calibrated, state[val_rows], num_actions=num_actions)
    actual_groups = episode_groups(done) if groups is None else np.asarray(groups).reshape(-1)
    split_report = {}
    for name, indices in partition.items():
        selected = np.zeros(len(state), dtype=bool)
        selected[indices] = True
        split_report[name] = {"rows": len(indices), "groups": len(np.unique(actual_groups[indices])),
            "episodes": sum(bool(selected[rows.start]) for rows in episode_slices(done)),
            "action_counts": np.bincount(action[indices], minlength=num_actions).tolist()}
    report = {"random_seed": random_seed, "rf_parameters": rf.get_params(),
        "split_fraction_by_group": {"fit": 1-calibration_fraction-validation_fraction,
                                    "calibration": calibration_fraction, "validation": validation_fraction},
        "split": split_report, "rf_classes": rf.classes_.astype(int).tolist(),
        "absent_fit_actions": np.setdiff1d(np.arange(num_actions), rf.classes_).tolist(),
        "calibration": "sigmoid one-vs-rest with multiclass normalization; FrozenEstimator; no RF refit",
        "validation": {"raw": probability_metrics(action[val_rows], raw_prob, num_actions=num_actions),
                       "calibrated": probability_metrics(action[val_rows], calibrated_prob, num_actions=num_actions)},
        "validation_used_for_selection": selection == "validation", "final_refit": False,
        "policy_support_mask": False, "saved_probability_floor": None,
        "_partition_indices": partition}
    # The tie-break is declared before any test predictions: prefer the simpler
    # raw model when validation log loss is exactly equal.
    report["selection_rule"] = "minimum independent train-validation multiclass log loss; ties choose raw"
    report["selected_behavior"] = "calibrated" if report["validation"]["calibrated"]["log_loss"] < report["validation"]["raw"]["log_loss"] else "raw"

    if selection == 'calibrated':
        report['selection_rule'] = 'sigmoid-calibrated RF fixed before test evaluation; validation diagnostic only'
        report['selected_behavior'] = 'calibrated'
    report['num_actions'] = num_actions
    return rf, calibrated, report


# Validated algorithm registry

POLICIES = {'DDQN': DDQN, 'DQN': StandardDQN, 'BCQ': FixedLearningRateBCQ, 'CQL': CQL}
