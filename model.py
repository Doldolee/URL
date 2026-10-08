"""Q networks, bounded FQE critics and serializable behavior predictors."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from util import NUM_ACTIONS


# Q networks

class BCQNet(nn.Module):
	def __init__(self, state_dim, num_actions, hidden_node, activation='relu'):
		super(BCQNet, self).__init__()
		self.share_q = nn.Linear(state_dim, hidden_node)
		self.share_bn = nn.BatchNorm1d(num_features=hidden_node)
		
		self.value_q1 = nn.Linear(hidden_node, hidden_node)
		self.value_bn1 = nn.BatchNorm1d(num_features=hidden_node)
		self.value_q2 = nn.Linear(hidden_node, 1)
		self.value_bn2 = nn.BatchNorm1d(num_features=1)
		
		self.adv_q1 = nn.Linear(hidden_node, hidden_node)
		self.adv_bn1 = nn.BatchNorm1d(num_features=hidden_node)
		self.adv_q2 = nn.Linear(hidden_node, num_actions)
		self.adv_bn2 = nn.BatchNorm1d(num_features=num_actions)
		
		self.i1 = nn.Linear(state_dim, hidden_node)
		self.i2 = nn.Linear(hidden_node, hidden_node)
		self.i3 = nn.Linear(hidden_node, num_actions)
		
		if activation.lower() == 'relu':
			self.activation = F.relu
		elif activation.lower() == 'tanh':
			self.activation = F.tanh
		else:
			raise ValueError("Unsupported activation function")

	def forward(self, state):
		x = self.activation(self.share_bn(self.share_q(state)))
		
		value = self.activation(self.value_bn1(self.value_q1(x)))
		value = self.activation(self.value_bn2(self.value_q2(value)))
		
		adv = self.activation(self.adv_bn1(self.adv_q1(x)))
		adv = self.activation(self.adv_bn2(self.adv_q2(adv)))
		
		adv_average = torch.mean(adv, dim=1, keepdim=True)
		q = value + adv - adv_average

		i = self.activation(self.i1(state))
		i = self.activation(self.i2(i))
		i = self.i3(i)
		return q, F.log_softmax(i, dim=1), i


class CQLNet(torch.nn.Module):
    """Dueling Q-network with Batch Normalization for discrete actions"""
    def __init__(self, state_dim, num_actions, hidden_node, activation='relu'):
        super(CQLNet, self).__init__()
        # Shared layers
        self.fc1 = torch.nn.Linear(state_dim, hidden_node)
        self.bn1 = torch.nn.BatchNorm1d(hidden_node)
        self.fc2 = torch.nn.Linear(hidden_node, hidden_node)
        self.bn2 = torch.nn.BatchNorm1d(hidden_node)

        # Value stream
        self.value_fc = torch.nn.Linear(hidden_node, hidden_node)
        self.value_bn = torch.nn.BatchNorm1d(hidden_node)
        self.value_out = torch.nn.Linear(hidden_node, 1)

        # Advantage stream
        self.adv_fc = torch.nn.Linear(hidden_node, hidden_node)
        self.adv_bn = torch.nn.BatchNorm1d(hidden_node)
        self.adv_out = torch.nn.Linear(hidden_node, num_actions)

        # Activation
        if activation.lower() == 'relu':
            self.act = F.relu
        elif activation.lower() == 'tanh':
            self.act = torch.tanh
        else:
            raise ValueError("Unsupported activation")

    def forward(self, state):
        # Shared
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)

        # Value
        v = self.value_fc(x)
        v = self.value_bn(v)
        v = self.act(v)
        v = self.value_out(v)

        # Advantage
        a = self.adv_fc(x)
        a = self.adv_bn(a)
        a = self.act(a)
        a = self.adv_out(a)

        # Combine streams
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class DQNNet(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_node, activation='relu'):
        super(DQNNet, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_node)
        self.bn1 = nn.BatchNorm1d(hidden_node)
        self.fc2 = nn.Linear(hidden_node, hidden_node)
        self.bn2 = nn.BatchNorm1d(hidden_node)
        # The output layer directly returns Q-values.
        self.fc3 = nn.Linear(hidden_node, num_actions)

        # Configure the activation function.
        act = activation.lower()
        if act == 'relu':
            self.act = F.relu
        elif act == 'tanh':
            self.act = torch.tanh
        else:
            raise ValueError("Unsupported activation: choose 'relu' or 'tanh'")

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)

        # Return the Q-value for each action directly from the final layer.
        q = self.fc3(x)
        return q


# Bounded auxiliary critics

class FQECritic(nn.Module):
    """Small MLP returning physical-scale action values in [-1, 1]."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        if min(state_dim, num_actions, hidden_dim) < 1:
            raise ValueError("Network dimensions must be positive.")
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Tanh(),
        )
        # Begin at zero return rather than inheriting large CQL action values.
        nn.init.zeros_(self.network[-2].weight)
        nn.init.zeros_(self.network[-2].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class HeparinFQECritic(FQECritic):
    def __init__(self, state_dim, num_actions, hidden_dim=128, value_bound=50.):
        super().__init__(state_dim, num_actions, hidden_dim)
        self.value_bound = float(value_bound)

    def forward(self, state):
        # Keep unit output gradient near zero while retaining the return bound.
        logits = self.network[:-1](state)
        return self.value_bound * torch.tanh(logits / self.value_bound)


# Serializable behavior predictors

def _classes(estimator):
    raw = np.asarray(estimator.classes_)
    if raw.ndim != 1 or not np.equal(raw, raw.astype(np.int64)).all():
        raise ValueError("Behavior classes must be integer action indices")
    classes = raw.astype(np.int64)
    if not np.array_equal(np.sort(classes), np.arange(NUM_ACTIONS)):
        raise ValueError("Behavior estimator must contain all 25 action classes")
    return classes


class FullActionPredictor:
    """Serializable sklearn-style predictor with explicit columns 0 through 24."""

    def __init__(self, estimator):
        _classes(estimator)
        self.estimator = estimator
        self.classes_ = np.arange(NUM_ACTIONS, dtype=np.int64)
        self.params = {}

    def predict_proba(self, state):
        return full_action_proba(self.estimator, state)


class TemperatureScaledPredictor:
    """Apply one scalar temperature to logistic logits, then map class columns."""

    def __init__(self, estimator, temperature):
        self.source_classes_ = _classes(estimator)
        self.estimator = estimator
        self.temperature = float(temperature)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.classes_ = np.arange(NUM_ACTIONS, dtype=np.int64)
        self.params = {"temperature": self.temperature}

    def predict_proba(self, state):
        from scipy.special import logsumexp

        scores = np.asarray(self.estimator.decision_function(state), dtype=np.float64)
        if scores.shape != (len(state), NUM_ACTIONS) or not np.isfinite(scores).all():
            raise ValueError("Expected finite logits for all 25 behavior actions")
        scaled = scores / self.temperature
        probability = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
        result = np.zeros_like(probability)
        result[:, self.source_classes_] = probability
        return result


def full_action_proba(model, state, num_actions=NUM_ACTIONS):
    """Map classes to the explicit action space without smoothing zero support."""
    classes = np.asarray(model.classes_, dtype=np.int64)
    if len(np.unique(classes)) != len(classes) or not ((classes >= 0) & (classes < num_actions)).all():
        raise ValueError("invalid behavior model class labels")
    compact = np.asarray(model.predict_proba(state), dtype=np.float64)
    result = np.zeros((len(state), num_actions), dtype=np.float64)
    result[:, classes] = compact
    if not np.isfinite(result).all() or (result < 0).any() or not np.allclose(result.sum(axis=1), 1, atol=1e-8):
        raise ValueError("invalid behavior probabilities")
    return result


class BehaviorLogitPredictor:
    """Generic, serializable multinomial predictor with aligned action columns.

    This wrapper is separate from the historical 25-action behavior interface.
    Temperature scaling changes probabilities only; it never inserts a floor.
    """

    def __init__(self, estimator, num_actions, temperature=1.):
        self.estimator = estimator
        self.num_actions = int(num_actions)
        source = np.asarray(estimator.classes_)
        if (source.ndim != 1 or not np.array_equal(source, source.astype(np.int64))
                or not np.array_equal(np.sort(source), np.arange(self.num_actions))):
            raise ValueError("Behavior estimator must contain all action classes")
        self.source_classes_ = source.astype(np.int64)
        self.classes_ = np.arange(self.num_actions, dtype=np.int64)
        self.temperature = float(temperature)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.params = {}

    def decision_function(self, state):
        scores = np.asarray(self.estimator.decision_function(state), dtype=np.float64)
        if self.num_actions == 2 and scores.ndim == 1:
            scores = np.column_stack([np.zeros(len(scores)), scores])
        if scores.shape != (len(state), self.num_actions) or not np.isfinite(scores).all():
            raise ValueError("Expected finite logits for all behavior actions")
        aligned = np.empty_like(scores)
        aligned[:, self.source_classes_] = scores / self.temperature
        return aligned

    def predict_proba(self, state):
        from scipy.special import logsumexp
        scores = self.decision_function(state)
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


class NumpyMLPBehaviorPredictor:
    """Frozen CPU MLP logits stored as arrays, without a device dependency."""

    def __init__(self, scaler, weights, biases, num_actions, batch_size=1024):
        self.scaler = scaler
        self.weights = [np.asarray(w, dtype=np.float32).copy() for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32).copy() for b in biases]
        self.num_actions = int(num_actions)
        self.classes_ = np.arange(self.num_actions, dtype=np.int64)
        self.batch_size = int(batch_size)

    def decision_function(self, state):
        state = np.asarray(state)
        result = np.empty((len(state), self.num_actions), dtype=np.float64)
        # Match training's CPU float32 linear operations. The stored arrays
        # remain independent of torch module/device serialization conventions.
        weights = [torch.from_numpy(weight) for weight in self.weights]
        biases = [torch.from_numpy(bias) for bias in self.biases]
        with torch.no_grad():
            for begin in range(0, len(state), self.batch_size):
                values = torch.from_numpy(self.scaler.transform(
                    state[begin:begin+self.batch_size]).astype(np.float32))
                for index, (weight, bias) in enumerate(zip(weights, biases)):
                    values = F.linear(values, weight, bias)
                    if index < len(self.weights)-1:
                        values = F.relu(values)
                result[begin:begin+len(values)] = values.numpy()
        if not np.isfinite(result).all():
            raise ValueError("MLP produced non-finite logits")
        return result

    def predict_proba(self, state):
        from scipy.special import logsumexp
        scores = self.decision_function(state)
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


class NeighborCountBehaviorPredictor:
    """Empirical neighbor actions with a Dirichlet fit-action prior."""

    def __init__(self, scaler, neighbors, fit_actions, prior, n_neighbors,
                 concentration=1., batch_size=256, query_neighbors=None):
        self.scaler = scaler
        self.neighbors = neighbors
        self.fit_actions = np.asarray(fit_actions, dtype=np.int64).copy()
        self.prior = np.asarray(prior, dtype=np.float64).copy()
        self.num_actions = len(self.prior)
        self.classes_ = np.arange(self.num_actions, dtype=np.int64)
        self.n_neighbors = min(int(n_neighbors), len(self.fit_actions))
        # Using the same max-k query at calibration and later inference also
        # gives the same prefix when equal-distance neighbors are tied.
        self.query_neighbors = min(int(query_neighbors or n_neighbors), len(self.fit_actions))
        self.concentration = float(concentration)
        self.batch_size = int(batch_size)
        self.params = {}
        if (self.n_neighbors < 1 or self.query_neighbors < self.n_neighbors
                or not np.isfinite(self.concentration)
                or self.concentration <= 0 or (self.prior <= 0).any()
                or not np.isclose(self.prior.sum(), 1.)):
            raise ValueError("Neighbor smoothing requires positive fit prior and concentration")

    def action_counts(self, state):
        state = np.asarray(state)
        counts = np.zeros((len(state), self.num_actions), dtype=np.float64)
        for begin in range(0, len(state), self.batch_size):
            values = self.scaler.transform(state[begin:begin+self.batch_size])
            index = self.neighbors.kneighbors(values, n_neighbors=self.query_neighbors,
                                             return_distance=False)[:, :self.n_neighbors]
            actions = self.fit_actions[index]
            local = counts[begin:begin+len(values)]
            np.add.at(local, (np.repeat(np.arange(len(values)), self.n_neighbors),
                             actions.reshape(-1)), 1.)
        return counts

    def probabilities_from_counts(self, counts):
        counts = np.asarray(counts, dtype=np.float64)
        return ((counts + self.concentration*self.prior)
                / (counts.sum(axis=1, keepdims=True) + self.concentration))

    def predict_proba(self, state):
        return self.probabilities_from_counts(self.action_counts(state))

    def decision_function(self, state):
        return np.log(self.predict_proba(state))


class ClusterCountBehaviorPredictor:
    """Cluster-level clinician action counts with a Dirichlet fit prior."""

    def __init__(self, scaler, clustering, cluster_action_counts, prior,
                 concentration=1., batch_size=512):
        self.scaler = scaler
        self.clustering = clustering
        self.cluster_action_counts = np.asarray(cluster_action_counts, dtype=np.float64).copy()
        self.prior = np.asarray(prior, dtype=np.float64).copy()
        self.num_actions = len(self.prior)
        self.classes_ = np.arange(self.num_actions, dtype=np.int64)
        self.concentration = float(concentration)
        self.batch_size = int(batch_size)
        self.params = {}
        if (not np.isfinite(self.concentration) or self.concentration <= 0
                or (self.prior <= 0).any() or not np.isclose(self.prior.sum(), 1.)
                or self.cluster_action_counts.shape[1] != self.num_actions):
            raise ValueError("Cluster smoothing requires positive fit prior and concentration")

    def action_counts(self, state):
        state = np.asarray(state)
        counts = np.empty((len(state), self.num_actions), dtype=np.float64)
        for begin in range(0, len(state), self.batch_size):
            values = self.scaler.transform(state[begin:begin+self.batch_size])
            labels = self.clustering.predict(values)
            counts[begin:begin+len(values)] = self.cluster_action_counts[labels]
        return counts

    def probabilities_from_counts(self, counts):
        counts = np.asarray(counts, dtype=np.float64)
        return ((counts + self.concentration*self.prior)
                / (counts.sum(axis=1, keepdims=True) + self.concentration))

    def predict_proba(self, state):
        return self.probabilities_from_counts(self.action_counts(state))

    def decision_function(self, state):
        return np.log(self.predict_proba(state))
