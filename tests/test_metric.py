"""Integration contracts for the corrected Sepsis wrappers; no clinical fits."""

import agent
import metric
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

import metric as metric
from agent import BCQ
from agent import CQL
from model import FQECritic


class TableQ(torch.nn.Module):
    def __init__(self, q, imitation=None):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(q, dtype=torch.float32))
        if imitation is not None:
            self.register_buffer("imitation", torch.tensor(imitation, dtype=torch.float32))

    def forward(self, state):
        rows = state[:, 0].long()
        q = self.values[rows]
        if hasattr(self, "imitation"):
            im = self.imitation[rows].log()
            return q, im, im
        return q


class TablePolicy:
    def __init__(self, q, imitation=None):
        self.Q = TableQ(q, imitation)
        self.device = "cpu"
        self.threshold = 0.3
        self.algorithm = "BCQ" if imitation is not None else "CQL"

    def action(self, state):
        # Compare against actual repository action methods, not test reimplementations.
        method = BCQ.action if self.algorithm == "BCQ" else CQL.action
        return method(self, state)


def buffer(state, reward, done, action=None, next_state=None):
    state = np.asarray(state, dtype=np.float32)
    if action is None:
        action = np.zeros(len(state), dtype=np.int64)
    if next_state is None:
        next_state = state.copy()
        for row in range(len(state) - 1):
            if not done[row]:
                next_state[row] = state[row + 1]
    return types.SimpleNamespace(
        state=state,
        next_state=np.asarray(next_state, dtype=np.float32),
        action=np.asarray(action).reshape(-1, 1),
        reward=np.asarray(reward, dtype=np.float32).reshape(-1, 1),
        done=np.asarray(done, dtype=np.float32).reshape(-1, 1),
        bc_prob=np.full((len(state), 1), 0.5),
        crt_size=len(state),
    )


class MetricIntegrationTests(unittest.TestCase):
    def test_frozen_greedy_matches_actual_checkpoint_action(self):
        policy = TablePolicy([[1, 4], [3, 0]])
        states = np.array([[0, 1], [1, 1]], dtype=np.float32)
        before = {k: v.clone() for k, v in policy.Q.state_dict().items()}
        expected = [policy.action(torch.tensor(row[None])) for row in states]
        frozen = agent.frozen_policy_arrays("CQL", policy, states, batch_size=1)
        np.testing.assert_array_equal(frozen.argmax(1), expected)
        np.testing.assert_array_equal(frozen, [[0, 1], [1, 0]])
        self.assertFalse(policy.Q.training)
        for name, value in policy.Q.state_dict().items():
            torch.testing.assert_close(value, before[name])
        softmax = agent.frozen_policy_arrays("CQL", policy, states, mode="softmax")
        self.assertTrue((softmax > 0).all())
        self.assertFalse(np.array_equal(frozen, softmax))

    def test_bcq_freeze_uses_actual_imitation_mask(self):
        policy = TablePolicy([[9, 1], [2, 8]], imitation=[[0.01, 0.99], [0.99, 0.01]])
        states = np.array([[0, 1], [1, 1]], dtype=np.float32)
        expected = [policy.action(torch.tensor(row[None])) for row in states]
        frozen = agent.frozen_policy_arrays("BCQ", policy, states, batch_size=1)
        np.testing.assert_array_equal(frozen.argmax(1), expected)
        np.testing.assert_array_equal(frozen, [[0, 1], [1, 0]])
        self.assertFalse(np.array_equal(frozen.argmax(1), [0, 1]))  # unmasked Q differs

    def test_fqe_requires_independent_fit_buffer_before_fitting(self):
        evaluation = buffer([[0, 1]], [1], [1])
        policy = TablePolicy([[1, 0]])
        with mock.patch.object(metric, "fit_fqe") as fitter:
            for fit_buffer in [None, evaluation]:
                with self.assertRaisesRegex(ValueError, "independent training buffer"):
                    metric.eval_fqe_ci("CQL", policy, evaluation, fit_buffer=fit_buffer)
            fitter.assert_not_called()

    def test_full_pipeline_rejects_identical_train_test_buffer(self):
        evaluation = buffer([[0, 1]], [1], [1])
        policy = TablePolicy([[1, 0]])
        with mock.patch("agent.fit_behavior", side_effect=AssertionError("must reject before fitting")):
            with self.assertRaisesRegex(ValueError, "independent|different|same"):
                metric.evaluate_sepsis_policy("CQL", policy, evaluation, evaluation)

    def test_scalar_behavior_and_missing_critic_are_rejected(self):
        evaluation = buffer([[0, 1]], [1], [1])
        policy = TablePolicy([[1, 0]])
        with self.assertRaisesRegex(ValueError, "Full.*behavior"):
            metric.eval_wis_ci("CQL", policy, evaluation, n_bootstrap=0)
        with self.assertRaisesRegex(ValueError, "shape"):
            metric.eval_wis_ci("CQL", policy, evaluation, behavior_probs=evaluation.bc_prob, n_bootstrap=0)
        with self.assertRaisesRegex(ValueError, "independently fitted"):
            metric.eval_multi_step_doubly_robust_ci("CQL", policy, evaluation, behavior_probs=[[1, 0]], n_bootstrap=0)

    def test_subject_overlap_rejected_before_nuisance_fitting(self):
        training = buffer([[0, 1]], [1], [1])
        evaluation = buffer([[1, 1]], [-1], [1])
        policy = TablePolicy([[1, 0], [1, 0]])
        with mock.patch("agent.fit_behavior") as fitter:
            with self.assertRaisesRegex(ValueError, "Test subjects remain"):
                metric.evaluate_sepsis_policy("CQL", policy, training, evaluation, train_groups=[7], test_groups=[7])
            fitter.assert_not_called()

    def test_one_frozen_actor_and_positive_gamma_reach_all_estimators(self):
        training = buffer([[0, 1], [1, 1], [2, 1], [3, 1]], [0, 1, 0, -1], [0, 1, 0, 1])
        evaluation = buffer([[4, 1], [5, 1], [6, 1]], [1, 0, -1], [1, 0, 1])
        policy = TablePolicy([[2, 1]] * 7)
        critic = FQECritic(2, 2, 8)  # no fit; controlled zero critic for wiring identity
        before = {k: v.clone() for k, v in policy.Q.state_dict().items()}
        classifier = object()
        behavior_report = {"selected_behavior": "raw", "_partition_indices": {}}
        full_probs = lambda model, states, **kwargs: np.tile([1.0, 0.0], (len(states), 1))
        with mock.patch("agent.fit_behavior", return_value=(classifier, classifier, behavior_report)) as behavior_fit, \
             mock.patch("model.full_action_proba", side_effect=full_probs), \
             mock.patch.object(metric, "fit_fqe", return_value=(critic, {"scope": "training only"})) as critic_fit:
            result = metric.evaluate_sepsis_policy(
                "CQL", policy, training, evaluation,
                train_groups=[1, 1, 2, 2], test_groups=[3, 4, 4],
                gamma=0.6, seed=19, n_bootstrap=20,
            )
        # On-policy returns are +1 and -gamma, so all three OPE means equal .2.
        for name in ["dr", "wdr", "wis"]:
            self.assertAlmostEqual(result[name], 0.2)
        self.assertEqual(result["episodes_used"], 2)
        self.assertEqual(result["episodes_excluded"], 0)
        self.assertEqual(result["fqe"]["value"], 0.0)
        self.assertTrue(np.shares_memory(behavior_fit.call_args.args[0], training.state))
        self.assertFalse(np.shares_memory(behavior_fit.call_args.args[0], evaluation.state))
        self.assertTrue(np.shares_memory(critic_fit.call_args.args[0], training.state))
        self.assertFalse(np.shares_memory(critic_fit.call_args.args[0], evaluation.state))
        np.testing.assert_array_equal(critic_fit.call_args.args[5], [[1, 0]] * 4)
        self.assertEqual(critic_fit.call_args.kwargs["gamma"], 0.6)
        self.assertEqual(critic_fit.call_args.kwargs["seed"], 19)
        np.testing.assert_array_equal(critic_fit.call_args.kwargs["groups"], [1, 1, 2, 2])
        for name, value in policy.Q.state_dict().items():
            torch.testing.assert_close(value, before[name])
        json.dumps(result, allow_nan=False)


class MainIntegrationTests(unittest.TestCase):
    def test_main_filters_subjects_before_updates_then_evaluates_one_final_checkpoint(self):
        original_training = buffer(
            [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1]],
            [0, 1, 0, 1, 0, -1], [0, 1, 0, 1, 0, 1],
        )
        original_test = buffer([[6, 1], [7, 1], [8, 1]], [1, 0, -1], [1, 0, 1])
        buffers = []
        training_calls = []

        class FakeBuffer:
            def __init__(self, **kwargs):
                self.device = kwargs["device"]
                buffers.append(self)

            def load_data(self, only_test_set=False):
                template = original_test if only_test_set else original_training
                for name in ["state", "next_state", "action", "reward", "done", "bc_prob"]:
                    setattr(self, name, getattr(template, name).copy())
                self.crt_size = template.crt_size
                return self

        class FakeAgent:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.Q = torch.nn.Linear(2, 2)
                # Patient filtering must precede even policy construction.
                if buffers[0].crt_size != 4:
                    raise AssertionError("Test subjects were not removed before policy construction")

            def train(self, supplied):
                training_calls.append(supplied.state.copy())
                np.testing.assert_array_equal(supplied.state, original_training.state[2:])
                self.asserted_device = supplied.device

        def evaluate_once(algorithm, policy, training, evaluation, **kwargs):
            self.assertEqual(len(training_calls), 2)
            self.assertEqual(training.crt_size, 4)
            self.assertEqual(evaluation.crt_size, 3)
            np.testing.assert_array_equal(kwargs["train_groups"], [1, 1, 2, 2])
            np.testing.assert_array_equal(kwargs["test_groups"], [3, 4, 4])
            self.assertEqual(kwargs["gamma"], 0.63)
            self.assertEqual(kwargs["seed"], 42)
            self.assertEqual(kwargs["policy_mode"], "greedy")
            self.assertEqual(kwargs["fqe_epochs"], 30)
            return {
                "dr": 0.2, "wdr": 0.2, "wis": 0.2,
                "bootstrap": {"intervals": {name: {"low": 0.0, "high": 0.4} for name in ["dr", "wdr", "wis"]}},
                "fqe": {"value": 0.2, "low": 0.0, "high": 0.4},
                "weights": {"trajectory_ess": 2.0},
            }

        mlflow_stub = types.ModuleType("mlflow")
        for name in ["log_param", "log_metric", "log_artifact"]:
            setattr(mlflow_stub, name, mock.Mock())
        config_stub = types.ModuleType("configs.config")
        config_stub.get_params = mock.Mock()  # main.train receives explicit params below.
        source = Path(__file__).resolve().parents[1] / "scripts/train_policy.py"
        spec = importlib.util.spec_from_file_location("rl_main_test_only", source)
        main = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"mlflow": mlflow_stub, "configs.config": config_stub}):
            spec.loader.exec_module(main)
        keep = np.array([False, False, True, True, True, True])
        groups = (np.array([3, 3, 1, 1, 2, 2]), np.array([3, 4, 4]), keep, {"excluded_train_episodes": 1})
        params = {
            "device": "auto", "target_data": "sepsis", "state_dim": 2,
            "batch_size": 2, "algorithm": "CQL", "max_timesteps": 2,
            "eval_freq": 1, "discount": 0.63, "seed": 42,
            "ope_fqe_epochs": 30, "ope_policy_mode": "greedy",
        }
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                params["project_root"] = directory
                data = Path("dataset/sepsis")
                data.mkdir(parents=True)
                for split, template in [("train", original_training), ("test", original_test)]:
                    for name in ["state", "next_state", "action", "reward", "done"]:
                        np.save(data / f"{split}_{name}.npy", getattr(template, name))
                with mock.patch.object(main, "ReplayBuffer", FakeBuffer), \
                     mock.patch.object(main, "CQL", FakeAgent), \
                     mock.patch.object(main, "evaluate_sepsis_policy", side_effect=evaluate_once) as evaluator, \
                     mock.patch("util.recover_subject_groups", return_value=groups), \
                     mock.patch.object(torch.cuda, "is_available", return_value=False):
                    policy = main.train(params)
                self.assertEqual(evaluator.call_count, 1)
                self.assertEqual(len(training_calls), 2)
                self.assertEqual(policy.params["device"], "cpu")
                checkpoints = list(Path("outputs").glob("*/policy_final.pth"))
                self.assertEqual(len(checkpoints), 1)
                checkpoint = torch.load(checkpoints[0], weights_only=True, map_location="cpu")
                self.assertEqual(checkpoint["training_step"], 2)
                self.assertEqual(checkpoint["seed"], 42)
                self.assertEqual(checkpoint["params"]["discount"], 0.63)
                self.assertEqual(len(checkpoint["dataset_sha256"]), 10)
                evaluation = json.loads((checkpoints[0].parent / "evaluation.json").read_text())
                self.assertEqual(evaluation["subject_mapping"]["excluded_train_episodes"], 1)
                self.assertEqual(len(evaluation["policy_checkpoint_sha256"]), 64)
                for call in mlflow_stub.log_metric.call_args_list:
                    self.assertEqual(call.kwargs["step"], 2)
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
