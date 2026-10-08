"""Regression coverage for relocated artifacts, OOD scores and drawing boundaries."""
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from metric import evaluate_ood_scores, evaluate_external_ood_scores
from metric import variance_scores, entropy_scores, energy_scores
from model import FullActionPredictor
from plot import plot_synthetic_roc, plot_external_roc, plot_clustered_roc
from plot import draw_variance_kde
from plot import plot_and_save_cm
from plot import plot_id_vs_ood_embedding
from util import load_behavior_model
from util import project_root


class Classifier:
    classes_ = np.arange(25)
    def predict_proba(self, x):
        return np.tile(np.arange(1,26)/325., (len(x),1))


class ModuleLayoutTests(unittest.TestCase):
    def test_historical_pickle_load_restores_existing_namespace_and_exception(self):
        name = "RL_behavior_retrain"
        prior = types.ModuleType(name)
        class HistoricalPredictor(FullActionPredictor):
            pass
        HistoricalPredictor.__name__ = "FullActionPredictor"
        HistoricalPredictor.__qualname__ = "FullActionPredictor"
        HistoricalPredictor.__module__ = name
        prior.FullActionPredictor = HistoricalPredictor
        x = np.zeros((3,2))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {name: prior}):
            file = Path(directory)/"old.joblib"
            joblib.dump(HistoricalPredictor(Classifier()), file, compress=3)
            model = load_behavior_model(file)
            self.assertIs(type(model), FullActionPredictor)
            self.assertIs(sys.modules[name], prior)
            np.testing.assert_array_equal(model.predict_proba(x), Classifier().predict_proba(x))
            with self.assertRaises(FileNotFoundError):
                load_behavior_model(Path(directory)/"missing.joblib")
            self.assertIs(sys.modules[name], prior)
        self.assertNotIn(name, sys.modules)

    def test_ensemble_scores_match_independent_definitions(self):
        q = np.random.default_rng(7).normal(size=(5,13,6))
        temperature = .7
        probabilities = np.exp(q/temperature-(q/temperature).max(axis=2,keepdims=True))
        probabilities /= probabilities.sum(axis=2,keepdims=True)
        expected_entropy = (-(probabilities*np.log(probabilities)).sum(2)).var(0)
        expected_energy = (-temperature*np.log(np.exp(q/temperature).sum(2))).var(0)
        np.testing.assert_array_equal(variance_scores(q),q.max(2).var(0))
        np.testing.assert_allclose(entropy_scores(q,temperature),expected_entropy,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(energy_scores(q,temperature),expected_energy,rtol=1e-14,atol=1e-14)

    def test_continuous_union_majority_preserve_both_normalization_protocols(self):
        y = np.r_[np.zeros(7),np.ones(7)]
        scores = np.array([[1,7,3,2,4,9,0,5,12,3,6,8,4,11],
                           [80,22,110,10,9,30,52,11,10,130,92,51,31,88],
                           [-5,5,1,2,9,11,-1,4,6,11,2,13,15,10]],dtype=float)
        z = (scores-scores.mean(1,keepdims=True))/(scores.std(1,keepdims=True)+1e-8)
        for fn,values in [(evaluate_ood_scores,z),(evaluate_external_ood_scores,scores)]:
            report = fn(y,*scores)
            self.assertEqual(report["union"]["auc"],roc_auc_score(y,values.max(0)))
            self.assertEqual(report["maj"]["auc"],roc_auc_score(y,np.sort(values,axis=0)[1]))
            for name,score in zip(["var","ent","eng"],scores):
                self.assertEqual(report[name]["auc"],roc_auc_score(y,score))
            self.assertGreater(len(report["union"]["fpr"]),3)

    def test_plotters_accept_precomputed_inputs_without_models_or_datasets(self):
        rng = np.random.default_rng(4)
        values = [rng.random(16) for _ in range(3)]
        report = evaluate_ood_scores(np.r_[np.zeros(8),np.ones(8)],*values)
        data = dict(X_id_umap=rng.normal(size=(8,2)),X_ood_umap=rng.normal(size=(8,2)),
                    center_ood=np.ones(2),id_centers_2d=np.zeros((2,2)),db_index=.9,sil_score=.4)
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            for i,fn in enumerate([plot_synthetic_roc,plot_external_roc,plot_clustered_roc]):
                fn(report,directory/(str(i)+".png"))
                plt.close("all")
            draw_variance_kde(rng.random(32),rng.random(32),.95,directory/"kde.png")
            plot_and_save_cm(np.array([[7,1],[2,8]]),filename=directory/"cm.png")
            plot_id_vs_ood_embedding(data,directory/"embedding.png")
            self.assertEqual(len(list(directory.glob("*.png"))),6)
            self.assertTrue(all(p.stat().st_size>1000 for p in directory.glob("*.png")))

    def test_project_root_does_not_depend_on_working_directory(self):
        with patch("pathlib.Path.cwd",return_value=Path("/")):
            self.assertEqual(project_root(),Path(__file__).resolve().parents[1])


    def test_all_detector_orchestrators_use_their_roc_protocol_without_cuda(self):
        from detector import SyntheticOOD
        from detector import ExternalOOD
        from detector import ExternalKmeansOOD
        import contextlib
        rng = np.random.default_rng(18)
        states = [rng.normal(size=(11,4)), rng.normal(size=(9,4))]
        q = [rng.normal(size=(5,11,6)), rng.normal(size=(5,9,6))]
        for cls, module, draw in [(SyntheticOOD,"detector","plot_synthetic_roc"),
                                 (ExternalOOD,"detector","plot_external_roc"),
                                 (ExternalKmeansOOD,"detector","plot_clustered_roc")]:
            detector = cls.__new__(cls)
            detector.policies = []
            detector.get_q_outputs = lambda policies, x: q[0] if x is states[0] else q[1]
            detector.get_threshold = lambda **kw: (.1,.01,.1)
            detector.get_gaussian_ood_states = lambda **kw: states
            detector.get_external_ood_states = lambda: states
            detector.get_external_kmeans_ood_states = lambda: states
            with patch(module+"."+draw) as renderer, contextlib.redirect_stdout(io.StringIO()):
                detector.predict_ood()
            self.assertEqual(renderer.call_count,1)
            report = renderer.call_args.args[0]
            self.assertEqual(set(report),{"var","ent","eng","union","maj"})
            self.assertGreater(len(report["union"]["fpr"]),3)

    def test_source_manifest_rejects_changed_notebook_interface_and_runtime(self):
        import json
        from util import source_hashes, sha256_file, verify_source_manifest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'util.py';source.write_text('x = 1\n')
            notebook=root/'preprocess.ipynb';notebook.write_text('{"cells": []}')
            manifest=dict(sources=source_hashes(root),interfaces={notebook.name:sha256_file(notebook)})
            (root/'source_manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_source_manifest(root),manifest)
            notebook.write_text('{"cells": [1]}')
            with self.assertRaisesRegex(ValueError,'interface manifest mismatch'):
                verify_source_manifest(root)
            notebook.write_text('{"cells": []}')
            source.write_text('x = 2\n')
            with self.assertRaisesRegex(ValueError,'source manifest mismatch'):
                verify_source_manifest(root)


    def test_previous_package_behavior_pickle_loads_without_leaving_aliases(self):
        from util import _historical_behavior_namespace
        import model
        x=np.zeros((2,3))
        predictor=FullActionPredictor(Classifier())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"package.joblib"
            old_module=FullActionPredictor.__module__
            try:
                with _historical_behavior_namespace():
                    FullActionPredictor.__module__="models.behavior"
                    joblib.dump(predictor,path,compress=3)
            finally:
                FullActionPredictor.__module__=old_module
            loaded=load_behavior_model(path)
            self.assertIs(type(loaded),model.FullActionPredictor)
            np.testing.assert_array_equal(loaded.predict_proba(x),predictor.predict_proba(x))
        for name in ["models", "models.behavior", "RL_behavior_retrain"]:
            self.assertNotIn(name,sys.modules)
