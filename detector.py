"""Ensemble OOD detection and embedding data preparation."""
from __future__ import annotations
import numpy as np
import torch
from agent import BCQ, CQL
from metric import (
    embedding_quality,
    energy_scores,
    entropy_scores,
    evaluate_external_ood_scores,
    evaluate_ood_scores,
    report_clustered_masks,
    report_external_masks,
    report_synthetic_masks,
    variance_scores,
)
from plot import (
    configure_cluster_style,
    configure_synthetic_roc_style,
    plot_clustered_roc,
    plot_external_roc,
    plot_synthetic_roc,
)
from sklearn.cluster import KMeans
from util import ReplayBuffer, project_path


# Ensemble loading and scores

class BaseUncertiantyEstimationRL:
    def __init__(self):
        from configs.config import get_params
        self.params = get_params()
        self.policies = self.load_policy()

    def load_data(self):
        train_buffer = ReplayBuffer(
            state_dim=self.params['state_dim'],
            batch_size=self.params['batch_size'],
            target_data=self.params['target_data'],
            buffer_path=project_path(f"dataset/{self.params['target_data']}")
        ).load_data(only_test_set=False)

        test_buffer = ReplayBuffer(
            state_dim=self.params['state_dim'],
            batch_size=self.params['batch_size'],
            target_data=self.params['target_test_data'],
            buffer_path=project_path(f"dataset/{self.params['target_test_data']}")
        ).load_data(only_test_set=True)

        X_train = np.array(train_buffer.state)  # (N_train, state_dim)
        X_test  = np.array(test_buffer.state)   # (N_test, state_dim)
        
        return X_train, X_test
    
    def load_policy(self):
        """Load and return the ensemble policies (BCQ or CQL) specified by params."""
        checkpoints = [
            project_path(f"pth/{self.params['target_data']}_{self.params['algorithm']}_0.pth"),
            project_path(f"pth/{self.params['target_data']}_{self.params['algorithm']}_1.pth"),
            project_path(f"pth/{self.params['target_data']}_{self.params['algorithm']}_2.pth"),
            project_path(f"pth/{self.params['target_data']}_{self.params['algorithm']}_3.pth"),
            project_path(f"pth/{self.params['target_data']}_{self.params['algorithm']}_4.pth"),
        ]
        policies = []
        for ckpt in checkpoints:
            if self.params['algorithm'] == "CQL":
                policy = CQL(**self.params)
            else:
                policy = BCQ(**self.params)
            loaded = torch.load(ckpt, map_location='cpu')
            policy.Q.load_state_dict(loaded['model_state_dict'])
            policy.Q.to('cuda')
            policy.Q.eval()
            policies.append(policy)
        print("-policy load complete-")
        return policies
    
    def get_q_outputs(self, policies, states_np):
        """Compute Q-values for each policy on states_np and return a NumPy array with shape (num_policies, N, num_actions)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states_np).to('cuda')
            all_q = []
            for pol in policies:
                if isinstance(pol, CQL):
                    q = pol.Q(state_tensor)            # (N, num_actions)
                else:
                    q = pol.Q(state_tensor)[0]         # (N, num_actions)
                all_q.append(q.cpu().numpy())
            return np.stack(all_q, axis=0)             # (num_policies, N, num_actions)

    def compute_variance_scores(self, policies, states_np):
        return variance_scores(self.get_q_outputs(policies, states_np))

    def compute_entropy_scores(self, policies, states_np, temperature=1.0):
        return entropy_scores(self.get_q_outputs(policies, states_np), temperature=temperature)


    def compute_energy_scores(self, policies, states_np, temperature=1.0):
        return energy_scores(self.get_q_outputs(policies, states_np), temperature=temperature)
    

            
    def get_threshold(self, quantile_thr, t_entropy=1.0, t_energy=1.0):
        X_train, _ = self.load_data()

        # compute scores of each method
        var_train_scores = self.compute_variance_scores(self.policies, X_train)
        ent_train_scores = self.compute_entropy_scores(self.policies, X_train, temperature=t_entropy)
        eng_train_scores = self.compute_energy_scores(self.policies, X_train, temperature=t_energy)

        # get quantile-based throsholds
        thr_var = np.quantile(var_train_scores, quantile_thr)
        thr_ent = np.quantile(ent_train_scores, quantile_thr)
        thr_eng = np.quantile(eng_train_scores, quantile_thr)

        return thr_var, thr_ent, thr_eng


# Gaussian synthetic OOD

class SyntheticOOD(BaseUncertiantyEstimationRL):
    def __init__(self):
        super().__init__()

    def get_gaussian_ood_states(self, gaussian_noise):
        _, X_test = self.load_data()

        X_test_ood_gaussian = X_test + np.random.normal(0.3, gaussian_noise, X_test.shape)

        return X_test, X_test_ood_gaussian

    def predict_ood(self, quantile_thr=0.95, gaussian_noise=5.0):

        configure_synthetic_roc_style()
        # ==========================
        # 1) synthetic states
        # ==========================
        X_id, X_ood = self.get_gaussian_ood_states(
            gaussian_noise=gaussian_noise
        )

        # ==========================
        # 2) Compute scores.
        # ==========================
        var_test_scores = self.compute_variance_scores(self.policies, X_id)
        var_ood_scores  = self.compute_variance_scores(self.policies, X_ood)

        ent_test_scores = self.compute_entropy_scores(self.policies, X_id)
        ent_ood_scores  = self.compute_entropy_scores(self.policies, X_ood)

        eng_test_scores = self.compute_energy_scores(self.policies, X_id)
        eng_ood_scores  = self.compute_energy_scores(self.policies, X_ood)

        # ==========================
        # 3) Compute thresholds.
        # ==========================
        thr_var, thr_ent, thr_eng = self.get_threshold(quantile_thr=quantile_thr)

        # Predict OOD status for each method using Boolean masks.
        # variance: variance > thr_var indicates OOD.
        # entropy: entropy > thr_ent indicates OOD.
        # energy: energy > thr_eng indicates OOD.
        is_ood_var_test = var_test_scores > thr_var    # (N_test,)
        is_ood_ent_test = ent_test_scores > thr_ent
        is_ood_eng_test = eng_test_scores > thr_eng

        is_ood_var_ood  = var_ood_scores  > thr_var    # (N_test,)
        is_ood_ent_ood  = ent_ood_scores  > thr_ent
        is_ood_eng_ood  = eng_ood_scores  > thr_eng

        # ==========================
        # 4) Create the shared labels.
        # ==========================
        labels = np.concatenate(
            [
                np.zeros(len(X_id), dtype=int),  # ID
                np.ones(len(X_ood), dtype=int),   # OOD
            ],
            axis=0,
        )  # (2 * N_test,)

        # ==========================
        # 5) Compute metrics for each method.
        # ==========================

        # ==========================
        # 6) Report metrics for each method.
        # ==========================
        report_synthetic_masks(labels, is_ood_var_test, is_ood_var_ood, "Variance", quantile_thr)
        report_synthetic_masks(labels, is_ood_ent_test, is_ood_ent_ood, "Entropy", quantile_thr)
        report_synthetic_masks(labels, is_ood_eng_test, is_ood_eng_ood, "Energy", quantile_thr)

        # ==========================
        # 7) Combine predictions using union (OR).
        # ==========================
        combined_test_mask = (
            is_ood_var_test |
            is_ood_ent_test |
            is_ood_eng_test
        )
        combined_ood_mask = (
            is_ood_var_ood |
            is_ood_ent_ood |
            is_ood_eng_ood
        )
        report_synthetic_masks(labels, combined_test_mask, combined_ood_mask,
                        "Union (Variance ∪ Entropy ∪ Energy)", quantile_thr)

        # ==========================
        # 8) Majority vote: at least two of the three methods must predict OOD.
        # ==========================
        vote_test = (
            is_ood_var_test.astype(int)
            + is_ood_ent_test.astype(int)
            + is_ood_eng_test.astype(int)
        )
        vote_ood = (
            is_ood_var_ood.astype(int)
            + is_ood_ent_ood.astype(int)
            + is_ood_eng_ood.astype(int)
        )

        maj_test_mask = vote_test >= 2   # (N_test,)
        maj_ood_mask  = vote_ood  >= 2   # (N_test,)

        report_synthetic_masks(labels, maj_test_mask, maj_ood_mask,
                        "Majority vote (≥2 methods)", quantile_thr)

        # ==========================
        # 9) Sweep thresholds on continuous scores to compute ROC curves and AUC.
        # Plot a separate ROC curve for variance, entropy, and energy.
        # roc_curve internally sweeps all score thresholds.
        # ==========================
        # Keep the existing labels: ID = 0 and OOD = 1.
        var_scores_all = np.concatenate([var_test_scores, var_ood_scores], axis=0)
        ent_scores_all = np.concatenate([ent_test_scores, ent_ood_scores], axis=0)
        eng_scores_all = np.concatenate([eng_test_scores, eng_ood_scores], axis=0)

        roc_report = evaluate_ood_scores(labels, var_scores_all, ent_scores_all, eng_scores_all)
        auc_var = roc_report['var']["auc"]
        auc_ent = roc_report['ent']["auc"]
        auc_eng = roc_report['eng']["auc"]
        auc_union = roc_report['union']["auc"]
        auc_maj = roc_report['maj']["auc"]
        print("\n=== ROC AUC (continuous scores, threshold sweep) ===")
        print(f"Variance-based OOD AUC        : {auc_var:.4f}")
        print(f"Entropy-based  OOD AUC        : {auc_ent:.4f}")
        print(f"Energy-based   OOD AUC        : {auc_eng:.4f}")
        print(f"Union (max)    OOD AUC        : {auc_union:.4f}")
        print(f"Majority (2nd largest) OOD AUC: {auc_maj:.4f}")

        # ==========================
        # 10) ROC curve plot
        # ==========================
        plot_synthetic_roc(roc_report, project_path("plot/ROC.png"))


# External-cohort OOD

class ExternalOOD(BaseUncertiantyEstimationRL):
    def __init__(self):
        super().__init__()

    def get_external_ood_states(self):
        _, X_test_ood_external = self.load_data()

        test_buffer = ReplayBuffer(
            state_dim=self.params['state_dim'],
            batch_size=self.params['batch_size'],
            target_data=self.params['target_data'],
            buffer_path=project_path(f"dataset/{self.params['target_data']}")
        ).load_data(only_test_set=True)
        X_test = np.array(test_buffer.state)  # (N_train, state_dim)
        

        return X_test, X_test_ood_external

    def predict_ood(self, quantile_thr=0.95):
        # ==========================
        # 1) synthetic states
        # ==========================
        X_id, X_ood = self.get_external_ood_states()

        # ==========================
        # 2) Compute scores.
        # ==========================
        var_test_scores = self.compute_variance_scores(self.policies, X_id)
        var_ood_scores  = self.compute_variance_scores(self.policies, X_ood)

        ent_test_scores = self.compute_entropy_scores(self.policies, X_id)
        ent_ood_scores  = self.compute_entropy_scores(self.policies, X_ood)

        eng_test_scores = self.compute_energy_scores(self.policies, X_id)
        eng_ood_scores  = self.compute_energy_scores(self.policies, X_ood)

        # ==========================
        # 3) Compute thresholds.
        # ==========================
        thr_var, thr_ent, thr_eng = self.get_threshold(quantile_thr=quantile_thr)

        # Predict OOD status for each method using Boolean masks.
        # variance: variance > thr_var indicates OOD.
        # entropy: entropy > thr_ent indicates OOD.
        # energy: energy > thr_eng indicates OOD.
        is_ood_var_test = var_test_scores > thr_var    # (N_test,)
        is_ood_ent_test = ent_test_scores > thr_ent
        is_ood_eng_test = eng_test_scores > thr_eng

        is_ood_var_ood  = var_ood_scores  > thr_var    # (N_test,)
        is_ood_ent_ood  = ent_ood_scores  > thr_ent
        is_ood_eng_ood  = eng_ood_scores  > thr_eng

        # ==========================
        # 4) Create the shared labels.
        # ==========================
        labels = np.concatenate(
            [
                np.zeros(len(X_id), dtype=int),  # ID
                np.ones(len(X_ood), dtype=int),   # OOD
            ],
            axis=0,
        )  # (2 * N_test,)

        # ==========================
        # 5) Compute metrics for each method.
        # ==========================

        # ==========================
        # 6) Report metrics for each method.
        # ==========================
        report_external_masks(labels, is_ood_var_test, is_ood_var_ood, "Variance", quantile_thr)
        report_external_masks(labels, is_ood_ent_test, is_ood_ent_ood, "Entropy", quantile_thr)
        report_external_masks(labels, is_ood_eng_test, is_ood_eng_ood, "Energy", quantile_thr)

        # ==========================
        # 7) Combine predictions using union (OR).
        # ==========================
        combined_test_mask = (
            is_ood_var_test |
            is_ood_ent_test |
            is_ood_eng_test
        )
        combined_ood_mask = (
            is_ood_var_ood |
            is_ood_ent_ood |
            is_ood_eng_ood
        )
        report_external_masks(labels, combined_test_mask, combined_ood_mask,
                        "Union (Variance ∪ Entropy ∪ Energy)", quantile_thr)

        # ==========================
        # 8) Majority vote: at least two of the three methods must predict OOD.
        # ==========================
        vote_test = (
            is_ood_var_test.astype(int)
            + is_ood_ent_test.astype(int)
            + is_ood_eng_test.astype(int)
        )
        vote_ood = (
            is_ood_var_ood.astype(int)
            + is_ood_ent_ood.astype(int)
            + is_ood_eng_ood.astype(int)
        )

        maj_test_mask = vote_test >= 2   # (N_test,)
        maj_ood_mask  = vote_ood  >= 2   # (N_test,)

        report_external_masks(labels, maj_test_mask, maj_ood_mask,
                        "Majority vote (≥2 methods)", quantile_thr)

        # ==========================
        # 9) Sweep thresholds on continuous scores to compute ROC curves and AUC.
        # Plot a separate ROC curve for variance, entropy, and energy.
        # roc_curve internally sweeps all score thresholds.
        # ==========================
        # Keep the existing labels: ID = 0 and OOD = 1.
        var_scores_all = np.concatenate([var_test_scores, var_ood_scores], axis=0)
        ent_scores_all = np.concatenate([ent_test_scores, ent_ood_scores], axis=0)
        eng_scores_all = np.concatenate([eng_test_scores, eng_ood_scores], axis=0)

        roc_report = evaluate_external_ood_scores(labels, var_scores_all, ent_scores_all, eng_scores_all)
        auc_var = roc_report['var']["auc"]
        auc_ent = roc_report['ent']["auc"]
        auc_eng = roc_report['eng']["auc"]
        auc_union = roc_report['union']["auc"]
        auc_maj = roc_report['maj']["auc"]
        print("\n=== ROC AUC (continuous scores, threshold sweep) ===")
        print(f"Variance-based OOD AUC        : {auc_var:.4f}")
        print(f"Entropy-based  OOD AUC        : {auc_ent:.4f}")
        print(f"Energy-based   OOD AUC        : {auc_eng:.4f}")
        print(f"Union (max)    OOD AUC        : {auc_union:.4f}")
        print(f"Majority (2nd largest) OOD AUC: {auc_maj:.4f}")

        # ==========================
        # 10) ROC curve plot
        # ==========================
        plot_external_roc(roc_report, project_path("plot/ROC.png"))


# Cluster-selected external OOD

class ExternalKmeansOOD(BaseUncertiantyEstimationRL):
    def __init__(self):
        super().__init__()

    def get_external_kmeans_ood_states(
        self,
        font_tick=20,
        font_label=22,
        font_legend=20,
        font_colorbar=22,
        font_title=20
    ):
        # -------------------------
        # Control the default font settings through rcParams.
        # -------------------------
        configure_cluster_style(font_tick, font_label, font_legend, font_title)

        # -------------------------
        # 0) Load data.
        # -------------------------
        _, X_mimic4 = self.load_data()
        test_buffer = ReplayBuffer(
            state_dim=self.params['state_dim'],
            batch_size=self.params['batch_size'],
            target_data=self.params['target_data'],
            buffer_path=project_path(f"dataset/{self.params['target_data']}")
        ).load_data(only_test_set=True)
        X_mimic3 = np.array(test_buffer.state)

        n_clusters = 6 # sepsis : 6,. heparin: 5
        random_state = 42
        min_cluster_size = 30  # Minimum cluster size.

        # -------------------------
        # 1) Combine the data.
        # -------------------------
        X_all = np.vstack([X_mimic3, X_mimic4])
        n3 = X_mimic3.shape[0]

        # -------------------------
        # 2. K-means
        # -------------------------
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels_all = kmeans.fit_predict(X_all)
        labels3 = labels_all[:n3]
        labels4 = labels_all[n3:]

        # -------------------------
        # 2-1) Remove small clusters.
        # -------------------------
        counts_all = np.bincount(labels_all, minlength=n_clusters)
        small_clusters = np.where(counts_all < min_cluster_size)[0]
        valid_clusters = np.setdiff1d(np.arange(n_clusters), small_clusters)
        if valid_clusters.size == 0:
            valid_clusters = np.arange(n_clusters)
        valid_clusters_sorted = np.sort(valid_clusters)
        n_valid = len(valid_clusters_sorted)

        # -------------------------
        # 3) Compute proportions.
        # -------------------------
        def cluster_ratio(labels, valid_ids):
            mask = np.isin(labels, valid_ids)
            kept = labels[mask]
            cnt = np.bincount(kept, minlength=n_clusters).astype(float)
            total = cnt.sum()
            return cnt / total if total > 0 else np.zeros_like(cnt)

        p3_all = cluster_ratio(labels3, valid_clusters_sorted)
        p4_all = cluster_ratio(labels4, valid_clusters_sorted)
        p3 = p3_all[valid_clusters_sorted]
        p4 = p4_all[valid_clusters_sorted]

        # -------------------------
        # 4) Visualize with t-SNE and remap the cluster indices.
        # -------------------------
        # tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
        #             init="pca", random_state=random_state)
        # X_2d_all = tsne.fit_transform(X_all)

        mask_keep = np.isin(labels_all, valid_clusters_sorted)
        # X_2d_kept = X_2d_all[mask_keep]
        labels_kept = labels_all[mask_keep]

        # Map each original cluster ID to its new ID.
        orig_to_new = {c: i for i, c in enumerate(valid_clusters_sorted)}
        # labels_new = np.array([orig_to_new[c] for c in labels_kept])
        # n_valid = len(valid_clusters_sorted)

        # plt.figure(figsize=(7, 5))
        # cmap = plt.cm.get_cmap("tab10", n_valid)
        # scatter = plt.scatter(X_2d_kept[:, 0], X_2d_kept[:, 1],
        #                     c=labels_new, cmap=cmap, s=5, alpha=0.6,
        #                     vmin=-0.5, vmax=n_valid - 0.5)
        # cbar = plt.colorbar(scatter, ticks=range(n_valid))
        # cbar.ax.tick_params(labelsize=font_colorbar)
        # plt.tight_layout()
        # plt.savefig(project_path("./plot/external_kmeans_tsne_clusters.png"), dpi=600)
        # plt.close()

        # -------------------------
        # 4-2) Plot bars with consecutive indices.
        # -------------------------
        # idx = np.arange(n_valid)
        # plt.figure(figsize=(7, 5))
        # plt.bar(idx - 0.4/2, p3, width=0.4, alpha=0.7, label="MIMIC-III")
        # plt.bar(idx + 0.4/2, p4, width=0.4, alpha=0.7, label="MIMIC-IV")
        # plt.xlabel("Cluster idx")
        # plt.ylabel("Proportion")
        # plt.legend()
        # plt.xticks(idx, idx)
        # plt.tight_layout()
        # plt.savefig(project_path("./plot/external_kmeans_distribution_clusters.png"), dpi=600)
        # plt.close()

        # -------------------------
        # 5) Split ID and OOD samples using the remapped IDs.
        # -------------------------
        # For example, use new index 0 for ID and indices 1 and 2 for OOD.
        id_cluster_ids_new = [0, 1, 3] # heparin: 0,2,3 | sepsis: 0,1,3
        ood_cluster_ids_new = [4] # heparin: 1 | sepsis: 4

        # Apply the same remapping to labels4.
        mask_valid4 = np.isin(labels4, valid_clusters_sorted)
        labels4_valid = labels4[mask_valid4]
        labels4_new = np.array([orig_to_new[c] for c in labels4_valid])

        X_mimic4_valid = X_mimic4[mask_valid4]
        mask_id = np.isin(labels4_new, id_cluster_ids_new)
        mask_ood = np.isin(labels4_new, ood_cluster_ids_new)

        X_mimic4_id = X_mimic4_valid[mask_id]
        X_mimic4_ood = X_mimic4_valid[mask_ood]

        return X_mimic4_id, X_mimic4_ood



    def predict_ood(self, quantile_thr=0.95):
        configure_synthetic_roc_style()
        # ==========================
        # 1) synthetic states
        # ==========================
        X_id, X_ood = self.get_external_kmeans_ood_states()

        # ==========================
        # 2) Compute scores.
        # ==========================
        var_test_scores = self.compute_variance_scores(self.policies, X_id)
        var_ood_scores  = self.compute_variance_scores(self.policies, X_ood)

        ent_test_scores = self.compute_entropy_scores(self.policies, X_id)
        ent_ood_scores  = self.compute_entropy_scores(self.policies, X_ood)

        eng_test_scores = self.compute_energy_scores(self.policies, X_id)
        eng_ood_scores  = self.compute_energy_scores(self.policies, X_ood)

        # ==========================
        # 3) Compute thresholds.
        # ==========================
        thr_var, thr_ent, thr_eng = self.get_threshold(quantile_thr=quantile_thr)

        # Predict OOD status for each method using Boolean masks.
        # variance: variance > thr_var indicates OOD.
        # entropy: entropy > thr_ent indicates OOD.
        # energy: energy > thr_eng indicates OOD.
        is_ood_var_test = var_test_scores > thr_var    # (N_test,)
        is_ood_ent_test = ent_test_scores > thr_ent
        is_ood_eng_test = eng_test_scores > thr_eng

        is_ood_var_ood  = var_ood_scores  > thr_var    # (N_test,)
        is_ood_ent_ood  = ent_ood_scores  > thr_ent
        is_ood_eng_ood  = eng_ood_scores  > thr_eng

        # ==========================
        # 4) Create the shared labels.
        # ==========================
        labels = np.concatenate(
            [
                np.zeros(len(X_id), dtype=int),  # ID
                np.ones(len(X_ood), dtype=int),   # OOD
            ],
            axis=0,
        )  # (2 * N_test,)

        # ==========================
        # 5) Compute metrics for each method.
        # ==========================

        # ==========================
        # 6) Report metrics for each method.
        # ==========================
        report_clustered_masks(labels, is_ood_var_test, is_ood_var_ood, "Variance", quantile_thr)
        report_clustered_masks(labels, is_ood_ent_test, is_ood_ent_ood, "Entropy", quantile_thr)
        report_clustered_masks(labels, is_ood_eng_test, is_ood_eng_ood, "Energy", quantile_thr)

        # ==========================
        # 7) Combine predictions using union (OR).
        # ==========================
        combined_test_mask = (
            is_ood_var_test |
            is_ood_ent_test |
            is_ood_eng_test
        )
        combined_ood_mask = (
            is_ood_var_ood |
            is_ood_ent_ood |
            is_ood_eng_ood
        )
        report_clustered_masks(labels, combined_test_mask, combined_ood_mask,
                        "Union (Variance ∪ Entropy ∪ Energy)", quantile_thr)

        # ==========================
        # 8) Majority vote: at least two of the three methods must predict OOD.
        # ==========================
        vote_test = (
            is_ood_var_test.astype(int)
            + is_ood_ent_test.astype(int)
            + is_ood_eng_test.astype(int)
        )
        vote_ood = (
            is_ood_var_ood.astype(int)
            + is_ood_ent_ood.astype(int)
            + is_ood_eng_ood.astype(int)
        )

        maj_test_mask = vote_test >= 2   # (N_test,)
        maj_ood_mask  = vote_ood  >= 2   # (N_test,)

        report_clustered_masks(labels, maj_test_mask, maj_ood_mask,
                        "Majority vote (≥2 methods)", quantile_thr)

        # ==========================
        # 9) Sweep thresholds on continuous scores to compute ROC curves and AUC.
        # Plot a separate ROC curve for variance, entropy, and energy.
        # roc_curve internally sweeps all score thresholds.
        # ==========================
        # Keep the existing labels: ID = 0 and OOD = 1.
        var_scores_all = np.concatenate([var_test_scores, var_ood_scores], axis=0)
        ent_scores_all = np.concatenate([ent_test_scores, ent_ood_scores], axis=0)
        eng_scores_all = np.concatenate([eng_test_scores, eng_ood_scores], axis=0)

        roc_report = evaluate_ood_scores(labels, var_scores_all, ent_scores_all, eng_scores_all)
        auc_var = roc_report['var']["auc"]
        auc_ent = roc_report['ent']["auc"]
        auc_eng = roc_report['eng']["auc"]
        auc_union = roc_report['union']["auc"]
        auc_maj = roc_report['maj']["auc"]
        print("\n=== ROC AUC (continuous scores, threshold sweep) ===")
        print(f"Variance-based OOD AUC        : {auc_var:.4f}")
        print(f"Entropy-based  OOD AUC        : {auc_ent:.4f}")
        print(f"Energy-based   OOD AUC        : {auc_eng:.4f}")
        print(f"Union (max)    OOD AUC        : {auc_union:.4f}")
        print(f"Majority (2nd largest) OOD AUC: {auc_maj:.4f}")

        # ==========================
        # 10) ROC curve plot
        # ==========================
        plot_clustered_roc(roc_report, project_path("plot/ROC.png"))


# Embedding data preparation

def prepare_id_vs_ood_embedding(params, sigma=1., n_neighbors=30, umap_random_state=42,
                                id_kmeans_random_state=42, n_id_clusters=2):
    import umap
    # ---------------------------------------
    # 0) Load the ReplayBuffer with test data only: only_test_set=True.
    # ---------------------------------------
    train_buffer = ReplayBuffer(
        state_dim=params['state_dim'],
        batch_size=params['batch_size'],
        target_data=params['target_data'],
        buffer_path=project_path(f"dataset/{params['target_data']}")
    ).load_data(only_test_set=True)

    # 1) Original ID data, with shape (N, D).
    X_id = np.array(train_buffer.state)   # NumPy array
    N, D = X_id.shape

    # 2) Generate OOD data by adding Gaussian noise.
    noise_gauss = np.random.normal(loc=0.3, scale=sigma, size=(N, D))
    X_ood_gauss = X_id + noise_gauss

    # 3) Combine ID data and synthetic OOD data.
    X_all_gauss = np.vstack([X_id, X_ood_gauss])  # shape: (2N, D)

    # 3-1) Create ground-truth labels: 0 = ID and 1 = OOD.
    y_all_gauss = np.hstack([
        np.zeros(N, dtype=int),
        np.ones(N, dtype=int)
    ])  # shape: (2N,)

    # 4) Reduce the data to two dimensions using UMAP.
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=umap_random_state,
    )
    X_umap_gauss = umap_reducer.fit_transform(X_all_gauss)  # (2N, 2)

    # 5) Separate ID and OOD samples in the two-dimensional UMAP space.
    mask_id  = (y_all_gauss == 0)
    mask_ood = (y_all_gauss == 1)

    X_id_umap  = X_umap_gauss[mask_id]   # (N, 2)
    X_ood_umap = X_umap_gauss[mask_ood]  # (N, 2)

    # 5-1) Compute the center of all OOD samples.
    center_ood = X_ood_umap.mean(axis=0)  # (2,)

    # 5-2) Cluster ID samples into n_id_clusters groups using K-means in UMAP space.
    id_kmeans = KMeans(
        n_clusters=n_id_clusters,
        random_state=id_kmeans_random_state,
        n_init=10
    )
    id_cluster_labels = id_kmeans.fit_predict(X_id_umap)  # (N,)
    id_centers_2d = id_kmeans.cluster_centers_            # (n_id_clusters, 2)

    # 5-3) Compute the Davies-Bouldin index and silhouette score.
    labels_db = np.empty_like(y_all_gauss)
    labels_db[mask_id] = id_cluster_labels               # 0,1,...
    labels_db[mask_ood] = n_id_clusters                  # The last label identifies the OOD cluster.

    db_index, sil_score = embedding_quality(X_umap_gauss, labels_db)

    # ---------------------------------------
    # 6) Plot the results.
    # ---------------------------------------
    return dict(X_id_umap=X_id_umap, X_ood_umap=X_ood_umap, center_ood=center_ood,
                id_centers_2d=id_centers_2d, db_index=db_index, sil_score=sil_score)
