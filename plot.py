"""ROC, distribution, confusion matrix and embedding figures."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from util import project_path


# ROC figures

def plot_synthetic_roc(report, path):
    fpr_var, tpr_var, auc_var = report['var']["fpr"], report['var']["tpr"], report['var']["auc"]
    fpr_ent, tpr_ent, auc_ent = report['ent']["fpr"], report['ent']["tpr"], report['ent']["auc"]
    fpr_eng, tpr_eng, auc_eng = report['eng']["fpr"], report['eng']["tpr"], report['eng']["auc"]
    fpr_union, tpr_union, auc_union = report['union']["fpr"], report['union']["tpr"], report['union']["auc"]
    fpr_maj, tpr_maj, auc_maj = report['maj']["fpr"], report['maj']["tpr"], report['maj']["auc"]
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_var,   tpr_var,   label=f"Q-value  (AUC={auc_var:.3f})", linewidth=2.5)
    plt.plot(fpr_ent,   tpr_ent,   label=f"Entropy  (AUC={auc_ent:.3f})", linewidth=2.5)
    plt.plot(fpr_eng,   tpr_eng,   label=f"Energy   (AUC={auc_eng:.3f})", linewidth=2.5)
    plt.plot(fpr_maj,   tpr_maj,   label=f"Majority (AUC={auc_maj:.3f})", linewidth=2.5)
    plt.plot(fpr_union, tpr_union, label=f"Union     (AUC={auc_union:.3f})", linewidth=2.5)


    # random guess baseline
    plt.plot([0, 1], [0, 1], "--", label="Random")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=600)


def configure_synthetic_roc_style():

    # Set the font sizes together.
    plt.rcParams.update({
        "axes.labelsize": 22,   # xlabel, ylabel
        "xtick.labelsize": 20,  # X-axis tick labels.
        "ytick.labelsize": 20,  # Y-axis tick labels.
        "legend.fontsize": 20   # Legend font size.
    })


def plot_external_roc(report, path):
    fpr_var, tpr_var, auc_var = report['var']["fpr"], report['var']["tpr"], report['var']["auc"]
    fpr_ent, tpr_ent, auc_ent = report['ent']["fpr"], report['ent']["tpr"], report['ent']["auc"]
    fpr_eng, tpr_eng, auc_eng = report['eng']["fpr"], report['eng']["tpr"], report['eng']["auc"]
    fpr_union, tpr_union, auc_union = report['union']["fpr"], report['union']["tpr"], report['union']["auc"]
    fpr_maj, tpr_maj, auc_maj = report['maj']["fpr"], report['maj']["tpr"], report['maj']["auc"]
    plt.figure(figsize=(7, 7))
    plt.plot(fpr_var,   tpr_var,   label=f"Variance (AUC={auc_var:.3f})")
    plt.plot(fpr_ent,   tpr_ent,   label=f"Entropy  (AUC={auc_ent:.3f})")
    plt.plot(fpr_eng,   tpr_eng,   label=f"Energy   (AUC={auc_eng:.3f})")
    plt.plot(fpr_union, tpr_union, label=f"Union-max (AUC={auc_union:.3f})")
    plt.plot(fpr_maj,   tpr_maj,   label=f"Majority-2nd (AUC={auc_maj:.3f})")

    # random guess baseline
    plt.plot([0, 1], [0, 1], "--", label="Random")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ID vs Gaussian OOD ROC (Var / Ent / Eng / Union / Majority)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=600)


def plot_clustered_roc(report, path):
    fpr_var, tpr_var, auc_var = report['var']["fpr"], report['var']["tpr"], report['var']["auc"]
    fpr_ent, tpr_ent, auc_ent = report['ent']["fpr"], report['ent']["tpr"], report['ent']["auc"]
    fpr_eng, tpr_eng, auc_eng = report['eng']["fpr"], report['eng']["tpr"], report['eng']["auc"]
    fpr_union, tpr_union, auc_union = report['union']["fpr"], report['union']["tpr"], report['union']["auc"]
    fpr_maj, tpr_maj, auc_maj = report['maj']["fpr"], report['maj']["tpr"], report['maj']["auc"]
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_var,   tpr_var,   label=f"Q-value  (AUC={auc_var:.3f})", linewidth=2.5)
    plt.plot(fpr_ent,   tpr_ent,   label=f"Entropy  (AUC={auc_ent:.3f})", linewidth=2.5)
    plt.plot(fpr_eng,   tpr_eng,   label=f"Energy   (AUC={auc_eng:.3f})", linewidth=2.5)
    plt.plot(fpr_maj,   tpr_maj,   label=f"Majority (AUC={auc_maj:.3f})", linewidth=2.5)
    plt.plot(fpr_union, tpr_union, label=f"Union     (AUC={auc_union:.3f})", linewidth=2.5)

    # random guess baseline
    plt.plot([0, 1], [0, 1], "--", label="Random")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=600)


# Score distributions

def draw_variance_kde(
    variance_train,
    variance_test,
    q_threshold,
    save_path="./plot/variance_kde.png",
    xlabel="Q-value variance",
    font_size=14,   # Adjust all font sizes together.
):
    """variance_train and variance_test are one-dimensional Q-value variance arrays for the training and test states. q_threshold is a quantile; for example, 0.95 selects the 95th percentile of training variance."""

    save_path = project_path(save_path)
    variance_train = np.asarray(variance_train).ravel()
    variance_test  = np.asarray(variance_test).ravel()

    # Remove NaN and infinite values.
    variance_train = variance_train[np.isfinite(variance_train)]
    variance_test  = variance_test[np.isfinite(variance_test)]

    if variance_train.size == 0 or variance_test.size == 0:
        raise ValueError("variance_train / variance_test 중 유효한 값이 없습니다.")

    if not (0.0 < q_threshold < 1.0):
        raise ValueError("q_threshold 는 (0, 1) 사이의 quantile 값이어야 합니다.")

    # Compute the quantile-based threshold.
    thr_value = np.quantile(variance_train, q_threshold)

    # Compute basic statistics.
    mean_train = variance_train.mean()
    std_train  = variance_train.std()
    mean_test  = variance_test.mean()
    std_test   = variance_test.std()

    # Leave enough space on the x-axis to include the threshold.
    zoom_upper = max(
        mean_train + 3 * std_train,
        mean_test  + 3 * std_test,
        thr_value * 1.2,
        variance_train.max(),
        variance_test.max(),
    )
    if zoom_upper < 1e-6:
        zoom_upper = max(0.01, thr_value * 1.2)

    x_max = zoom_upper

    # KDE
    kde_train = stats.gaussian_kde(variance_train)
    kde_test  = stats.gaussian_kde(variance_test)

    x_zoom = np.linspace(0, x_max, 1000)
    pdf_train = kde_train(x_zoom)
    pdf_test  = kde_test(x_zoom)

    # Configure all fonts together through rc_context.
    with plt.rc_context({
        "font.size": font_size,
        "axes.labelsize": font_size+2,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size-3,
    }):
        plt.figure(figsize=(7, 6))

        # Train KDE
        plt.plot(x_zoom, pdf_train, color="C0", linewidth=2, label="Train KDE")
        plt.fill_between(x_zoom, pdf_train, color="C0", alpha=0.3)

        # Test KDE
        plt.plot(x_zoom, pdf_test, color="C1", linewidth=2, label="Test KDE")
        plt.fill_between(x_zoom, pdf_test, color="C1", alpha=0.3)

        # Draw the quantile-based threshold line.
        plt.axvline(
            x=thr_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"OOD Threshold (q={q_threshold:.2f}, v={thr_value:.2f})",
        )

        # Shade the OOD region from thr_value to x_max.
        plt.axvspan(thr_value, x_max, color="red", alpha=0.1)

        # Position the OOD label.
        ymax = max(pdf_train.max(), pdf_test.max())
        x_mid = (thr_value + x_max) / 2.0
        y_text = ymax * 0.6
        plt.text(
            x_mid,
            y_text,
            "OOD Range",
            color="red",
            ha="center",
            va="center",
        )

        plt.xlim(0, x_max)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(save_path, dpi = 600)
        plt.close()

    print(f"KDE plot saved to {save_path}")


# Confusion matrices

def plot_and_save_cm(cm, labels_list=None, filename='confusion_matrix.png'):
    # Use the blue and orange colors from tab10.
    base_colors = plt.get_cmap('tab10').colors[:2]
    cmap = LinearSegmentedColormap.from_list('blue_orange', base_colors, N=256)

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap=cmap)  # Display the heatmap without a color bar.

    n = cm.shape[0]
    if labels_list is None:
        labels_list = list(map(str, range(n)))
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels_list)
    ax.set_yticklabels(labels_list)

    # Display the value in each cell.
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j],
                    ha='center', va='center',
                    fontsize=17,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')

    ax.set_xlabel('Predicted label', fontsize=20)
    ax.set_ylabel('True label', fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    ax.set_title('Sepsis', fontweight='bold')

    plt.tight_layout()
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)


# Embedding figures

def plot_id_vs_ood_embedding(data, path, id_color="#339231", ood_color="#FAAB43", center_color="red",
                            fs_title=18, fs_tick=13, fs_legend=13, fs_metrics=13, fs_annotation=13):
    plt.rcParams.update({"xtick.labelsize": fs_tick, "ytick.labelsize": fs_tick})
    X_id_umap, X_ood_umap = data["X_id_umap"], data["X_ood_umap"]
    center_ood, id_centers_2d = data["center_ood"], data["id_centers_2d"]
    db_index, sil_score = data["db_index"], data["sil_score"]
    fig, ax = plt.subplots()

    # Plot sample points with alpha=0.3.
    ax.scatter(
        X_id_umap[:, 0],
        X_id_umap[:, 1],
        s=10,
        alpha=0.5,
        edgecolors='none',
        color=id_color
    )
    ax.scatter(
        X_ood_umap[:, 0],
        X_ood_umap[:, 1],
        s=10,
        alpha=0.5,
        edgecolors='none',
        color=ood_color
    )

    # Collect the center coordinates: multiple ID centers and one OOD center.
    all_centers = np.vstack([id_centers_2d, center_ood[None, :]])

    # Draw the center markers.
    ax.scatter(
        all_centers[:, 0],
        all_centers[:, 1],
        s=170,
        marker="+",
        linewidths=2.0,
        color=center_color,
    )

    # Connect each ID center to the OOD center with a dashed line and label its distance.
    for c in id_centers_2d:
        ax.plot(
            [c[0], center_ood[0]],
            [c[1], center_ood[1]],
            linestyle="--",
            linewidth=1.5,
            color=center_color,
            alpha=0.8
        )

        dist = np.linalg.norm(c - center_ood)
        mid = (c + center_ood) / 2.0
        ax.text(
            mid[0],
            mid[1],
            f"{dist:.2f}",
            fontsize=fs_annotation,
            color=center_color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
        )

    # ax.set_title(f"UMAP: ID vs Gaussian OOD (sigma={sigma})",
    #              fontsize=fs_title, fontweight='bold')

    # Legend 1: upper right, showing ID/OOD/center labels.
    marker_handles = [
        Line2D([], [], linestyle="None",
               marker='o', color=id_color,
               markersize=6, label="Original ID"),
        Line2D([], [], linestyle="None",
               marker='o', color=ood_color,
               markersize=6, label="Gaussian OOD"),
        Line2D([], [], linestyle="None",
               marker='+', color=center_color,
               markersize=8, markeredgewidth=1.5,
               label="Cluster center")
    ]
    legend_markers = ax.legend(
        handles=marker_handles,
        loc='upper right',
        fontsize=fs_legend,
        markerscale=1.0
    )
    ax.add_artist(legend_markers)

    # Legend 2: lower right, showing DBI and silhouette metrics.
    metric_handles = [
        Line2D([], [], linestyle="None", marker='',
               label=f"Davies–Bouldin index = {db_index:.3f}"),
        Line2D([], [], linestyle="None", marker='',
               label=f"Silhouette score = {sil_score:.3f}")
    ]
    legend_metrics = ax.legend(
        handles=metric_handles,
        loc='lower right',
        fontsize=fs_metrics,
        frameon=True,
        borderpad=0.2,
        labelspacing=0.3,
        handlelength=0.0,
        handletextpad=0.2
    )
    frame = legend_metrics.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    frame.set_edgecolor("black")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path,
                dpi=600)
    plt.close(fig)


def generate_and_plot_id_vs_ood(
    params,
    sigma=1.0,             # Standard deviation of the Gaussian noise.
    laplace_scale=5.0,     # Currently unused.
    n_neighbors=30,        # UMAP n_neighbors
    umap_random_state=42,
    id_kmeans_random_state=42,  # Random seed for K-means clustering of ID samples in UMAP space.
    n_id_clusters=2,              # Number of ID clusters; the default is 2.

    # Color settings.
    id_color="#339231",          # Color of ID points.
    ood_color="#FAAB43",         # Color of OOD points.
    center_color="red",          # Color of centers, connecting lines, and distance labels.

    # Font-size settings; adjust the values here.
    fs_title=18,                 # Title.
    fs_tick=13,                  # Axis tick labels.
    fs_legend=13,                # Upper-right legend: ID/OOD/center.
    fs_metrics=13,               # Lower-right legend: DBI/silhouette.
    fs_annotation=13             # Distance labels.
):

    from detector import prepare_id_vs_ood_embedding
    # Preserve the original style configuration before data preparation.
    plt.rcParams.update({"xtick.labelsize": fs_tick, "ytick.labelsize": fs_tick})
    data = prepare_id_vs_ood_embedding(params, sigma, n_neighbors, umap_random_state,
                                       id_kmeans_random_state, n_id_clusters)
    plot_id_vs_ood_embedding(data,
        project_path(f"plot/{params['target_data']}_umap_id_ood_gaussian.png"),
        id_color, ood_color, center_color, fs_title, fs_tick, fs_legend, fs_metrics, fs_annotation)


def configure_cluster_style(font_tick, font_label, font_legend, font_title):
    from matplotlib import rcParams
    rcParams['xtick.labelsize'] = font_tick
    rcParams['ytick.labelsize'] = font_tick
    rcParams['axes.labelsize'] = font_label
    rcParams['legend.fontsize'] = font_legend
    rcParams['axes.titlesize'] = font_title
