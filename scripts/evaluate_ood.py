from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import project_path
from detector import BaseUncertiantyEstimationRL
from plot import draw_variance_kde

if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    kde = BaseUncertiantyEstimationRL()
    train, test = kde.load_data()
    policies = kde.load_policy()

    var_train_score = kde.compute_variance_scores(policies, train)
    var_test_score = kde.compute_variance_scores(policies, test)

    draw_variance_kde(var_train_score,
                      var_test_score, 
                      0.95, 
                      save_path=project_path("./plot/sepsis_q_value_variance_kde.png"), 
                      xlabel="Q-value ensemble variance",
                      font_size=20
                      )
    
    entropy_train_score = kde.compute_entropy_scores(policies, train)
    entropy_test_score = kde.compute_entropy_scores(policies, test)

    draw_variance_kde(entropy_train_score,
                      entropy_test_score, 
                      0.95, 
                      save_path=project_path("./plot/sepsis_entropy_variance_kde.png"), 
                      xlabel="Entropy ensemble variance",
                      font_size=20
                      )

    energy_train_score = kde.compute_energy_scores(policies, train)
    energy_test_score = kde.compute_energy_scores(policies, test)

    draw_variance_kde(energy_train_score,
                      energy_test_score, 
                      0.95, 
                      save_path=project_path("./plot/sepsis_energy_variance_kde.png"), 
                      xlabel="Energy ensemble variance",
                      font_size=20
                      )

    






