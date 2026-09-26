import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("FinalModel2.pkl", "rb"))

# Check if model supports feature_importances_
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
    features = [
        "baseline_value", "accelerations", "fetal_movement", "uterine_contractions",
        "light_decelerations", "severe_decelerations", "prolongued_decelerations",
        "abnormal_short_term_variability", "mean_value_of_short_term_variability",
        "percentage_of_time_with_abnormal_long_term_variability",
        "mean_value_of_long_term_variability", "histogram_width", "histogram_min",
        "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes",
        "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance",
        "histogram_tendency"
    ]
    # Sort features by importance
    sorted_idx = np.argsort(importance)[::-1]
    print("Feature Importance Ranking:")
    for idx in sorted_idx:
        print(f"{features[idx]}: {importance[idx]:.4f}")
else:
    print("This model does not support feature importance.")
