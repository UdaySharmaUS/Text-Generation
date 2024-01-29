import pandas as pd
import numpy as np

custom_data = pd.read_csv("CUSTOM_DATA.csv")

custom_model_size = custom_data["Custom_Model_Size_GB"].values
custom_inference_time = custom_data["Custom_Inference_Time_ms"].values
custom_BLEU_score = custom_data["Custom_BLEU_Score"].values
custom_fact_checking_score = custom_data["Custom_Fact_Checking_Score_(0-100)"].values

custom_weights = np.array([0.25, 0.25, 0.3, 0.2])

custom_normalized_matrix = np.column_stack(
    [
        np.max(custom_model_size) / custom_model_size,             
        np.max(custom_inference_time) / custom_inference_time,        
        custom_BLEU_score / np.max(custom_BLEU_score),                
        custom_fact_checking_score / np.max(custom_fact_checking_score) 
    ]
)

custom_weighted_normalized_matrix = custom_normalized_matrix * custom_weights

custom_ideal_solution = np.max(custom_weighted_normalized_matrix, axis=0)
custom_negative_ideal_solution = np.min(custom_weighted_normalized_matrix, axis=0)

custom_distance_to_ideal = np.sqrt(
    np.sum((custom_weighted_normalized_matrix - custom_ideal_solution) ** 2, axis=1)
)
custom_distance_to_negative_ideal = np.sqrt(
    np.sum((custom_weighted_normalized_matrix - custom_negative_ideal_solution) ** 2, axis=1)
)

custom_topsis_scores = custom_distance_to_negative_ideal / (
    custom_distance_to_ideal + custom_distance_to_negative_ideal
)

custom_data["Custom_TOPSIS_Score"] = custom_topsis_scores
custom_data["Custom_Rank"] = custom_data["Custom_TOPSIS_Score"].rank(ascending=False)

print("Custom Model Ranking:")
print(custom_data[["Custom_Model", "Custom_TOPSIS_Score", "Custom_Rank"]])

custom_data.to_csv("custom_result.csv", index=False)
