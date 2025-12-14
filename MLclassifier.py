from Indicator import *
import pandas as pd

data_rows = []

# Example for ONE image (you will loop this)
row = {
    "EAR_left": EAR_left,
    "EAR_right": EAR_right,
    "EAR_diff": abs(EAR_left - EAR_right),
    "Redness_left": np.mean(redness_index(left_eye_crop)),
    "Redness_right": np.mean(redness_index(right_eye_crop)),
    "Yellow_left": yellow_index(left_eye_crop),
    "Yellow_right": yellow_index(right_eye_crop),
    "label": 1   # <-- set label here
}

data_rows.append(row)

df = pd.DataFrame(data_rows)
df.to_csv("eye_features.csv", index=False)
