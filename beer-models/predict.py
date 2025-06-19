import pickle
import numpy as np

print("▶ 開始")

with open('黒ビール_model.pkl', 'rb') as f:
    model = pickle.load(f)

input_data = np.array([[0, 1.2, 2.5, 4.0, 5.6, 23.4, 0, 1, 6, 65.0]])

y_pred = model.predict(input_data)
print("✅ 予測完了")
print(f"予測販売数（黒ビール）: {y_pred[0]:.2f} 本")

import math
print(f"発注数（切り上げ）: {math.ceil(y_pred[0])} 本")
