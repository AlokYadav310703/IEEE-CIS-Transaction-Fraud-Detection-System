import lightgbm as lgb
import joblib
from pathlib import Path

model_path = Path(__file__).parent.parent / "models" / "lgb_gbdt.txt"
joblib_path = Path(__file__).parent.parent / "models" / "lgb_gbdt.joblib"

# Load LightGBM model
model = lgb.Booster(model_file=str(model_path))

# Save as joblib
joblib.dump(model, joblib_path)

print(f"Model saved successfully to: {joblib_path}")