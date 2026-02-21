import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "nb_pipeline.pkl"

model = joblib.load(model_path)

# New text sample
new_sample = [
    "fox attacks blairs tory lies tony blair lied when he took the uk to war"
]

prediction = model.predict(new_sample)

label_map = {
    0: "Politics",
    1: "Sport",
    2: "Technology",
    3: "Entertainment",
    4: "Business"
}

print("Predicted Category:", label_map[prediction[0]])