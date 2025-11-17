from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

MODEL_PATH = "models/sentiment"

print("🔍 Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

print("✅ Model loaded successfully!")

# ============
# TEST INPUT
# ============
test_texts = [
    "pelayanan sangat baik saya berterimakasih karena telah disembuhkan",
    "Dokternya jutek dan tidak ramah",
    "Prosesnya cepat dan mudah"
]

# label mapping (sesuaikan seperti saat training)
label_map = {
    0: "negatif",
    1: "netral",
    2: "positif"
}

print("\n🚀 Testing predictions...\n")

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
        label = label_map[pred_id]

    print(f"📝 Ulasan: {text}")
    print(f"➡️ Prediksi: {label}  (class_id={pred_id})\n")