from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from BasicCNN import BasicCNN
import torch
from dataset_preparation import test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BasicCNN().to(device)


model.eval()
with torch.no_grad():
    preds = []
    labelss = []
    for input, labels in test_loader:
        input = input.to(device)
        labels = labels.to(device)
        output = model(input)
        _ , pred = torch.max(output, 1)

        preds.extend(pred.cpu().numpy())

        labelss.extend(labels.cpu().numpy())
    accuracy = accuracy_score(labelss, preds)
    precision = precision_score(labelss, preds, average="weighted")
    recall = recall_score(labelss, preds, average="weighted")
    f1 = f1_score(labelss, preds, average="weighted")

    # 🔹 Print Results
    print(f"✅ Model Evaluation Complete!")
    print(f"📌 Accuracy: {accuracy:.4f}")
    print(f"📌 Precision: {precision:.4f}")
    print(f"📌 Recall: {recall:.4f}")
    print(f"📌 F1-score: {f1:.4f}")