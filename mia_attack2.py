import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for hospital in ['H1', 'H2']:
            label = 0 if hospital == 'H1' else 1
            hospital_dir = os.path.join(root_dir, hospital)
            for img_name in os.listdir(hospital_dir):
                self.image_paths.append(os.path.join(hospital_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

# Load Dataset
train_dataset = CustomDataset(root_dir='mia_all2/train', transform=transform)
val_dataset = CustomDataset(root_dir='mia_all2/val', transform=transform)
test_dataset = CustomDataset(root_dir='mia_all2/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet50 models
resnet50 = models.resnet50(pretrained=True)
resnet50 = resnet50.to(device)
resnet50.eval()

# Extract features
def extract_features(loader, model):
    features = []
    labels = []
    paths = []

    with torch.no_grad():
        for inputs, targets, img_paths in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
            paths.extend(img_paths)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels, paths

train_features, train_labels, _ = extract_features(train_loader, resnet50)
val_features, val_labels, _ = extract_features(val_loader, resnet50)
test_features, test_labels, test_paths = extract_features(test_loader, resnet50)

combined_features = np.concatenate([train_features, val_features], axis=0)
combined_labels = np.concatenate([train_labels, val_labels], axis=0)

# Train random forest classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(combined_features, combined_labels)

test_preds = rf_classifier.predict(test_features)
test_probs = rf_classifier.predict_proba(test_features)

results = {
    "Image Path": test_paths,
    "True Label": test_labels,
    "Prob(H1)": test_probs[:, 0],  # Predicted probability of belonging to H1
    "Prob(H2)": test_probs[:, 1],  # Predicted probability of belonging to H2
}

results_df = pd.DataFrame(results)
results_df.to_excel(r"C:\Users\DELL\Desktop\AMIA\mia dataset2\MIA_results2.xlsx", index=False)

print("Classification Report:")
print(classification_report(test_labels, test_preds, target_names=["H1", "H2"]))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(test_labels, test_preds)
print(conf_matrix)

test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Overall Test Accuracy: {test_accuracy:.4f}")

h1_test_indices = np.where(test_labels == 0)[0]
h2_test_indices = np.where(test_labels == 1)[0]

h1_test_preds = test_preds[h1_test_indices]
h1_test_labels = test_labels[h1_test_indices]
h1_accuracy = accuracy_score(h1_test_labels, h1_test_preds)
print(f"H1 Test Accuracy: {h1_accuracy:.4f}")

h2_test_preds = test_preds[h2_test_indices]
h2_test_labels = test_labels[h2_test_indices]
h2_accuracy = accuracy_score(h2_test_labels, h2_test_preds)
print(f"H2 Test Accuracy: {h2_accuracy:.4f}")