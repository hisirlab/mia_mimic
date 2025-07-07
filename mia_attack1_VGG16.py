import os
import torch
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd

weightPath = r"C:\Users\DELL\Desktop\AMIA\model_2\vgg16\H1_model_VGG16.pth"


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data_from_dir(split_dir):
    image_paths = []
    labels = []

    normal_dir = os.path.join(split_dir, 'normal')
    tumor_dir = os.path.join(split_dir, 'tuberculosis')

    if os.path.exists(normal_dir):
        normal_images = os.listdir(normal_dir)
        for img in normal_images:
            image_paths.append(os.path.join(normal_dir, img))
            labels.append(1)  # normal

    if os.path.exists(tumor_dir):
        tumor_images = os.listdir(tumor_dir)
        for img in tumor_images:
            image_paths.append(os.path.join(tumor_dir, img))
            labels.append(0)  # TB

    return image_paths, labels

def build_model_vgg16():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5):
    early_stopping = EarlyStopping(patience=patience, delta=0.001)
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).view(-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {correct / total}")

        val_labels, val_preds, val_probs = evaluate_model(model, val_loader, device)
        val_loss = criterion(torch.tensor(val_probs), torch.tensor(val_labels).float()).item()

        print(f"Validation Loss: {val_loss}")
        val_losses.append(val_loss)

        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    return val_losses


def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []  # Store predicted probabilities
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = outputs.view(-1).cpu().numpy()  # Convert logits to probabilities
            predicted = (probs > 0.5).astype(int)  # Convert probabilities to binary predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted)
            all_probs.extend(probs)  # Store probabilities for ROC curve

    return all_labels, all_preds, all_probs  # Return probabilities for ROC curve


def plot_roc_curve(test_labels, predictions, save_path=None):
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_loss_curve(val_losses, save_path=None):
    plt.figure()
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curve')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(cm, labels, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Spectral", xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 16})
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU instead.")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = r"C:\Users\DELL\Desktop\AMIA\TB_Chest_Radiography_Database\MIA_2\1_Split\train"
    val_dir = r"C:\Users\DELL\Desktop\AMIA\TB_Chest_Radiography_Database\MIA_2\1_Split\val"
    test_dir = r"C:\Users\DELL\Desktop\AMIA\TB_Chest_Radiography_Database\MIA_2\1_Split\test"

    train_paths, train_labels = load_data_from_dir(train_dir)
    val_paths, val_labels = load_data_from_dir(val_dir)
    test_paths, test_labels = load_data_from_dir(test_dir)

    train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transform)
    test_dataset = CustomDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model_vgg16().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    save_dir = r"D:\Codes\PycharmProjects\amia\results_2\VGG16\H1_1"
    os.makedirs(save_dir, exist_ok=True)

    val_losses = train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device)

    torch.save(model.state_dict(), weightPath)

    test_labels, pred_labels, predictions = evaluate_model(model, test_loader, device)

    print(classification_report(test_labels, pred_labels, target_names=['Tuberculosis', 'Normal'], digits=3))

    cm = confusion_matrix(test_labels, pred_labels)
    print(cm)

    plot_loss_curve(val_losses, os.path.join(save_dir, "loss_curve.png"))
    plot_roc_curve(test_labels, predictions, os.path.join(save_dir, "roc_curve.png"))
    plot_confusion_matrix(cm, ['Tuberculosis', 'Normal'], os.path.join(save_dir, "confusion_matrix.png"))