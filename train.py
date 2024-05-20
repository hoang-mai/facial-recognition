import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from get_dataset import MaiHoang


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path, models):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_state_dict = models.state_dict()

    pretrained_dict = {k: v for k, v in state_dict.items() if
                       k in model_state_dict and v.size() == model_state_dict[k].size()}

    model_state_dict.update(pretrained_dict)

    models.load_state_dict(model_state_dict)
    print("true")


def main(args):
    if args.model == 'IR_50':
        from backbone.IR_50 import model
    elif args.model == 'IR_50ViT':
        from backbone.IR_50ViT import model
    elif args.model == 'vgg19':
        from backbone.vgg19 import model
    else:
        from backbone.resnet50 import model

    print(model)
    if args.checkpoint_path:
        load_model(args.checkpoint_path, model)

    rafdb_dataset_train = MaiHoang(img_dir=args.image_train,
                                   csv_file=args.csv_train,)
    data_train_loader = DataLoader(rafdb_dataset_train, batch_size=32, shuffle=True, num_workers=4)
    train_image, train_label = next(iter(data_train_loader))
    print(f"Train batch: image shape {train_image.shape}, labels shape {train_label.shape}")

    rafdb_dataset_vali = MaiHoang(img_dir=args.image_test,
                                  csv_file=args.csv_test,)
    data_vali_loader = DataLoader(rafdb_dataset_vali, batch_size=32, shuffle=False, num_workers=0)
    vali_image, vali_label = next(iter(data_vali_loader))
    print(f"Vali batch: image shape {vali_image.shape}, labels shape {vali_label.shape}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    patience = 15
    best_val_acc = 0
    patience_counter = 0
    num_epochs = 40
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in tqdm(data_train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(data_train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in data_vali_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(data_vali_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_acc} , Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epochs.")

        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break
    print("Train Accuracy:", train_accuracies)
    print("Train Loss:", train_losses)
    print("Validation Accuracy:", val_accuracies)
    print("Validation Loss:", val_losses)

    # Lưu vào file JSON
    history_data = {
        'train_accuracy': train_accuracies,
        'train_loss': train_losses,
        'val_accuracy': val_accuracies,
        'val_loss': val_losses
    }

    with open('training_history.json', 'w') as f:
        json.dump(history_data, f)

    print("History data saved to training_history.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--model', default='IR_50ViT', type=str, help='tên model cần train')
    parser.add_argument('--image_train',default='C:\\Users\\Laptop\\Desktop\\dataset\\RAF-DB\\DATASET\\train',help='đường dẫn tới thư mục ảnh train')
    parser.add_argument('--csv_train',default='C:\\Users\\Laptop\\Desktop\\dataset\\RAF-DB\\train_labels.csv',help='đường dẫn tới thư mục csv của ảnh train')
    parser.add_argument('--image_test',default='C:\\Users\\Laptop\\Desktop\\dataset\\RAF-DB\\DATASET\\test',help='đường dẫn tới thư mục ảnh test')
    parser.add_argument('--csv_test',default='C:\\Users\\Laptop\\Desktop\\dataset\\RAF-DB\\test_labels.csv',help='đường dẫn tới thư mục csv ảnh test')
    parser.add_argument('--checkpoint_path',default='C:\\Users\\Laptop\\Desktop\\dataset\\backbone_ir50_ms1m_epoch120.pth',help='đường dẫn tới thư mục chứa file pretrain')

    args = parser.parse_args()
    main(args)
