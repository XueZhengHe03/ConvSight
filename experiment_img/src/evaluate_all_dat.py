# evaluate_all_dat.py
import torch
from torch.utils.data import DataLoader
from train_baseline_dat import ConvergenceDataset, build_model

models = [
    ("resnet50", "../checkpoints/best_resnet50_dat.pth"),
    ("efficientnet_b0", "../checkpoints/best_efficientnet_b0_dat.pth"),
    ("convnext_tiny", "../checkpoints/best_convnext_tiny_dat.pth"),
    ("vit_small_patch16_224", "../checkpoints/best_vit_small_patch16_224_dat.pth"),
    ("swin_tiny_patch4_window7_224", "../checkpoints/best_swin_tiny_patch4_window7_224_dat.pth"),
    ("efficientnetv2_s", "../checkpoints/best_efficientnetv2_s_dat.pth")
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = ConvergenceDataset('../dataset_new/test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("📊 Baseline 测试结果（dat dataset）")
print("-" * 40)

for name, path in models:
    try:
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

        acc = 100.0 * correct / total
        print(f"{name:<30} : {acc:.2f}%")
    except Exception as e:
        print(f"{name:<30} : ❌ 错误 - {str(e)[:50]}...")
