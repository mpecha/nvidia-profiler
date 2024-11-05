import torch as tch
import torchvision as tch_vision
import torch.optim as tch_optim
import torch.autograd.profiler as profiler

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset

from alive_progress import alive_bar

from nvidia_profiler.data import OxfordPetDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


__all__ = [
    'objectDetectionBenchmark',
    'runProfiling'
]

def objectDetectionBenchmark(
    batch_size: int = 2,
    num_epochs: int = 10,
    device: tch.device = tch.device('cuda')
) -> None:
    # Set up the model
    model = tch_vision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 2

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the dataset
    root = 'data/oxford-iiit-pet'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datasets: VisionDataset = OxfordPetDataset(root=root, transform=transform)
    data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Train the model
    optimizer = tch_optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0005)
    lr_scheduler = tch_optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        with alive_bar(len(data_loader)) as bar:
            for batch_id, (images, targets) in enumerate(data_loader):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                optimizer.zero_grad()
                losses.backward()

                optimizer.step()
                lr_scheduler.step()
                bar()

def runProfiling(
    device: tch.device = tch.device('cuda')
) -> None:
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        objectDetectionBenchmark(device=device, num_epochs=1, batch_size=2)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))


def main() -> None:
    device = tch.device('cuda') if tch.cuda.is_available() else tch.device('cpu')

    runProfiling(device=device)

if __name__ == '__main__':
    main()