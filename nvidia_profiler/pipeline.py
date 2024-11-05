import torch as tch
import torchvision as tch_vision
import torch.optim as tch_optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from alive_progress import alive_bar

from nvidia_profiler.config import setupEnvironment
from nvidia_profiler.data import OxfordPetDataset

__all__ = [
    'objectDetectionBenchmark',
    'runProfiling'
]

def objectDetectionBenchmark(
    batch_size: int = 2,
    num_epochs: int = 10,
    prefetch_factor: int = 2,
    num_workers: int = 2,
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
    data_loader = DataLoader(
        datasets,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

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
    num_epoch: int = 10,
    batch_size: int = 2,
    prefetch_factor: int = 2,
    num_workers: int = 2,
    device: tch.device = tch.device('cuda:0')
) -> None:
    def printElapsedTime(elapsed_ms: float) -> None:
        # Convert milliseconds to seconds
        seconds = elapsed_ms / 1000

        # Calculate hours, minutes, and remaining seconds
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60

        # Print the result in hours, minutes, and seconds
        print(f'Elapsed time: {hours}:{minutes}:{remaining_seconds:.2f} (hh:mm:ss)')

    start_event = tch.cuda.Event(enable_timing=True)
    end_event = tch.cuda.Event(enable_timing=True)

    start_event.record()
    objectDetectionBenchmark(
        num_epochs=num_epoch,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        device=device
    )
    end_event.record()

    tch.cuda.synchronize()
    printElapsedTime(start_event.elapsed_time(end_event))

def main() -> None:
    config: dict = setupEnvironment()
    print(config)

    if 'device' not in config:
        device: tch.device = tch.device('cuda') if tch.cuda.is_available() else tch.device('cpu')
    else:
        device: tch.device = tch.device(config['device'])

    runProfiling(
        device=device,
        num_epoch=config['num_epochs'],
        batch_size=config['batch_size'],
        prefetch_factor=config['prefetch_factor'],
        num_workers=config['num_workers']
    )

if __name__ == '__main__':
    main()