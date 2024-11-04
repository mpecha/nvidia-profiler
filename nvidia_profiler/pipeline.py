from torchvision.datasets import VisionDataset
from nvidia_profiler.data import getDatasets

def runProfiling() -> None:
    datasets: tuple[VisionDataset, VisionDataset] = getDatasets()
    # TODO train the model

def main() -> None:
    runProfiling()

if __name__ == '__main__':
    main()