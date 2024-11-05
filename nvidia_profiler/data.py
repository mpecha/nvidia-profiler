import numpy as np
import torch as tch
import torchvision as tch_vision


__all__ = [
    'OxfordPetDataset',
]


class OxfordPetDataset(tch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        transform=None
    ) -> None:
        self._dataset = tch_vision.datasets.OxfordIIITPet(
            root=root,
            target_types='segmentation',
            download=True,
            transform=None
        )
        self.transforms = transform

    def __getitem__(
        self,
        idx
    ) -> (tch.Tensor, dict):
        img, mask = self._dataset[idx]

        # Convert the mask to numpy array and adjust labels
        mask = np.array(mask)
        mask[mask == 3] = 0  # Background is 0
        obj_ids = np.unique(mask)[1:]  # Unique object IDs, ignore background (0)

        # Bounding boxes
        boxes = []
        for obj_id in obj_ids:
            pos = np.where(mask == obj_id)
            xmin, ymin, xmax, ymax = np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) > 0:
            boxes = tch.as_tensor(boxes, dtype=tch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = tch.zeros((0, 4), dtype=tch.float32)  # Empty boxes
            area = tch.tensor([])  # Empty area

        labels = tch.ones((len(obj_ids),), dtype=tch.int64)
        masks = tch.as_tensor(mask == obj_ids[:, None, None], dtype=tch.uint8)

        image_id = tch.tensor([idx])
        iscrowd = tch.zeros((len(obj_ids),), dtype=tch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(
        self
    ) -> int:
        return len(self._dataset)

