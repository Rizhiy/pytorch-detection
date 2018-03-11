import numpy as np
import torch

from utils.config import cfg


def resize_collate(batch):
    """
    Resize and pad images so that they become the same size
    :param batch:
    :return:
    """

    def get_batched_array(shape):
        return np.random.randint(256, size=shape)

    shapes = np.array([data['shape'] for img, data in batch])
    max_shape = np.max(shapes, axis=0)

    # Check if images at full size will fit and if not, find maximum possible size
    max_area = max_shape[0] * max_shape[1]
    batched_data = [{}] * len(batch)
    if max_area > cfg.TRAIN.MAX_AREA:
        resize_factor = np.sqrt(cfg.TRAIN.MAX_AREA / max_area)
        max_shape = np.floor(max_shape * resize_factor)

        assert max_shape[0] * max_shape[1] <= cfg.TRAIN.MAX_AREA

        # TODO: Check whether it is better to fill with random values or mean values
        batched_images = get_batched_array((len(batch), *max_shape.astype(int)[::-1], cfg.DATASET.IMG_CHANNELS))
        for idx, (img, data) in enumerate(batch):
            img_resize_factor = np.max(data['shape'] / max_shape)
            target_size = (data['shape'] / img_resize_factor).astype(int)
            new_img = img.resize(target_size)
            new_boxes = data['boxes'] * img_resize_factor
            new_scale = data['scale'] * img_resize_factor
            new_shape = new_img.size
            new_data = data
            new_data.update({'scale': new_scale,
                             'boxes': new_boxes,
                             'shape': new_shape})
            batched_images[idx, :new_shape[1], :new_shape[0], :] = np.array(new_img)
            batched_data[idx] = new_data
    else:
        batched_images = get_batched_array((len(batch), *max_shape[::-1], cfg.DATASET.IMG_CHANNELS))

        for idx, (img, data) in enumerate(batch):
            shape = data['shape']
            batched_images[idx, :shape[1], :shape[0], :] = np.array(img)
            batched_data[idx] = data

    # Normalise images
    batched_images = (batched_images / 256 - cfg.NETWORK.PIXEL_MEANS) / cfg.NETWORK.PIXEL_STDS
    return torch.from_numpy(batched_images), batched_data
