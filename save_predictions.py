from configs.config import Config
from utils.dataset import CarSegmentationDataset, collate_fn
from utils.postprocess import postprocess

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import pipeline
import ultralytics


data2id = {
    'ade20k': 20,
    'coco': 2,
    'sidewalks': 10
}


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: Config):
    model_cfg = cfg.model
    car_id = data2id[model_cfg.dataset]
    img_size = (cfg.img_size, cfg.img_size)

    dataset = CarSegmentationDataset(cfg.img_dir, cfg.mask_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    if model_cfg.short_name == 'yolov8':
        model = ultralytics.YOLO(model_cfg.name)
    else:
        model = pipeline("image-segmentation", model=model_cfg.name, device=cfg.device)

    for images, masks, filenames in dataloader:
        pred_masks = []
        if model_cfg.short_name == 'yolov8':
            pred = model(images)

            for p in pred:
                if p.masks:
                    conf = p.boxes.conf.cpu().numpy()
                    cls = p.boxes.cls.cpu()
                    car_idx = torch.where(cls == car_id)
                    conf_masks = p.masks.data.cpu() * conf[:, None, None]
                    car_masks = conf_masks[car_idx]
                    if car_masks.shape[0] > 0:
                        p_mask = np.maximum.reduce(car_masks)
                    else:
                        p_mask = np.zeros(img_size, dtype=np.uint8)
                else:
                    p_mask = np.zeros(img_size, dtype=np.uint8)
                pred_masks.append(p_mask)
        else:
            for image in images:
                preprocessed = model.preprocess(image)
                model_outputs = model.forward(preprocessed)
                p_mask = postprocess(model_outputs,  img_size, queries=model_cfg.queries)
                pred_masks.append(p_mask[0, car_id, :, :] if p_mask.shape[0] else np.zeros(img_size, dtype=np.uint8))
        for filename, p_mask in zip(filenames, pred_masks):
            with open(f"pred/{model_cfg.short_name}/{filename.split('.')[0]}.npy", "wb") as f:
                np.save(f, np.around(p_mask, decimals=3))


if __name__ == '__main__':
    main()
