import torch
import numpy as np
from torch import nn
import pydoc
from glob import glob
import os
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
import cv2
import time

import warnings
warnings.filterwarnings('ignore')

from src.utils import get_config, parse_args
from src.dataset import Dataset
from src.transforms import ToTensor, ToFloat, RandomCrop
from src.metrics import PSNR

def main():

    args = parse_args()
    config = get_config(args.config)

    o_img_paths = sorted(glob(os.path.join(config['paths']['data'], 'Original/*')))
    f_img_paths = sorted(glob(os.path.join(config['paths']['data'], 'Filtered/*')))

    img_paths = {'original': o_img_paths,
                 'filtered': f_img_paths
                 }

    device = config['inference_params']['device']
    if not torch.cuda.is_available():
        device = 'cpu'
    print(f'Current device is {device}')

    model = pydoc.locate(config['inference_params']['model'])().to(device)
    model = nn.DataParallel(model)
    model_dumps = torch.load(config['paths']['weights'], map_location=device)
    model.load_state_dict(model_dumps['model_state_dict'])
    crop_size = config['inference_params']['crop_size']


    model.eval()
    torch.set_grad_enabled(False)

    if crop_size:
        transforms = Compose([ToFloat(), RandomCrop(crop_size), ToTensor()])
    else:
        transforms = Compose([ToFloat(), ToTensor()])

    dataset = Dataset(paths=img_paths,
                      transform=transforms
                     )

    dataloader = DataLoader(dataset,
                            batch_size=config['data_params']['batch_size'],
                            num_workers=config['data_params']['num_workers'],
                            shuffle=False
                            )
    psnr_scores_model = []
    psnr_scores_ref = []

    tic = time.time()
    for sample in dataloader:
        X = sample['X'].to(device)
        Y = sample['Y'].to(device)
        name = sample['name']

        Y_pred = model(X)

        X = X.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()
        Y_pred = Y_pred.detach().cpu().numpy()

        psnr_scores_model.append(PSNR(Y, Y_pred))
        psnr_scores_ref.append(PSNR(Y, X))

        # save predicted images
        for idx in range(X.shape[0]):
            cv2.imwrite(os.path.join(config['paths']['results'], name[idx]),
                        Y_pred[idx, 0, :, :] * 255)

    toc = time.time()

    # save PSNR scores to logs
    model_name = config['inference_params']['model_name']
    with open(config['paths']['log'], 'a') as f:
        f.write(f'Model {model_name}\n')
        f.write('='*20 + '\n')
        f.write(f'PSNR (mean ± std) of refs and predicted images: {np.mean(psnr_scores_model)} ± {np.std(psnr_scores_model)}\n')
        f.write(f'PSNR (mean ± std) of refs and noisy images: {np.mean(psnr_scores_ref)} ± {np.std(psnr_scores_ref)}\n')
        f.write(f'Elapsed time: {toc - tic}, s\n')
        f.write('='*50 + '\n'*2)

if __name__ == '__main__':
    main()