import time
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose=0):
    """
    args:
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        verbose: level of feedback
    """
    if verbose > 0:
        print(f'Processing a total of {len(loader)} batches.')

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():    
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)
            
            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    
    return output_path

def process_slide(bag_candidate_idx, bags_dataset, args, model, img_transforms, loader_kwargs):
    """
    Function to process each slide in parallel.
    """
    slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
    bag_name = slide_id + '.h5'
    h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
    slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
    
    if not args.no_auto_skip and os.path.exists(os.path.join(args.feat_dir, 'pt_files', bag_name + '.pt')):
        print(f'Skipped {slide_id}')
        return

    print(f'Processing {slide_id} ({bag_candidate_idx}/{len(bags_dataset)})')

    output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
    time_start = time.time()
    wsi = openslide.open_slide(slide_file_path)

    dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
    output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)

    time_elapsed = time.time() - time_start
    print(f'Computing features for {output_file_path} took {time_elapsed} seconds.')

    with h5py.File(output_file_path, "r") as file:
        features = file['features'][:]
        print('Features size: ', features.shape)
        print('Coordinates size: ', file['coords'].shape)

    features = torch.from_numpy(features)
    torch.save(features, os.path.join(args.feat_dir, 'pt_files', os.path.splitext(bag_name)[0] + '.pt'))

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction with Parallelism')
    parser.add_argument('--data_h5_dir', type=str, default=None)
    parser.add_argument('--data_slide_dir', type=str, default=None)
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for parallel processing')
    args = parser.parse_args()

    if args.csv_path is None:
        raise NotImplementedError("CSV path is required.")

    print('Initializing dataset...')
    bags_dataset = Dataset_All_Bags(args.csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model.eval()
    model = model.to(device)

    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == 'cuda' else {}

    total = len(bags_dataset)
    print(f'Total slides to process: {total}')

    process_fn = partial(process_slide, bags_dataset=bags_dataset, args=args, model=model, img_transforms=img_transforms, loader_kwargs=loader_kwargs)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(process_fn, range(total)), total=total))

if __name__ == '__main__':
    main()
