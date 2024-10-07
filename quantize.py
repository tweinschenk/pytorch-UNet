#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from unet.unet import UNet2D
from unet.model import Model
from unet.utils import Logger, MetricList
from unet.metrics import jaccard_index, f1_score, LogNLLLoss
from functools import partial
from unet.dataset import JointTransform2D, ImageToImage2D


def quantize(model_path, quant_mode, batchsize, output_folder, dataset_folder):
    # use GPU if available
    if (torch.cuda.device_count() > 0):
        print('You have', torch.cuda.device_count(), 'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device', str(i), ': ', torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available...selecting CPU')
        device = torch.device('cpu')

    # load trained model
    unet = torch.load(model_path)
    model = unet.to(device)

    # override batchsize if in test mode
    if (quant_mode == 'test'):
        batchsize = 1

    rand_in = torch.randn([1, 4, 256, 256])
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=output_folder)
    quantized_model = quantizer.quant_model

    tf_val = JointTransform2D(crop=(256, 256), p_flip=0.5, color_jitter_params=None, long_mask=True)
    predict_dataset = ImageToImage2D(dataset_folder, tf_val)
    val_loader = DataLoader(predict_dataset, batchsize, shuffle=False)
    loss = LogNLLLoss()
    quantized_model.loss = loss

    if quant_mode == 'calib':
        for img, mask, names in val_loader:
            img = img.to(device)
            _ = quantized_model(img)

    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()

    if quant_mode == 'test':
        logger = Logger(verbose=True)
        metric_list = MetricList({'jaccard': partial(jaccard_index),
                                  'f1': partial(f1_score)})
        logs = val_epoch(model=quantized_model, dataset=predict_dataset, device=device, n_batch=batchsize,
                         metric_list=metric_list)
        logger.log(logs)
        logger.to_csv(os.path.join(output_folder, 'logs.csv'))
        quantized_model.loss = None
        quantizer.export_xmodel(deploy_check=True, output_dir=output_folder)
    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dset_dir', type=str, default='./images_folder',
                    help='Path to test & train datasets. Default is dataset')
    ap.add_argument('-m', '--model_path', type=str, default='/checkpoint/unet2D/best_model.pt',
                    help='path of trained model. Default is /checkpoint/unet2D/best_model.pt')
    ap.add_argument('-q', '--quant_mode', type=str, default='calib', choices=['calib', 'test'],
                    help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('-b', '--batchsize', type=int, default=10,
                    help='Testing batchsize - must be an integer. Default is 10')
    ap.add_argument('-o', '--output_folder', type=str, default='./build/quant_model/', help='Path to output folder')
    args = ap.parse_args()

    print("\n")
    print('PyTorch version : ', torch.__version__)
    print(sys.version)
    print("\n")
    print(' Command line options:')
    print('--dset_dir  : ', args.dset_dir)
    print('--model_path  : ', args.model_path)
    print('--quant_mode   : ', args.quant_mode)
    print('--batchsize    : ', args.batchsize)
    print('--output_folder  : ', args.output_folder)
    print("\n")

    quantize(args.model_path, args.quant_mode, args.batchsize, args.output_folder, args.dset_dir)

    return


def val_epoch(model, dataset, device, n_batch=1, metric_list=MetricList({})):
    """
  Validation of given dataset.

  Args:
       dataset: an instance of unet.dataset.ImageToImage2D
       n_batch: size of batch during training
       metric_list: unet.utils.MetricList object, which contains metrics
          to be recorded during validation

  Returns:
      logs: dictionary object containing the validation loss and
          the metrics given by the metric_list object
  """

    metric_list.reset()
    running_val_loss = 0.0

    for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch)):
        X_batch = Variable(X_batch.to(device=device))
        y_batch = Variable(y_batch.to(device=device))

        y_out = model(X_batch)
        training_loss = model.loss(y_out, y_batch)
        running_val_loss += training_loss.item()
        metric_list(y_out, y_batch)

    del X_batch, y_batch

    logs = {'val_loss': running_val_loss / (batch_idx + 1),
            **metric_list.get_results(normalize=batch_idx + 1)}

    return logs


if __name__ == '__main__':
    run_main()
