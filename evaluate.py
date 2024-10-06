import os

from argparse import ArgumentParser
from functools import partial

import torch

from unet.model import Model
from unet.dataset import ImageToImage2D, JointTransform2D
from unet.metrics import jaccard_index, f1_score, LogNLLLoss
from unet.utils import Logger, MetricList

parser = ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--results_path', required=True, type=str)
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--batch_size', default=10, type=int)
args = parser.parse_args()

tf_val = JointTransform2D(crop=(256, 256), p_flip=0.5, color_jitter_params=None, long_mask=True)
predict_dataset = ImageToImage2D(args.dataset, tf_val)
model = torch.load(args.model_path)

if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)

loss = LogNLLLoss()
model = Model(model, loss=loss, optimizer=None, checkpoint_folder=args.results_path, device=args.device)

logger = Logger(verbose=True)
metric_list = metric_list = MetricList({'jaccard': partial(jaccard_index),
                                  'f1': partial(f1_score)})
logs = model.val_epoch(dataset=predict_dataset,n_batch=args.batch_size,  metric_list=metric_list)
logger.log(logs)
logger.to_csv(os.path.join(args.results_path, 'evaluation_logs.csv'))
