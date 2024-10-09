import os

from argparse import ArgumentParser
import time
import torch

from unet.model import Model
from unet.dataset import Image2D, ImageToImage2D, JointTransform2D

parser = ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--results_path', required=True, type=str)
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
args = parser.parse_args()

transform = JointTransform2D(crop=(256, 256), p_flip=0, color_jitter_params=None, long_mask=True)
predict_dataset = ImageToImage2D(args.dataset, transform)
model = torch.load(args.model_path)

if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)

model = Model(model, loss=None, optimizer=None, checkpoint_folder=args.results_path, device=args.device)

time1 = time.time()
model.predict_dataset(predict_dataset, args.results_path)
time2 = time.time()
timetotal = time2 - time1
fps = float(len(predict_dataset) / timetotal)
print(" ")
print("FPS=%.2f, total frames = %.0f, time=%.4f seconds" %(fps, len(predict_dataset), timetotal))
print(" ")
