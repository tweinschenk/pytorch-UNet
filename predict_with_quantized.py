import numpy as np
import argparse
import time

import torch
from skimage import io

from unet.dataset import JointTransform2D, ImageToImage2D

from pytorch_nndct.apis import torch_quantizer


quant_mode = "test"

def predict(dataset_path, float_model_path, quant_model_path, output_path):
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


    #load data
    transform = JointTransform2D(crop=(256, 256), p_flip=0, color_jitter_params=None, long_mask=True)
    dataset = ImageToImage2D(dataset_path, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    #load model
    unet = torch.load(float_model_path)
    model = unet.to(device)

    shape = torch.randn([1, 4, 256, 256])
    quantizer = torch_quantizer(quant_mode, model, shape, output_dir=quant_model_path)
    quantized_model = quantizer.quant_model

    time1 = time.time()
    for img, mask, names in data_loader:
        img = img.to(device)
        output = quantized_model(img).cpu().data.numpy()
        output = output * 255
        io.imsave(output_path + "channel_0_" + names[0], output[0,1,:,:].astype(np.uint8))
        io.imsave(output_path + "channel_1_" + names[0], output[0,0,:,:].astype(np.uint8))
    time2 = time.time()
    timetotal = time2 - time1
    fps = float(len(dataset) / timetotal)
    print(" ")
    print("FPS=%.2f, total frames = %.0f, time=%.4f seconds" %(fps, len(dataset), timetotal))
    print(" ")



def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dset_dir', type=str, default='./images_folder',
                    help='Path to test & train datasets.')
    ap.add_argument('--model_path', type=str, default='/checkpoint/unet2D/best_model.pt',
                    help='path of trained model. Default is /checkpoint/unet2D/best_model.pt')
    ap.add_argument('--quant_model_folder', type=str, default='./build/quant_model', help='Path to output folder')
    ap.add_argument('--image_save_folder', type=str, default='./images_folder/inference_quant/')
    args = ap.parse_args()
    
    predict(dataset_path=args.dset_dir, float_model_path=args.model_path,
            quant_model_path=args.quant_model_folder, output_path=args.image_save_folder)


if __name__ == '__main__':
    run_main()

