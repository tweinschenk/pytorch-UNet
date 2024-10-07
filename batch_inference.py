from skimage import io
import numpy as np
import re
import xir
import vitis_ai_library
import os
import time
import sys
import argparse

from torchvision import transforms as T
from torchvision.transforms import functional as F


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dimgs(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images
    
    
def preprocess_fn(image_path):
    image = io.imread(image_path)
    image = F.to_pil_image(image)
    # random crop
    i, j, h, w = T.RandomCrop.get_params(image, (256, 256))
    image = F.crop(image, i, j, h, w)
    data = np.asarray( image, dtype="float32" ) / 255
    return data


def reshape_inputs(image_list, inputTensors):
    pass

        
def run_inference(runner, image_list):
    inputTensors = runner.get_input_tensors()
    output_tensor_buffers = runner.get_outputs() 

    #list to buffer outputs
    outputs = []
    
    for i in range(0, len(image_list)):
        image_list[i] = image_list[i].reshape(inputTensors[0].dims)
    
    
    #DPU execution
    print("run DPU")
    time1 = time.time()
    
    for image in image_list:
        #prepare inputData
        inputData = []
        for inputTensor in inputTensors:
            inputData.append(image)
        
        job_id = runner.execute_async(inputData, output_tensor_buffers)
        runner.wait(job_id)
        outputs.append(np.array(output_tensor_buffers[0]))
    
    time2 = time.time()
    timetotal = time2 - time1
    fps = float(len(image_list) / timetotal)
    print(" ")
    print("FPS=%.2f, total frames = %.0f , time=%.4f seconds" %(fps,len(image_list), timetotal))
    print(" ")
    
    return outputs


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dset_dir', type=str, default='./images_folder/',
                    help='Path to test & train datasets. Default is dataset')
    ap.add_argument('-m', '--model_path', type=str, default='./build_test/comp_model/UNet2D_compiled.xmodel',
                    help='path of trained and compiled model.')
    args = ap.parse_args()
    
    #Pre Processing images    
    images_list=[f for f in os.listdir(os.path.join(args.dset_dir + "images")) if re.match(r'.*\.png', f)]
    runTotal = len(images_list)           
    print('Found',len(images_list),'images - processing',runTotal,'of them')
    img = []
    for i in range(runTotal):
        path = os.path.join(os.path.join(args.dset_dir + "images"),images_list[i])
        img.append(preprocess_fn(path))
    
    g = xir.Graph.deserialize(args.model_path)          
    runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
                                       
    run_inference(runner, img)

                                       
if __name__ == "__main__":
    main()
