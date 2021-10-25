import os
import json
import argparse

import torch
import gdown
from PIL import Image

from helpers import draw_box, url_to_img, img_to_bytes, bytes_to_img
from syndicai import PythonPredictor


sample_data = ""
output_dir = "./output"
save_response = True


def run(opt):

    # Convert image url to JSON string
    sample_json = {"url": opt.image}

    # Run a model using PythonPredictor from syndicai.py
    model = PythonPredictor([])
    response = model.predict(sample_json)

    # Save output image locally 
    if opt.save:

        # Convert base64 to .JPEG
        img, format = bytes_to_img(response)

        # Create output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save image
        img_name = f"output.{format}"
        img.save(os.path.join(output_dir, img_name))

    # Print a response in the terminal
    if opt.response:
        print(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="https://raw.githubusercontent.com/marcin-laskowski/university-delaware/master/carolina_bays/sample_data/bay.png", type=str, help='URL to a sample input data')
    parser.add_argument('--save', action='store_true', help='Save output image in the ./output directory')
    parser.add_argument('--response', default=True, type=bool, help='Print a response in the terminal')
    opt = parser.parse_args()
    run(opt)
