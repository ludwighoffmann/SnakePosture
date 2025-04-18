"""
This file is for loading in a folder of images, analyzing all of them, and
then outputting the results of each image analysis into folder extracted_data_signed for
use in later visualization code.
"""
import argparse
import os
import pickle
import numpy as np
import vid_helpers as vh
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p", "--ims_path", help="path of folder containing images", type=str
)
parser.add_argument(
    "-ppcm", "--px_per_cm", help="pixels per centimeter in the image", type=float
)

parser.add_argument(
    "-n",
    "--reaching_num",
    help="how many pixels counts as reaching high in an image",
    type=int,
)

args = parser.parse_args()
ims_path = args.ims_path
px_per_cm = args.px_per_cm
reaching_px = args.reaching_num
save_folder = ims_path + "/extracted_data_signed"
os.mkdir(save_folder)

files = []
for filename in os.listdir(ims_path):
    if filename.endswith(".png"):
        im_file = os.path.join(ims_path, filename)
        files.append(im_file)

# sort the list of files
def extract_nums_from_str(string: str) -> int:
    """
    Extract numbers from string of the form 'stringxxxx.png'
    """
    return int(string[-8:-4])


files = sorted(files, key=extract_nums_from_str)


i = 0
num_points = 25  # Has to be changed based on trial length. For 70cm 25 interpolation points. 
for im_file in tqdm(files):
    os.mkdir(ims_path + f"/extracted_data_signed/{i}")
    img = vh.load_img(img_path=im_file)
    front = vh.extract_front(img)
    bottom = vh.extract_bottom(img)
    hybrid = vh.hybridize(front=front, bottom=bottom, reaching_num=reaching_px)
    distance = vh.get_distance_parameter(outline=hybrid, px_per_cm=px_per_cm)
    interp_points, interpolated_outline, outline_interp = vh.get_outline_interpolation(
        distance=distance, outline=hybrid, num_points=num_points
    )
    (
        smooth_kappa,
        curvature_interp_pts,
        curvature_interpolator,
    ) = vh.get_curvature_interpolation(
        interpolated_outline=interpolated_outline,
        interpolation_points=interp_points,
        px_per_cm=px_per_cm,
    )
    # save the curvature interpolator and the points I do the interpolation at.
    with open(save_folder + f"/{i}/interpolator.pkl", "wb") as f:
        pickle.dump(curvature_interpolator, f)
    np.savez(
        save_folder + f"/{i}/arr",
        curvature_interp_pts,
    )
    i += 1
