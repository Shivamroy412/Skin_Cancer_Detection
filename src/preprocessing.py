import config
import os
import glob
from PIL import Image, ImageFile
from joblib import Parallel, delayed
from tqdm import tqdm
import config
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":


    def resize_single_image(image_path, output_folder_path, size):
        """Function to resize a single image"""

        image_file_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder_path, image_file_name)

        image = Image.open(image_path)
        image = image.resize(size, resample = Image.BILINEAR)

        image.save(output_path)


    def parallel_processing(folder_name = "train"):

        processed_image_path = os.path.join(config.INPUT_PATH, "processed_img_"+folder_name)
        os.makedirs(processed_image_path, exist_ok = True)

        image_data_folder = os.path.join(config.DATA_PATH, "jpeg", folder_name)   
        all_images = glob.glob(os.path.join(image_data_folder, "*.jpg"))

        Parallel(n_jobs = 12)(delayed(resize_single_image)(image_path_names, processed_image_path, config.IMAGE_SIZE)
                                                            for image_path_names in tqdm(all_images))    
                                                                                           
        print(f"All {folder_name} images preprocessed")


    parallel_processing("train")
    parallel_processing("test")