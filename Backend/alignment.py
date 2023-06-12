
from mtcnn import MTCNN
import cv2
import os
import numpy as np

# Augmentation
import imgaug.augmenters as iaa
from typing import List
from torchvision import datasets, transforms

class ImageAligner:
    def __init__(self, detector, pbar):
        self.progressBar = pbar
        self.completed = 0
        self.detector = detector

    def align(self, img):
        try:
            data = self.detector.detect_faces(img)  # Use the detector to detect the faces in the image
        except:
            return (False, img)
        biggest = 0
        if data != []:  # If at least one face is detected
            for faces in data:
                box = faces['box']  # Get the bounding box coordinates of the face
                area = box[3] * box[2]  # Calculate the area of the bounding box
                if area > biggest:  # If this is the largest face so far, update the biggest area
                    biggest = area
                    bbox = box
                    keypoints = faces['keypoints']  # Get the keypoints of the face
                    left_eye = keypoints['left_eye']  # Get the coordinates of the left eye
                    right_eye = keypoints['right_eye']  # Get the coordinates of the right eye
            lx, ly = left_eye  # Unpack the coordinates of the left eye
            rx, ry = right_eye  # Unpack the coordinates of the right eye
            dx = rx - lx  # Calculate the difference in x-coordinates between the eyes
            dy = ry - ly  # Calculate the difference in y-coordinates between the eyes
            tan = dy / dx  # Calculate the tangent of the angle between the eyes
            theta = np.arctan(tan)  # Calculate the angle in radians
            theta = np.degrees(theta)  # Convert the angle to degrees
            img = self.rotate_bound(img, theta)  # Rotate the image by the calculated angle
            return (True, img)  # Return a tuple indicating success and the rotated image
        else:  # If no face is detected
            return (False, img)  # Return a tuple indicating failure and no image

    def crop_image(self, img):
        data = self.detector.detect_faces(img)
        biggest = 0
        if data != []:
            for faces in data:
                box = faces['box']
                area = box[3] * box[2]
                if area > biggest:
                    biggest = area
                    bbox = box
            bbox[0] = 0 if bbox[0] < 0 else bbox[0]
            bbox[1] = 0 if bbox[1] < 0 else bbox[1]
            img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
            return (True, img)
        else:
            return (False, img)

    def rotate_bound(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))

    def align_db(self, main_folder, aligned_folder, height=None, width=None):

        folders = os.listdir(main_folder)
        if not os.path.exists(aligned_folder):
            os.makedirs(aligned_folder)

        self.completed = 0
        self.progressBar.setValue(int(self.completed))
        n = 100 / len(folders)
        for subfolder in folders:
            if self.progressBar:
                self.completed += n
                self.progressBar.setValue(int(self.completed))

            subfolder_path = os.path.join(main_folder, subfolder)
            if os.path.isdir(subfolder_path):

                aligned_subfolder_path = os.path.join(aligned_folder, subfolder)
                if not os.path.exists(aligned_subfolder_path):
                    os.mkdir(aligned_subfolder_path)

                for image_name in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, image_name)

                    aligned_image_path = os.path.join(aligned_subfolder_path, image_name)

                    img = cv2.imread(image_path)
                    success, img = self.align(img)
                    if success:
                        successT, img = self.crop_image(img)
                        if successT:
                            if height is not None and width is not None:
                                img = cv2.resize(img, (width, height))
                        cv2.imwrite(aligned_image_path, img)
                    else:
                        try:
                            cv2.imwrite(aligned_image_path, img)
                        except:
                            pass
    def augment(self, aligned_folder):
        # Define the augmentations
        seq = iaa.Sequential([
            iaa.Affine(rotate=(-20, 20)),  # rotate between -20 and 20 degrees
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),  # adjust brightness by up to +/-30
            iaa.GaussianBlur(sigma=(0, 1.5)),  # add Gaussian blur with sigma between 0 and 1.5
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
            # add Gaussian noise with standard deviation up to 5% of image intensity range
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),  # scale image by up to 20%
        ])
        # Loop over every subfolder in the main folder
        folders = os.listdir(aligned_folder)
        for folder in folders:
            folder_path = os.path.join(aligned_folder, folder)

            # Only process the subfolders that contain images
            if os.path.isdir(folder_path) and any(
                    filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) for filename in
                    os.listdir(folder_path)):

                # Loop over every image in the subfolder
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

                        # read image from file
                        image_path = os.path.join(folder_path, filename)
                        image = cv2.imread(image_path)

                        # Convert the image to a numpy array
                        img_arr = np.array(image)

                        # Augment the image 20 times
                        images_aug = seq(images=np.tile(img_arr, (20, 1, 1, 1)))

                        # Save the augmented images using OpenCV
                        for i in range(20):
                            # Save the image
                            cv2.imwrite(image_path + f'augmented_{i}.jpg', images_aug[i])


def initialize_db(db_folder, augment=True, pbar=None):

    aligned_folder = os.path.join(db_folder, 'aligned')

    # Load the MTCNN model
    detector = MTCNN()

    image_aligner = ImageAligner(detector, pbar)
    image_aligner.align_db(db_folder, aligned_folder, height=160, width=160)
    if augment:
        image_aligner.augment(aligned_folder)

    return aligned_folder


if __name__ == '__main__':
    initialize_db('/home/haroon/PythonProjects/DroneItDown/dataset_1')
