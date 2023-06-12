from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os


def init_mtcnn(image_size=160, margin=0, keep_all=False, min_face_size=40,
               device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    Initializes and returns an instance of the MTCNN face detection model.

    :param image_size: int, size of the input images for the MTCNN model (default=160)
    :param margin: int, amount of margin to add around the detected face in pixels (default=0)
    :param keep_all: bool, whether to return all detected faces instead of just the one with highest probability (default=False)
    :param min_face_size: int, minimum size of face in pixels that can be detected by the MTCNN model (default=40)
    :return: MTCNN object
    """
    return MTCNN(image_size=image_size, margin=margin, keep_all=keep_all, min_face_size=min_face_size, device=device)


def init_resnet(pretrained='vggface2'):
    """
    Initializes and returns an instance of the InceptionResnetV1 face recognition model.

    :param pretrained: str, type of pre-trained weights to use for the InceptionResnetV1 model (default='vggface2')
    :return: InceptionResnetV1 object
    """
    return InceptionResnetV1(pretrained=pretrained).eval()


def create_embedding_dataset(model_file, db_folder, pbar=None):
    """
    Given an input folder with aligned face images and a model file path, create and save an embedding dataset

    :param model_file: str, file path to save the embedding dataset
    :param db_folder: str, folder path with aligned face images
    :return: None
    """
    # Create dataset from the input folder
    dataset = datasets.ImageFolder(db_folder)
    # Map indices to class names in the dataset
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    def collate_fn(x):
        return x[0]

    # Create data loader with collate function to extract the image and its index from the dataset
    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = []
    embedding_list = []

    completed = 0
    #pbar.setValue(completed)
    n = 100/loader.__len__()
    # Iterate over the images in the data loader
    for img, idx in loader:
        # Detect the face and get its embedding
        face, prob = mtcnn(img, return_prob=True)
        if face is not None and prob > 0.90:
            emb = resnet(face.unsqueeze(0)).detach().cpu() # .to(device)
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])

        if pbar:
            completed += n
            pbar.setValue(int(completed))

    # Save the embedding dataset to the specified model file path
    data = [embedding_list, name_list]
    torch.save(data, model_file)


def single_image(model_file, image_path, file_path_image):
    """
    Function to perform face recognition on a single image.

    Args:
    - model_file: string, path to the model file containing saved embeddings and names
    - image_path: string, path to the image on which face recognition is to be performed
    - file_path_image: string, path to save the output image

    Returns:
    - None
    """
    # Load the saved embeddings and names from the model file
    load_data = torch.load(model_file)
    embedding_list = load_data[0]
    name_list = load_data[1]

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Detect faces in the image using MTCNN
    img_cropped_list, prob_list = mtcnn_v(img, return_prob=True)

    # If any faces are detected
    if img_cropped_list is not None:
        # Detect the face bounding boxes using MTCNN
        boxes, _ = mtcnn_v.detect(img)

        # For each detected face
        for i, prob in enumerate(prob_list):
            # If the face detection probability is greater than 0.90
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0).to(device)).detach().cpu()

                # Calculate the distance between the face embedding and the embeddings in the model file
                dist_list = []  # list of matched distances, minimum distance is used to identify the person
                for idx, emb_db in enumerate(embedding_list):
                    # dist = torch.dist(emb, emb_db).item()
                    dist = 1 - torch.cosine_similarity(emb, emb_db).item()
                    dist_list.append(dist)

                # Identify the name of the person with minimum distance
                min_dist = min(dist_list)  # get minumum dist value
                min_dist_idx = dist_list.index(min_dist)  # get minumum dist index
                name = name_list[min_dist_idx]  # get name corrosponding to minimum dist

                box = boxes[i]

                # use 0.8 for euclidean and 0.3 for cosine
                if min_dist < 0.3:
                    # Display the name of the person on the image
                    cv2.putText(img, f'{name}', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                                cv2.LINE_AA)

                # Draw a bounding box around the face
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 4)

    # Save the output image
    cv2.imwrite(file_path_image, img)
    # Close all OpenCV windows
    cv2.destroyAllWindows()


def update_recognized_count(name, recognized_count):
    """
    Updates the recognized count for a person.

    Args:
        name (str): The name of the recognized person.
        recognized_count (dict): A dictionary that stores the count of frames for each recognized person.
    """
    if name in recognized_count:
        recognized_count[name] += 1
    else:
        recognized_count[name] = 1


def video(model_file, video_path):
    """
    Runs face recognition on a video.

    Args:
        model_file (str): The file path of the saved model.
        video_path (str): The file path of the video.

    Returns:
        None
    """
    # Load the saved model
    load_data = torch.load(model_file)
    embedding_list = load_data[0]
    name_list = load_data[1]

    # Open the video file
    cam = cv2.VideoCapture(video_path)

    # Get the dimensions of the video
    width = int(cam.get(3))
    height = int(cam.get(4))

    # Create a window to display the video
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", width, height)

    # Initialize a dictionary to store the count of frames for each recognized person
    recognized_count = {}

    # Specify the minimum number of frames for a person to be recognized
    min_recognized_frames = 5

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame, try again")
            continue

        # Resize the frame for faster processing
        frame = cv2.resize(frame, (1920, 1080), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        # Detect faces in the frame
        img = Image.fromarray(frame)
        img_cropped_list, prob_list = mtcnn_v(img, return_prob=True)

        if img_cropped_list is not None:
            # Get the bounding boxes of the detected faces
            boxes, _ = mtcnn_v.detect(img)

            for i, prob in enumerate(prob_list):
                if prob > 0.90:
                    # Get the face embedding
                    emb = resnet(img_cropped_list[i].unsqueeze(0).to(device)).detach().cpu()

                    # Compare the face embedding to the embeddings in the database
                    dist_list = []  # list of matched distances, minimum distance is used to identify the person
                    for idx, emb_db in enumerate(embedding_list):
                        dist = 1 - torch.cosine_similarity(emb, emb_db).item()
                        dist_list.append(dist)

                    # Identify the recognized person with the minimum distance
                    min_dist = min(dist_list)  # get minumum dist value
                    min_dist_idx = dist_list.index(min_dist)  # get minumum dist index
                    name = name_list[min_dist_idx]  # get name corresponding to minimum dist

                    # Draw a rectangle around the detected face
                    box = boxes[i]
                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 4)

                    # use 0.8 for Euclidean and 0.3 for cosine
                    if min_dist < 0.30:
                        # Update the recognized count for the person
                        update_recognized_count(name, recognized_count)

                    if recognized_count.get(name, 0) >= min_recognized_frames:
                        # Add the name of the person to the frame
                        frame = cv2.putText(frame, name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Video", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print('Esc pressed, closing...')
            break

    cam.release()
    cv2.destroyAllWindows()


def save_video(model_file, video_path, file_path_video):
    """
    A function to save a video with bounding boxes and names of recognized people.

    Args:
    model_file (str): The file path to the saved model.
    video_path (str): The file path to the input video.
    file_path_video (str): The file path to save the output video.

    Returns:
    None
    """
    # Load the saved model
    load_data = torch.load(model_file)
    embedding_list = load_data[0]
    name_list = load_data[1]

    # Open the input video
    cam = cv2.VideoCapture(video_path)

    # Get the dimensions of the video
    width = int(cam.get(3))
    height = int(cam.get(4))

    # Create a window to display the video
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", width, height)

    # Define the codec and create a VideoWriter object. The output is stored in 'file_path_video'.
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(file_path_video, fourcc, 29.97, (width, height))

    # Initialize a dictionary to store the count of frames for each recognized person
    recognized_count = {}

    # Specify the minimum number of frames for a person to be recognized
    min_recognized_frames = 5

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame, try again.")
            break

        img = Image.fromarray(frame)
        img_cropped_list, prob_list = mtcnn_v(img, return_prob=True)

        if img_cropped_list is not None:
            boxes, _ = mtcnn_v.detect(img)

            for i, prob in enumerate(prob_list):
                if prob > 0.90:
                    emb = resnet(img_cropped_list[i].unsqueeze(0).to(device)).detach().cpu()

                    dist_list = []  # list of matched distances, minimum distance is used to identify the person

                    for idx, emb_db in enumerate(embedding_list):
                        dist = 1 - torch.cosine_similarity(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list)  # get minumum dist value
                    min_dist_idx = dist_list.index(min_dist)  # get minumum dist index
                    name = name_list[min_dist_idx]  # get name corrosponding to minimum dist
                    print(name)

                    box = boxes[i]

                    original_frame = frame.copy()  # storing copy of frame before drawing on it

                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 4)

                    # use 0.8 for euclidean and 0.3 for cosine
                    if min_dist < 0.30:
                        # Update the recognized count for the person
                        update_recognized_count(name, recognized_count)

                    if recognized_count.get(name, 0) >= min_recognized_frames:
                        # Add the name of the person to the frame
                        frame = cv2.putText(frame, name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 1, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("Video", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print('Esc pressed, closing...')
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()


def bulk_images(model_file, input_path, output_path):
    """
    Matches face id of the given photo with available data from data.pt file and saves the result to output folder.
    :param model_file: path to the .pt file containing the data for face recognition
    :param input_path: path to the folder containing the input images
    :param output_path: path to the folder where the output images will be saved
    """
    # Load the data from .pt file
    load_data = torch.load(model_file)
    embedding_list = load_data[0]
    name_list = load_data[1]

    # Create the new main folder if it doesn't exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Loop through all the subfolders in the main folder
    for subfolder in os.listdir(input_path):
        # Set the path to the subfolder
        subfolder_path = os.path.join(input_path, subfolder)

        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # Create a new subfolder in the aligned folder
            output_subfolder_path = os.path.join(output_path, subfolder)
            if not os.path.exists(output_subfolder_path):
                os.mkdir(output_subfolder_path)

            # Loop through all the images in the subfolder
            for image_name in os.listdir(subfolder_path):
                # Set the path to the image
                image_path = os.path.join(subfolder_path, image_name)

                # Load the image
                img = cv2.imread(image_path)

                img_cropped_list, prob_list = mtcnn(img, return_prob=True)

                if img_cropped_list is not None:
                    boxes, _ = mtcnn.detect(img)
                    if len(boxes) > 0 and len(prob_list) > 0:
                        for i, prob in enumerate(prob_list):
                            if prob > 0.90:
                                # Get the embedding of the cropped face
                                emb = resnet(img_cropped_list[i].unsqueeze(0).to(device)).detach().cpu()

                                # Compare the embeddings with the ones in the data file to identify the person
                                dist_list = []
                                for idx, emb_db in enumerate(embedding_list):
                                    dist = 1 - torch.cosine_similarity(emb, emb_db).item()
                                    dist_list.append(dist)
                                min_dist = min(dist_list)
                                min_dist_idx = dist_list.index(min_dist)
                                name = name_list[min_dist_idx]

                                box = boxes[i]

                                # Draw the name and bounding box on the image if the distance is less than the threshold
                                # use 0.8 for euclidean and 0.3 for cosine
                                if min_dist < 0.30:
                                    cv2.putText(img, f'{name}', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 1, cv2.LINE_AA)
                                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0),
                                              4)

                # Save the output image
                output_image_path = os.path.join(output_subfolder_path, image_name)
                cv2.imwrite(output_image_path, img)

                # Close all the open windows
                cv2.destroyAllWindows()



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

mtcnn = init_mtcnn()
mtcnn_v = init_mtcnn(keep_all = True)
resnet = init_resnet()

# INPUTS
db_folder = '../Databases/Alligned/fyp_group'
model_file = 'fyp_group.pt'
# image_path = 'Inputs/both_1.jpg'
# input_path = 'Inputs/test_1'
video_path =  0 #'Inputs/ibad.mp4' #0 for webcam or 'path/to/video.mp4' or link for live stream 'http://192.168.18.40:8080/video'

# OUTPUTS
# file_path_image = 'Results/Images/lolll.jpg'
# output_path = 'Results/Images/test_5'
# file_path_video = 'Results/Videos/lolll.mp4'

if __name__ == '__main__':
    create_embedding_dataset(model_file, db_folder)

    # single_image(model_file, image_path, file_path_image)
    # bulk_images(model_file, input_path, output_path)

    video(model_file, video_path)
    # save_video(model_file, video_path, file_path_video)
