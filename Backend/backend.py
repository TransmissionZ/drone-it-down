import asyncio

from fastapi import FastAPI, Response, File, UploadFile, Form
from fastapi.responses import StreamingResponse
# from torchvision import datasets
# from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2
import time
import os
from typing import List

from pathlib import Path
import shutil

app = FastAPI()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


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
    return InceptionResnetV1(pretrained=pretrained).eval().to(device)


mtcnn = init_mtcnn()
mtcnn_v = init_mtcnn(keep_all=True)
resnet = init_resnet()

upload_dir = os.path.join(os.getcwd(), "models")
rtmp = 'rtmp://34.66.216.13/live/'


@app.post("/upload_model")
async def upload_model(model=File(...)):
    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # get the destination path
    dest = os.path.join(upload_dir, model.filename)
    print(dest)

    # copy the file contents
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(model.file, buffer)

    return {"filename": model.filename}


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


matched = []


def stream_video(uid, model_file):
    global matched
    # Load the saved model
    load_data = torch.load(os.path.join(upload_dir, model_file))
    embedding_list = load_data[0]
    name_list = load_data[1]
    print('model loaded')
    # Define the video source
    # video = cv2.VideoCapture(1) # uid

    # Set the video encoding parameters
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 30.0
    frame_size = (1920, 1080)

    # Define the video writer
    # writer = cv2.VideoWriter("output.avi", fourcc, fps, frame_size)

    # Define a generator function to yield video frames
    # def gen_frames():
    try:
        url = rtmp + uid
        cam = cv2.VideoCapture(url)
    except Exception as e:
        return False, 'Error Fetching Video: ' + str(e)
    # Initialize a dictionary to store the count of frames for each recognized person
    recognized_count = {}

    # Specify the minimum number of frames for a person to be recognized
    min_recognized_frames = 5
    c = 0
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("fail to grab frame, try again")
            c += 1
            if c > 1000:
                raise asyncio.CancelledError
            break

        # Resize the frame for faster processing
        frame = cv2.resize(frame, (852, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        img = Image.fromarray(frame)
        # img_cropped_list, prob_list = mtcnn_v(img, return_prob=True)
        # Detect faces
        boxes, prob_list, batch_points = mtcnn_v.detect(img, landmarks=True)

        # Extract faces
        img_cropped_list = mtcnn_v.extract(img, boxes, None)

        if img_cropped_list is not None:
            # boxes, _ = mtcnn_v.detect(img)

            for i, prob in enumerate(prob_list):
                if prob > 0.90:
                    # emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()
                    # box = boxes[i]
                    # original_frame = frame.copy()  # storing copy of frame before drawing on it
                    # frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                    #                       (0, 0, 255), 4)

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
                        if name not in matched:
                            matched.append(name)
                        # Add the name of the person to the frame
                        frame = cv2.putText(frame, name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the video source and writer
    cam.release()


async def streamer(gen):
    try:
        for i in gen:
            yield i
            await asyncio.sleep(0.00000000001)
    except asyncio.CancelledError:
        print("caught cancelled error")


@app.get("/video_feed/{uid}/{model_file}")
async def video_feed(uid: str, model_file: str):
    # Define the response stream
    response = StreamingResponse(streamer(stream_video(uid, model_file)),
                                 media_type="multipart/x-mixed-replace; boundary=frame")
    return response


@app.get("/get_models/{uid}")
async def get_models(uid: str):
    return {'models': [f for f in os.listdir(upload_dir) if uid + '_' in f]}


@app.get("/get_matched/")
async def get_models():
    global matched
    return matched
