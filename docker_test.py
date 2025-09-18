from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import zmq
import base64
import io
from PIL import Image
from typing import List
from torchvision.ops import box_convert
import json
import torch
import numpy as np
import supervision as sv
import os

# NEW helper
def empty_response():
    return {
        "boxes": [],
        "logits": [],
        "phrases": [],
        "annotated_frame": ""  # empty string is fine for "no image"
    }


def serialize_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # Using PNG instead of JPEG
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def deserialize_image(image_data):
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {num+1}"
        for num, phrase
        in enumerate(phrases)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    # label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=0.8, text_thickness=1)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame, xyxy

def groundingdino_get_boxes(image_data, image_path, model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD):
    image = deserialize_image(image_data) if not image_path else image_path
    # # save image
    # image_array = np.array(image)
    # cv2.imwrite("received_image.jpg", image_array)
    # exit()
    image_source, image = load_image(image)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame, xyxy = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # # only get the first box
    # xyxy = [xyxy[0]]

    return xyxy, logits, phrases, annotated_frame


def main():
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    # ZeroMQ Request-Reply server setup
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5557")  # Bind to port 5557 for incoming requests

    while True:
        # Wait for a request from the client
        message = socket.recv()
        print("Received request")

        # Deserialize the request
        request = json.loads(message)
        image_data = request["image"]
        image_path = request["image_path"]
        TEXT_PROMPT = request["label"]
        if image_path and not os.path.isfile(image_path):
            socket.send_json(empty_response())
            print(f"Invalid image_path: {image_path} -> empty response sent")
            continue
        if not image_path and not image_data:
            socket.send_json(empty_response())
            print("No image supplied -> empty response sent")
            continue
        # if TEXT_PROMPT == "bottle": TEXT_PROMPT = "orange bottle"
        # if TEXT_PROMPT == "oven": TEXT_PROMPT = "green box"
        # if TEXT_PROMPT == "basket": TEXT_PROMPT = "grey basket"
        # if TEXT_PROMPT == "bread": TEXT_PROMPT = "yellow bread"
        # if TEXT_PROMPT == "cup": TEXT_PROMPT = "grey cup"
        # print("+++++++++++++++++++++++++++++++", TEXT_PROMPT)
        # image_data = deserialize_image(image_data)

        # Perform groundingdino on the image
        try:
            boxes, logits, phrases, annotated_frame = groundingdino_get_boxes(image_data, image_path, model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)
        except Exception as e:
            # Any failure in load/predict/annotate -> empty response
            socket.send_json(empty_response())
            print(f"Processing error: {e} -> empty response sent")
            continue
        # x1, y1, x2, y2 = boxes[0]
        # x1, y1, x2, y2 = int(x1)-10, int(y1)-10, int(x2)+10, int(y2)+10
        # boxes = [[x1, y1, x2, y2]]
        # logits = logits[0:1]  # Get the first box's logits
        # phrases = phrases[0:1]  # Get the first box's phrase
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1)-10, int(y1)-10, int(x2)+10, int(y2)+10
            # chekc if x1, y1, x2, y2 are within the image
            # greatest x value is width, greatest y value is height
            x2 = x2 if x2 < annotated_frame.shape[1] else annotated_frame.shape[1]
            y2 = y2 if y2 < annotated_frame.shape[0] else annotated_frame.shape[0]
            x1 = x1 if x1 > 0 else 0
            y1 = y1 if y1 > 0 else 0

            box = [x1, y1, x2, y2]

        cv2.imwrite("annotated_image.jpg", annotated_frame)
        import time
        time.sleep(5)

        # cobvert tensor list to list
        confidence_list = logits.cpu().detach().numpy().tolist()
        box_list = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes

        print(type(annotated_frame))
        annotated_frame = Image.fromarray(annotated_frame)
        annotated_frame = serialize_image(annotated_frame)

        response = {
            "boxes": box_list,
            "logits": confidence_list,
            "phrases": phrases,
            "annotated_frame": annotated_frame
        }

        socket.send_json(response)
        print("Response sent to planner")
        


if __name__ == "__main__":
    main()










# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
# # IMAGE_PATH = "/data/bobby/vstar_bench/relative_position/sa_61594.jpg"
# # IMAGE_PATH = "/home/aadhithya/bobby_wks/keyboard_guitar.png"
# # IMAGE_PATH = "/data/robot_images/2023-09-13_09-38-59/8.jpg"
# IMAGE_PATH = "/home/aadhithya/bobby_wks/output_image.jpg"
# # IMAGE_PATH = "/home/aadhithya/bobby_wks/multiple_cups.png"
# image = Image.open(IMAGE_PATH).convert("RGB")
# # bgr = True
# # # make it from rgb to bgr
# # img = np.array(image)
# # # if original image is in bgr, convert to rgb, if rgb, convert to bgr
# # if bgr:
# #     img = img[:,:,::-1] # convert to rgb
# #     img = Image.fromarray(img)
# #     img.save("image.jpg")
# #     img = cv2.imread("image.jpg")
# # cv2.imwrite("image.jpg", img)
# # exit()
# # TEXT_PROMPT = "chair . person . dog ."
# TEXT_PROMPT = "yellow bread"
# BOX_TRESHOLD = 0.35
# TEXT_TRESHOLD = 0.25

# image_source, image = load_image(image)

# boxes, logits, phrases = predict(
#     model=model,
#     image=image,
#     caption=TEXT_PROMPT,
#     box_threshold=BOX_TRESHOLD,
#     text_threshold=TEXT_TRESHOLD
# )

# # h, w, _ = image_source.shape
# # boxes = boxes * torch.Tensor([w, h, w, h])

# # print(boxes)
# # print(logits)
# # print(phrases)
# annotated_frame, xyxy = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# print(xyxy)
# print(logits)
# print(phrases)
# print(type(annotated_frame))
# # annotated_frame = Image.fromarray(annotated_frame)

# cv2.imwrite("annotated_image.jpg", annotated_frame)

# # crop annotated image to bounding box
# print(xyxy)
# image = cv2.imread(IMAGE_PATH)
# x1, y1, x2, y2 = xyxy[0]
# x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# x1, y1, x2, y2 = int(x1)-10, int(y1)-10, int(x2)+10, int(y2)+10
# # chekc if x1, y1, x2, y2 are within the image
# # greatest x value is width, greatest y value is height
# x2 = x2 if x2 < image.shape[1] else image.shape[1]
# y2 = y2 if y2 < image.shape[0] else image.shape[0]
# x1 = x1 if x1 > 0 else 0
# y1 = y1 if y1 > 0 else 0
# print(x1, y1, x2, y2)

# cropped_image = image[y1:y2, x1:x2]
# cv2.imwrite("bread.jpg", cropped_image)