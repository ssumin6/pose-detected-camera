import os
import io
import json
import flask
import numpy as np
from PIL import Image
from flask_cors import CORS
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

app = flask.Flask(__name__)
CORS(app)

def inference(image):
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('--pose_config', type=str, default='../configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py', help='Config file for detection')
    parser.add_argument('--pose_checkpoint', type=str, default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # make person bounding boxes
    person_results = []
    person = {}

    person['bbox'] = [0, 0, image.width, image.height] #[x, y, w, h] all of the image

    image = np.array(image)
    person_results.append(person)
   

    # test a single image, with a list of bboxes
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        image,
        person_results,
        bbox_thr=None,
        format='xywh',
        dataset=dataset,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)

    out_file = f'vis_test.jpg'

    vis_pose_result(
        pose_model,
        image,
        pose_results,
        dataset=dataset,
        kpt_score_thr=args.kpt_thr,
        show=False,
        out_file=out_file)
    return pose_results

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image.save('hello.jpg')
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/')
def test():
    return 'test'

@app.route("/img", methods=["POST"])
def img():
    if flask.request.method == "POST":
        if flask.request.files.get("img"):
            image = flask.request.files["img"].read()
            image = Image.open(io.BytesIO(image))
            pose_results = inference(image)
            print(pose_results)
            image.save('test.jpg')
            return 'hi'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
