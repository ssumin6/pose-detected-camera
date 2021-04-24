import os
import io
import json
import flask
import numpy as np
import time
from PIL import Image
from flask_cors import CORS
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

app = flask.Flask(__name__)
CORS(app)

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

def inference(image):
    """Visualize the demo images.

    Require the json_file containing boxes.
    """

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

    t1 = time.time()
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
    t2 = time.time()

    # In order to save an image with pose detection result, Uncomment the following lines
    # ----------
    # out_file = f'vis_test.jpg'

    # vis_pose_result(
    #     pose_model,
    #     image,
    #     pose_results,
    #     dataset=dataset,
    #     kpt_score_thr=args.kpt_thr,
    #     show=False,
    #     out_file=out_file)
    # ----------
    print(f"4-1: {t2-t1}")
    return pose_results

####################################################
# TODO : change the each number of reference pose picture as appropriate directory
ref_pose_path = ['./img1.png', './img1.png', './img1.png', './img1.png', './img1.png']
ref_results = []

def load_ref():
    global ref_pose_path
    global ref_results

    for path in ref_pose_path:
        img = Image.open(path, 'r')
        result = inference(img)
        ref_results.append(result)

load_ref()
####################################################

def vector(vec1, vec2):
    return np.arcsin(np.cross(vec1, vec2)/(np.dot(vec1, vec1)*np.dot(vec2, vec2)))

def compare_pose(pose, ref_num: int):
    ref_keypoints = ref_results[ref_num][0]['keypoints'][:, :2] # 비교대상
    pose_keypoints = pose[0]['keypoints'][:, :2] # 사용자 인풋

    # skeleton for the limb
    # 앞 네개는 다리, 뒤 5개는 팔과 몸통. 
    # 중간 index가 0-1라인과 1-2라인의 교점
    skeleton = [[16, 14, 12], [17, 15, 13], [6, 8, 10], [8, 6, 7], [6, 7, 9], [7, 9, 11]] 

    def find_angle(skel_triples, pose_ref=True):
        if pose_ref:
            keypoints = ref_keypoints
        else:
            keypoints = pose_keypoints

        vec1 = np.array(keypoints[sk[1]-1] - keypoints[sk[0]-1])
        vec2 = np.array(keypoints[sk[2]-1] - keypoints[sk[1]-1])
        return vector(vec1, vec2)

    loss = 0.0 
    ref_angles, pose_angles = [], []

    for sk_id, sk in enumerate(skeleton):
        ref_angle = find_angle(sk)
        pose_angle = find_angle(sk, pose_ref=False)
        
        ref_angles.append(ref_angle)
        pose_angles.append(pose_angle)
        loss += np.linalg.norm(ref_angle-pose_angle)

    thres = 0.00015 # manually set
    if loss < thres: 
        return True
    else:
        return False

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

# flask app
@app.route('/')
def test():
    return 'test'

@app.route("/img", methods=["POST"])
def img():
    if flask.request.method == "POST":
        if flask.request.files.get("img"):
            t1 = time.time()
            image = flask.request.files["img"].read()
            t2 = time.time()
            ref_num = int(flask.request.values.get('type'))
            t3 = time.time()
            image = Image.open(io.BytesIO(image))
            t4 = time.time()
            pose_results = inference(image)
            t5 = time.time()
            valid_or_not = compare_pose(pose_results, ref_num)
            t6 = time.time()
            print(f"1: {t2-t1}\n2: {t3-t2}\n3: {t4-t3}\n4: {t5-t4}\n5: {t6-t5}")
            return flask.json.dumps(valid_or_not)
        return flask.json.dumps(False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
