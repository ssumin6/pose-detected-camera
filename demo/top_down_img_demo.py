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

pose_image = Image.open('./img1.jpg', 'r')
pose1_results = inference(pose_image)

def vector(vec1, vec2):
    return np.arcsin(np.cross(vec1, vec2)/(np.dot(vec1, vec1)*np.dot(vec2, vec2)))

def compare_pose(pose):
    pose1_keypoints = pose1_results[0]['keypoints'][:, :2] # 비교대상
    pose_keypoints = pose[0]['keypoints'][:, :2] # 사용자 인풋

    # skeleton for the limb
    # 앞 네개는 다리, 뒤 5개는 몸체. 
    # 중간 index가 0-1라인과 1-2라인의 교점
    skeleton = [[16, 14, 12], [17, 15, 13], [6, 8, 10], [8, 6, 7], [6, 7, 9], [7, 9, 11]]

    def find_angle(skel_triples, pose_ref=True):
        if pose_ref:
            keypoints = pose1_keypoints
        else:
            keypoints = pose_keypoints

        vec1 = np.array(keypoints[sk[1]-1] - keypoints[sk[0]-1])
        vec2 = np.array(keypoints[sk[2]-1] - keypoints[sk[1]-1])
        return vector(vec1, vec2)

    pose1_angles = []
    pose_angles = []

    loss = 0.0 

    for sk_id, sk in enumerate(skeleton):
        pose1_angle = find_angle(sk)
        pose_angle = find_angle(sk, pose_ref=False)
        
        pose1_angles.append(pose1_angle)
        pose_angles.append(pose_angle)
        loss += np.linalg.norm(pose1_angle-pose_angle)

    print(pose1_angles)
    print(pose_angles)

    thres = 0.00015
    if loss < thres: 
        print("True")
    else:
        print("False")

    return loss

    # Keypoint
    # [{'bbox': array([   0,    0, 3087, 2315]), 'keypoints': array([[7.5993756e+02, 1.1378959e+03, 7.2253174e-01],
    #    [6.7952087e+02, 1.0976875e+03, 7.6915944e-01],
    #    [6.7952087e+02, 1.1781041e+03, 7.8295064e-01],
    #    [7.1972925e+02, 1.0574791e+03, 8.1751752e-01],
    #    [7.1972925e+02, 1.2585209e+03, 8.3430207e-01],
    #    [9.2077087e+02, 9.3685419e+02, 9.2750418e-01],
    #    [9.2077087e+02, 1.4193541e+03, 8.8891232e-01],
    #    [1.2022292e+03, 6.9560419e+02, 9.7241712e-01],
    #    [1.2022292e+03, 1.6606041e+03, 9.0575814e-01],
    #    [1.4836875e+03, 8.5643750e+02, 8.7432927e-01],
    #    [1.4434791e+03, 1.4595625e+03, 8.9340419e-01],
    #    [1.6043125e+03, 1.0172708e+03, 9.3229711e-01],
    #    [1.6043125e+03, 1.3389375e+03, 8.9351946e-01],
    #    [2.2074375e+03, 1.0574791e+03, 9.1125852e-01],
    #    [2.2074375e+03, 1.4595625e+03, 9.0366358e-01],
    #    [2.7703540e+03, 1.0172708e+03, 9.4388044e-01],
    #    [2.8105625e+03, 1.5399791e+03, 9.1489053e-01]], dtype=float32)}]

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
            # print(pose_results)
            ac_loss = compare_pose(pose_results)
            print(ac_loss)
            image.save('test.jpg')
            return 'hi'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
