from pathlib import Path

import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list

def main():
    device = torch.device('cuda:0')
    Engine = TRTModule('yolov8s.engine', device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    cap_pipeline = f'v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! videoconvert ! appsink'
    cap_send = cv2.VideoCapture(cap_pipeline, cv2.CAP_GSTREAMER)
	
    w = cap_send.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap_send.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap_send.get(cv2.CAP_PROP_FPS)
    out_send = cv2.VideoWriter('appsrc ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! video/x-raw(memory:NVMM) ! nvv4l2h264enc ! rtph264pay pt=96 config-interval=1 ! udpsink host=100.117.209.85 port=5201'\
         ,cv2.CAP_GSTREAMER\
         ,0\
         , int(fps)\
         , (int(w), int(h))\
         , True)
    if not cap_send.isOpened():
        print('VideoCapture not opened')
        exit(0)
    if not out_send.isOpened():
        print('VideoWriter not opened')
        exit(0)
    while True:
        ret,frame = cap_send.read()
        if not ret:
            print('empty frame')
            break
        draw = frame.copy()
        frame, ratio, dwdh = letterbox(frame, (W, H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
	dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw,tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255], thickness=2)
		if out_send.isOpened():
            out_send.write(draw)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    out_send.release()
    cap_send.release()

if __name__ == '__main__':
    main()
