from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
import threading
class detectEngine:
	def __init__(self):
		self.run = False
		self._in_pipeline = ''
		self._out_pipeline = ''
		self.cap_send = None
		self.out_send = None
		self.detectThread = threading.Thread(target = self.detectLoop)
		self.detectThread.start()
		self.stopEvent = threading.Event()
		
		#initialize engine
		self.device = torch.device('cuda:0')
		self.Engine = TRTModule('yolov8s.engine', self.device)
		self.H, self.W = self.Engine.inp_info[0].shape[-2:]
		self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
	def setPipeline(self, in_pipeline, out_pipeline):
		if self.run == True:
			self.run = False
			self.stopEvent.wait()
			self.stopEvent.clear()
		self._in_pipeline = in_pipeline
		self._out_pipeline = out_pipeline
		self.cap_send = cv2.VideoCapture(self._in_pipeline, cv2.CAP_GSTREAMER)
		w = self.cap_send.get(cv2.CAP_PROP_FRAME_WIDTH)
		h = self.cap_send.get(cv2.CAP_PROP_FRAME_HEIGHT)
		fps = self.cap_send.get(cv2.CAP_PROP_FPS)
		self.out_send = cv2.VideoWriter(self._out_pipeline\
				,cv2.CAP_GSTREAMER\
				,0\
				, int(fps)\
				, (int(w), int(h))\
				, True)
		if not self.cap_send.isOpened():
			print('VideoCapture not opened')
			exit(0)
		if not self.out_send.isOpened():
			print('VideoWriter not opened')
			exit(0)
		self.run = True
		print("set run true")
	def detectLoop(self):
		while True:
			
			if self.run == False:
				continue
			else:
				self.stopEvent.clear()
			ret,frame = self.cap_send.read()
			if not ret:
				print('empty frame')
				break
			draw = frame.copy()
			frame, ratio, dwdh = letterbox(frame, (self.W, self.H))
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			tensor = blob(rgb, return_seg=False)
			dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.device)
			tensor = torch.asarray(tensor, device=self.device)
			# inference
			data = self.Engine(tensor)

			bboxes, scores, labels = det_postprocess(data)
			bboxes -= dwdh
			bboxes /= ratio

			for (bbox, score, label) in zip(bboxes, scores, labels):
				bbox = bbox.round().int().tolist()
				cls_id = int(label)
				cls = CLASSES[cls_id]
				color = COLORS[cls]
				cv2.rectangle(draw,tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
				cv2.putText(draw,\
							f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),\
							cv2.FONT_HERSHEY_SIMPLEX,\
							0.75, [225, 255, 255], thickness=2)
			if self.out_send.isOpened():
				self.out_send.write(draw)
			self.stopEvent.set()

	def __del__(self):
		if self.out_send != None:
			self.out_send.release()
		if self.cap_send != None:
			self.cap_send.release()
			
			
			
		

def main():
	
	
	in_pipeline1 = 'v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 !\
						videoconvert ! appsink'
	out_pipeline1 = 'appsrc ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! video/x-raw(memory:NVMM) ! nvv4l2h264enc !\
						rtph264pay pt=96 config-interval=1 ! udpsink host=100.73.190.7 port=5201'
	in_pipeline2 = 'v4l2src device=/dev/video2 ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 !\
						videoconvert ! appsink'
	out_pipeline2 = 'appsrc ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! video/x-raw(memory:NVMM) ! nvv4l2h264enc !\
						rtph264pay pt=96 config-interval=1 ! udpsink host=100.73.190.7 port=5202'
	#detectengine1 = detectEngine()
	detectengine2 = detectEngine()
	#detectengine1.setPipeline(in_pipeline1, out_pipeline1)
	detectengine2.setPipeline(in_pipeline2, out_pipeline2)

if __name__ == '__main__':
	main()
