from ultralytics import YOLO
import os
import torch
import torch.optim as optim

# Load a model
model = YOLO('yolo11x-cls.pt')  
#model = YOLO('codes/yolov8n-cls.pt')
# model.train(device='1')
# Train the model
results = model.train(data='/home/carlos/workspace/synapse/mechanical_parts_classifier/mechanical_parts_classifier/training',
             epochs=200, imgsz=224, batch=16, device=0, patience=180, optimizer='auto', project = 'MECHANICAL_PARTS', name='PARTS_CLASSIFIER')
# parmetros train
# patience	       50	epochs to wait for no observable improvement for early stopping of training
# batch	           16	number of images per batch (-1 for AutoBatch)
# cache	           False	True/ram, disk or False. Use cache for data loading
# optimizer	       'auto'	optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
# momentum	       0.937	SGD momentum/Adam beta1

# parmetros predict
# "visualize"		visualize model features
# "max_det"         300 maximum number of detections per image

#######  #####  #####  #  #     #  ######
   #     #   #  #      #  # #   #  #    #
   #     #####  ###    #  #  #  #  #    #
   #     # #    #      #  #   # #  #    #
   #     #   #  #####  #  #    ##  ######
os.system('nvidia-smi')
