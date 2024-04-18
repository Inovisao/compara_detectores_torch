from mmdet.apis import DetInferencer
import glob

# Choose to use a config
config = 'Detectors/MMdetection/mmdetection/configs/sabl/train.py'
# Setup a checkpoint file to load
checkpoint = glob.glob('model_checkpoints/fold_5/sabl-faster-rcnn_r50_fpn_1x_coco/epoch_3*.pth')[0]

# Set the device to be used for evaluation
device = 'cuda:0'

# Initialize the DetInferencer
inferencer = DetInferencer(config, checkpoint, device)

# Use the detector to do inference
img = '../dataset/all/train/VPC-01-frame47_jpg.rf.71e4c2f5fd0355f5b0d2c1e0bc1388cc.jpg'
result = inferencer(img, out_dir='output')
#print(result['predictions'])
print(result['predictions'])
print(result['predictions'][0]['bboxes'])
input()
bboxes2 = []
for j in range(len(result['predictions']['bboxes'])):
    for bb in result[j]:
        obj = {'x1':int(bb[0]),'x2':int(bb[2]),'y1':int(bb[1]),'y2':int(bb[3]),'score_thr':bb[4],'class':j}
print(obj)
input()