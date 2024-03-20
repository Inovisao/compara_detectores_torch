import os

pasta_checkpoints=os.path.join(os.getcwd(),'checkpoints')

models_dict = {
    "dynamic_rcnn": {
        "dynamic-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/dynamic_rcnn_r50_fpn_1x-62a3f276.pth"
        }
    },
    "efficientnet": {
        "retinanet_effb3_fpn_8xb4-crop896-1x_coco": {
            "config_file": pasta_checkpoints + "configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py",
            "checkpoint": "/retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806-615a0dda.pth"
        }
    },
    "gn+ws": {
        "faster-rcnn_r50_fpn_gn-ws-all_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/faster-rcnn_r50_fpn_gn-ws-all_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth"
        },
        "faster-rcnn_r101_fpn_gn-ws-all_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/faster-rcnn_r101_fpn_gn-ws-all_1x_coco.py",
            "checkpoint": "/faster_rcnn_r101_fpn_gn_ws-all_1x_coco_20200205-a93b0d75.pth"
        },
        "faster-rcnn_x50-32x4d_fpn_gn-ws-all_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/faster-rcnn_x50-32x4d_fpn_gn-ws-all_1x_coco.py",
            "checkpoint": "/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco_20200203-839c5d9d.pth"
        },
        "faster-rcnn_x101-32x4d_fpn_gn-ws-all_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/faster-rcnn_x101-32x4d_fpn_gn-ws-all_1x_coco.py",
            "checkpoint": "/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco_20200212-27da1bc2.pth"
        },
        "mask-rcnn_r50_fpn_gn-ws-all_2x_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/mask-rcnn_r50_fpn_gn-ws-all_2x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_gn_ws-all_2x_coco_20200226-16acb762.pth"
        },
        "mask-rcnn_r101_fpn_gn-ws-all_2x_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/mask-rcnn_r101_fpn_gn-ws-all_2x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_gn_ws-all_2x_coco_20200212-ea357cd9.pth"
        },
        "mask-rcnn_x50-32x4d_fpn_gn-ws-all_2x_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/mask-rcnn_x50-32x4d_fpn_gn-ws-all_2x_coco.py",
            "checkpoint": "/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth"
        },
        "mask-rcnn_x101-32x4d_fpn_gn-ws-all_2x_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/mask-rcnn_x101-32x4d_fpn_gn-ws-all_2x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco_20200319-33fb95b5.pth"
        },
        "mask-rcnn_r50_fpn_gn-ws-all_20-23-24e_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/mask-rcnn_r50_fpn_gn-ws-all_20-23-24e_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco_20200213-487d1283.pth"
        },
        "mask-rcnn_r101_fpn_gn-ws-all_20-23-24e_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/mask-rcnn_r101_fpn_gn-ws-all_20-23-24e_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco_20200213-57b5a50f.pth"
        },
        "mask-rcnn_x50-32x4d_fpn_gn-ws-all_20-23-24e_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/mask-rcnn_x50-32x4d_fpn_gn-ws-all_20-23-24e_coco.py",
            "checkpoint": "/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200226-969bcb2c.pth"
        },
        "mask-rcnn_x101-32x4d_fpn_gn-ws-all_20-23-24e_coco": {
            "config_file": pasta_checkpoints + "configs/gn+ws/mask-rcnn_x101-32x4d_fpn_gn-ws-all_20-23-24e_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200316-e6cd35ef.pth"
        }
    },
    "pafpn": {
        "faster-rcnn_r50_pafpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pafpn/faster-rcnn_r50_pafpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth"
        }
    },
    "gfl": {
        "gfl_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gfl/gfl_r50_fpn_1x_coco.py",
            "checkpoint": "/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth"
        },
        "gfl_r50_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/gfl/gfl_r50_fpn_ms-2x_coco.py",
            "checkpoint": "/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth"
        },
        "gfl_r101_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/gfl/gfl_r101_fpn_ms-2x_coco.py",
            "checkpoint": "/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth"
        },
        "gfl_r101-dconv-c3-c5_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/gfl/gfl_r101-dconv-c3-c5_fpn_ms-2x_coco.py",
            "checkpoint": "/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth"
        },
        "gfl_x101-32x4d_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/gfl/gfl_x101-32x4d_fpn_ms-2x_coco.py",
            "checkpoint": "/gfl_x101_32x4d_fpn_mstrain_2x_coco_20200630_102002-50c1ffdb.pth"
        },
        "gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/gfl/gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco.py",
            "checkpoint": "/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth"
        }
    },
    "faster_rcnn": {
        "faster-rcnn_r50-caffe_c4-1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe_c4-1x_coco.py",
            "checkpoint": "/faster-rcnn_r50-caffe-c4_1x_coco"
        },
        "faster-rcnn_r50-caffe-dc5_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe-dc5_1x_coco.py",
            "checkpoint": "/faster-rcnn_r50-caffe-dc5_1x_coco"
        },
        "faster-rcnn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/faster-rcnn_r50-caffe_fpn_1x_coco"
        },
        "faster-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/faster-rcnn_r50_fpn_1x_coco"
        },
        "faster-rcnn_r50_fpn_amp-1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50_fpn_amp-1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth"
        },
        "faster-rcnn_r50_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py",
            "checkpoint": "/faster-rcnn_r50_fpn_2x_coco"
        },
        "faster-rcnn_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/faster-rcnn_r101-caffe_fpn_1x_coco"
        },
        "faster-rcnn_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py",
            "checkpoint": "/faster-rcnn_r101_fpn_1x_coco"
        },
        "faster-rcnn_r101_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py",
            "checkpoint": "/faster-rcnn_r101_fpn_2x_coco"
        },
        "faster-rcnn_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/faster-rcnn_x101-32x4d_fpn_1x_coco"
        },
        "faster-rcnn_x101-32x4d_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_x101-32x4d_fpn_2x_coco.py",
            "checkpoint": "/faster-rcnn_x101-32x4d_fpn_2x_coco"
        },
        "faster-rcnn_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/faster-rcnn_x101-64x4d_fpn_1x_coco"
        },
        "faster-rcnn_x101-64x4d_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_x101-64x4d_fpn_2x_coco.py",
            "checkpoint": "/faster-rcnn_x101-64x4d_fpn_2x_coco"
        },
        "faster-rcnn_r50_fpn_iou_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50_fpn_iou_1x_coco.py",
            "checkpoint": "/faster-rcnn_r50_fpn_iou_1x_coco"
        },
        "faster-rcnn_r50_fpn_giou_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50_fpn_giou_1x_coco.py",
            "checkpoint": "/faster-rcnn_r50_fpn_giou_1x_coco"
        },
        "faster-rcnn_r50_fpn_bounded-iou_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50_fpn_bounded-iou_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_bounded_iou_1x_coco-98ad993b.pth"
        },
        "faster-rcnn_r50-caffe-c4_ms-1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe-c4_ms-1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_caffe_c4_mstrain_1x_coco_20220316_150527-db276fed.pth"
        },
        "faster-rcnn_r50-caffe-dc5_ms-1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe-dc5_ms-1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco_20201028_233851-b33d21b9.pth"
        },
        "faster-rcnn_r50-caffe-dc5_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe-dc5_ms-3x_coco.py",
            "checkpoint": "/faster_rcnn_r50_caffe_dc5_mstrain_3x_coco_20201028_002107-34a53b2c.pth"
        },
        "faster-rcnn_r50-caffe_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-2x_coco.py",
            "checkpoint": "/faster-rcnn_r50-caffe_fpn_ms-2x_coco"
        },
        "faster-rcnn_r50-caffe_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-3x_coco.py",
            "checkpoint": "/faster-rcnn_r50-caffe_fpn_ms-3x_coco"
        },
        "faster-rcnn_r50_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50_fpn_ms-3x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth"
        },
        "faster-rcnn_r101-caffe_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r101-caffe_fpn_ms-3x_coco.py",
            "checkpoint": "/faster-rcnn_r101-caffe_fpn_ms-3x_coco"
        },
        "faster-rcnn_r101_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r101_fpn_ms-3x_coco.py",
            "checkpoint": "/faster-rcnn_r101_fpn_ms-3x_coco"
        },
        "faster-rcnn_x101-32x4d_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_x101-32x4d_fpn_ms-3x_coco.py",
            "checkpoint": "/faster-rcnn_x101-32x4d_fpn_ms-3x_coco"
        },
        "faster-rcnn_x101-32x8d_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_x101-32x8d_fpn_ms-3x_coco.py",
            "checkpoint": "/faster-rcnn_x101-32x8d_fpn_ms-3x_coco"
        },
        "faster-rcnn_x101-64x4d_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_x101-64x4d_fpn_ms-3x_coco.py",
            "checkpoint": "/faster-rcnn_x101-64x4d_fpn_ms-3x_coco"
        },
        "faster-rcnn_r50-caffe_fpn_ms-1x_coco-person": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py",
            "checkpoint": "/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
        },
        "faster-rcnn_r50-caffe_fpn_ms-1x_coco-person-bicycle-car": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person-bicycle-car.py",
            "checkpoint": "/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car_20201216_173117-6eda6d92.pth"
        },
        "faster-rcnn_r50-tnr-pre_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/faster_rcnn/faster-rcnn_r50-tnr-pre_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_tnr-pretrain_1x_coco_20220320_085147-efedfda4.pth"
        }
    },
    "yolo": {
        "yolov3_d53_8xb8-320-273e_coco": {
            "config_file": pasta_checkpoints + "configs/yolo/yolov3_d53_8xb8-320-273e_coco.py",
            "checkpoint": "/yolov3_d53_320_273e_coco-421362b6.pth"
        },
        "yolov3_d53_8xb8-ms-416-273e_coco": {
            "config_file": pasta_checkpoints + "configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py",
            "checkpoint": "/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth"
        },
        "yolov3_d53_8xb8-ms-608-273e_coco": {
            "config_file": pasta_checkpoints + "configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py",
            "checkpoint": "/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth"
        },
        "yolov3_d53_8xb8-amp-ms-608-273e_coco": {
            "config_file": pasta_checkpoints + "configs/yolo/yolov3_d53_8xb8-amp-ms-608-273e_coco.py",
            "checkpoint": "/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth"
        },
        "yolov3_mobilenetv2_8xb24-ms-416-300e_coco": {
            "config_file": pasta_checkpoints + "configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py",
            "checkpoint": "/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth"
        },
        "yolov3_mobilenetv2_8xb24-320-300e_coco": {
            "config_file": pasta_checkpoints + "configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco.py",
            "checkpoint": "/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"
        }
    },
    "yolof": {
        "yolof_r50-c5_8xb8-1x_coco": {
            "config_file": pasta_checkpoints + "configs/yolof/yolof_r50-c5_8xb8-1x_coco.py",
            "checkpoint": "/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth"
        }
    },
    "albu_example": {
        "mask-rcnn_r50_fpn_albu-1x_coco": {
            "config_file": pasta_checkpoints + "configs/albu_example/mask-rcnn_r50_fpn_albu-1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_albu_1x_coco_20200208-ab203bcd.pth"
        }
    },
    "instaboost": {
        "mask-rcnn_r50_fpn_instaboost-4x_coco": {
            "config_file": pasta_checkpoints + "configs/instaboost/mask-rcnn_r50_fpn_instaboost-4x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth"
        },
        "mask-rcnn_r101_fpn_instaboost-4x_coco": {
            "config_file": pasta_checkpoints + "configs/instaboost/mask-rcnn_r101_fpn_instaboost-4x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_instaboost_4x_coco_20200703_235738-f23f3a5f.pth"
        },
        "mask-rcnn_x101-64x4d_fpn_instaboost-4x_coco": {
            "config_file": pasta_checkpoints + "configs/instaboost/mask-rcnn_x101-64x4d_fpn_instaboost-4x_coco.py",
            "checkpoint": "/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_20200515_080947-8ed58c1b.pth"
        },
        "cascade-mask-rcnn_r50_fpn_instaboost-4x_coco": {
            "config_file": pasta_checkpoints + "configs/instaboost/cascade-mask-rcnn_r50_fpn_instaboost-4x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-c19d98d9.pth"
        }
    },
    "fcos": {
        "fcos_r50-caffe_fpn_gn-head_1x_coco": {
            "config_file": pasta_checkpoints + "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
            "checkpoint": "/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth"
        },
        "fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco": {
            "config_file": pasta_checkpoints + "configs/fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py",
            "checkpoint": "/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth"
        },
        "fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco": {
            "config_file": pasta_checkpoints + "configs/fcos/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py",
            "checkpoint": "/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth"
        },
        "fcos_r101-caffe_fpn_gn-head-1x_coco": {
            "config_file": pasta_checkpoints + "configs/fcos/fcos_r101-caffe_fpn_gn-head-1x_coco.py",
            "checkpoint": "/fcos_r101_caffe_fpn_gn-head_1x_coco-0e37b982.pth"
        },
        "fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco": {
            "config_file": pasta_checkpoints + "configs/fcos/fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco.py",
            "checkpoint": "/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth"
        },
        "fcos_r101-caffe_fpn_gn-head_ms-640-800-2x_coco": {
            "config_file": pasta_checkpoints + "configs/fcos/fcos_r101-caffe_fpn_gn-head_ms-640-800-2x_coco.py",
            "checkpoint": "/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth"
        },
        "fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco": {
            "config_file": pasta_checkpoints + "configs/fcos/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco.py",
            "checkpoint": "/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth"
        }
    },
    "crowddet": {
        "crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman": {
            "config_file": pasta_checkpoints + "configs/crowddet/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman.py",
            "checkpoint": "/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman_20221023_174954-dc319c2d.pth"
        },
        "crowddet-rcnn_refine_r50_fpn_8xb2-30e_crowdhuman": {
            "config_file": pasta_checkpoints + "configs/crowddet/crowddet-rcnn_refine_r50_fpn_8xb2-30e_crowdhuman.py",
            "checkpoint": "/crowddet-rcnn_refine_r50_fpn_8xb2-30e_crowdhuman_20221024_215917-45602806.pth"
        }
    },
    "solov2": {
        "solov2_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/solov2/solov2_r50_fpn_1x_coco.py",
            "checkpoint": "/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth"
        },
        "solov2_r50_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/solov2/solov2_r50_fpn_ms-3x_coco.py",
            "checkpoint": "/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth"
        },
        "solov2_r101_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/solov2/solov2_r101_fpn_ms-3x_coco.py",
            "checkpoint": "/solov2_r101_fpn_3x_coco_20220511_095119-c559a076.pth"
        },
        "solov2_r101-dcn_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/solov2/solov2_r101-dcn_fpn_ms-3x_coco.py",
            "checkpoint": "/solov2_r101_dcn_fpn_3x_coco_20220513_214734-16c966cb.pth"
        },
        "solov2_x101-dcn_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/solov2/solov2_x101-dcn_fpn_ms-3x_coco.py",
            "checkpoint": "/solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth"
        },
        "solov2-light_r18_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/solov2/solov2-light_r18_fpn_ms-3x_coco.py",
            "checkpoint": "/solov2_light_r18_fpn_3x_coco_20220511_083717-75fa355b.pth"
        },
        "solov2-light_r34_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/solov2/solov2-light_r34_fpn_ms-3x_coco.py",
            "checkpoint": "/solov2_light_r34_fpn_3x_coco_20220511_091839-e51659d3.pth"
        },
        "solov2-light_r50_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/solov2/solov2-light_r50_fpn_ms-3x_coco.py",
            "checkpoint": "/solov2_light_r50_fpn_3x_coco_20220512_165256-c93a6074.pth"
        }
    },
    "dab_detr": {
        "dab-detr_r50_8xb2-50e_coco": {
            "config_file": pasta_checkpoints + "configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py",
            "checkpoint": "/dab-detr_r50_8xb2-50e_coco_20221122_120837-c1035c8c.pth"
        }
    },
    "double_heads": {
        "dh-faster-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth"
        }
    },
    "dsdl": {
        "voc0712": {
            "config_file": pasta_checkpoints + "configs/dsdl/voc0712.py",
            "checkpoint": "/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth"
        },
        "coco": {
            "config_file": pasta_checkpoints + "configs/dsdl/coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        },
        "objects365v2": {
            "config_file": pasta_checkpoints + "configs/dsdl/objects365v2.py",
            "checkpoint": "/faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth"
        },
        "openimagesv6": {
            "config_file": pasta_checkpoints + "configs/dsdl/openimagesv6.py",
            "checkpoint": "/faster_rcnn_r50_fpn_32x2_cas_1x_openimages_20220306_202424-98c630e5.pth"
        },
        "coco_instance": {
            "config_file": pasta_checkpoints + "configs/dsdl/coco_instance.py",
            "checkpoint": "/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
        }
    },
    "mask2former": {
        "mask2former_r50_8xb2-lsj-50e_coco-panoptic": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py",
            "checkpoint": "/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth"
        },
        "mask2former_r101_8xb2-lsj-50e_coco-panoptic": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_r101_8xb2-lsj-50e_coco-panoptic.py",
            "checkpoint": "/mask2former_r101_8xb2-lsj-50e_coco-panoptic_20220329_225104-c74d4d71.pth"
        },
        "mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic.py",
            "checkpoint": "/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic_20220326_224553-3ec9e0ae.pth"
        },
        "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic.py",
            "checkpoint": "/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic_20220329_225200-4a16ded7.pth"
        },
        "mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic.py",
            "checkpoint": "/mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic_20220331_002244-8a651d82.pth"
        },
        "mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic.py",
            "checkpoint": "/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic_20220329_230021-05ec7315.pth"
        },
        "mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py",
            "checkpoint": "/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic_20220407_104949-82f8d28d.pth"
        },
        "mask2former_r50_8xb2-lsj-50e_coco": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py",
            "checkpoint": "/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth"
        },
        "mask2former_r101_8xb2-lsj-50e_coco": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_r101_8xb2-lsj-50e_coco.py",
            "checkpoint": "/mask2former_r101_8xb2-lsj-50e_coco_20220426_100250-ecf181e2.pth"
        },
        "mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco.py",
            "checkpoint": "/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco_20220508_091649-01b0f990.pth"
        },
        "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco": {
            "config_file": pasta_checkpoints + "configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py",
            "checkpoint": "/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
        }
    },
    "tridentnet": {},
    "detr": {
        "detr_r50_8xb2-150e_coco": {
            "config_file": pasta_checkpoints + "configs/detr/detr_r50_8xb2-150e_coco.py",
            "checkpoint": "/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"
        }
    },
    "resnest": {
        "faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco": {
            "config_file": pasta_checkpoints + "configs/resnest/faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco.py",
            "checkpoint": "/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20200926_125502-20289c16.pth"
        },
        "faster-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco": {
            "config_file": pasta_checkpoints + "configs/resnest/faster-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco.py",
            "checkpoint": "/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201006_021058-421517f1.pth"
        },
        "mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco": {
            "config_file": pasta_checkpoints + "configs/resnest/mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco.py",
            "checkpoint": "/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20200926_125503-8a2c3d47.pth"
        },
        "mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco": {
            "config_file": pasta_checkpoints + "configs/resnest/mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco.py",
            "checkpoint": "/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_215831-af60cdf9.pth"
        },
        "cascade-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco": {
            "config_file": pasta_checkpoints + "configs/resnest/cascade-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco.py",
            "checkpoint": "/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201122_213640-763cc7b5.pth"
        },
        "cascade-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco": {
            "config_file": pasta_checkpoints + "configs/resnest/cascade-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco.py",
            "checkpoint": "/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201005_113242-b9459f8f.pth"
        },
        "cascade-mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco": {
            "config_file": pasta_checkpoints + "configs/resnest/cascade-mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201122_104428-99eca4c7.pth"
        },
        "cascade-mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco": {
            "config_file": pasta_checkpoints + "configs/resnest/cascade-mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_113243-42607475.pth"
        }
    },
    "carafe": {
        "faster-rcnn_r50_fpn-carafe_1x_coco": {
            "config_file": pasta_checkpoints + "configs/carafe/faster-rcnn_r50_fpn-carafe_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth"
        },
        "mask-rcnn_r50_fpn-carafe_1x_coco": {
            "config_file": pasta_checkpoints + "configs/carafe/mask-rcnn_r50_fpn-carafe_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.393__segm_mAP-0.358_20200503_135957-8687f195.pth"
        }
    },
    "ghm": {
        "retinanet_r50_fpn_ghm-1x_coco": {
            "config_file": pasta_checkpoints + "configs/ghm/retinanet_r50_fpn_ghm-1x_coco.py",
            "checkpoint": "/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth"
        },
        "retinanet_r101_fpn_ghm-1x_coco": {
            "config_file": pasta_checkpoints + "configs/ghm/retinanet_r101_fpn_ghm-1x_coco.py",
            "checkpoint": "/retinanet_ghm_r101_fpn_1x_coco_20200130-c148ee8f.pth"
        },
        "retinanet_x101-32x4d_fpn_ghm-1x_coco": {
            "config_file": pasta_checkpoints + "configs/ghm/retinanet_x101-32x4d_fpn_ghm-1x_coco.py",
            "checkpoint": "/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131-e4333bd0.pth"
        },
        "retinanet_x101-64x4d_fpn_ghm-1x_coco": {
            "config_file": pasta_checkpoints + "configs/ghm/retinanet_x101-64x4d_fpn_ghm-1x_coco.py",
            "checkpoint": "/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131-dd381cef.pth"
        }
    },
    "sparse_rcnn": {
        "sparse-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth"
        },
        "sparse-rcnn_r50_fpn_ms-480-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/sparse_rcnn/sparse-rcnn_r50_fpn_ms-480-800-3x_coco.py",
            "checkpoint": "/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth"
        },
        "sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/sparse_rcnn/sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py",
            "checkpoint": "/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth"
        },
        "sparse-rcnn_r101_fpn_ms-480-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/sparse_rcnn/sparse-rcnn_r101_fpn_ms-480-800-3x_coco.py",
            "checkpoint": "/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth"
        },
        "sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/sparse_rcnn/sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco.py",
            "checkpoint": "/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth"
        }
    },
    "_base_": {},
    "ssd": {
        "ssd300_coco": {
            "config_file": pasta_checkpoints + "configs/ssd/ssd300_coco.py",
            "checkpoint": "/ssd300_coco_20210803_015428-d231a06e.pth"
        },
        "ssd512_coco": {
            "config_file": pasta_checkpoints + "configs/ssd/ssd512_coco.py",
            "checkpoint": "/ssd512_coco_20210803_022849-0a47a1ca.pth"
        },
        "ssdlite_mobilenetv2-scratch_8xb24-600e_coco": {
            "config_file": pasta_checkpoints + "configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py",
            "checkpoint": "/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth"
        }
    },
    "htc": {
        "htc_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/htc/htc_r50_fpn_1x_coco.py",
            "checkpoint": "/htc_r50_fpn_1x_coco_20200317-7332cf16.pth"
        },
        "htc_r50_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/htc/htc_r50_fpn_20e_coco.py",
            "checkpoint": "/htc_r50_fpn_20e_coco_20200319-fe28c577.pth"
        },
        "htc_r101_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/htc/htc_r101_fpn_20e_coco.py",
            "checkpoint": "/htc_r101_fpn_20e_coco_20200317-9b41b48f.pth"
        },
        "htc_x101-32x4d_fpn_16xb1-20e_coco": {
            "config_file": pasta_checkpoints + "configs/htc/htc_x101-32x4d_fpn_16xb1-20e_coco.py",
            "checkpoint": "/htc_x101_32x4d_fpn_16x1_20e_coco_20200318-de97ae01.pth"
        },
        "htc_x101-64x4d_fpn_16xb1-20e_coco": {
            "config_file": pasta_checkpoints + "configs/htc/htc_x101-64x4d_fpn_16xb1-20e_coco.py",
            "checkpoint": "/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth"
        },
        "htc_x101-64x4d-dconv-c3-c5_fpn_ms-400-1400-16xb1-20e_coco": {
            "config_file": pasta_checkpoints + "configs/htc/htc_x101-64x4d-dconv-c3-c5_fpn_ms-400-1400-16xb1-20e_coco.py",
            "checkpoint": "/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth"
        }
    },
    "ld": {
        "ld_r18-gflv1-r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/ld/ld_r18-gflv1-r101_fpn_1x_coco.py",
            "checkpoint": "/ld_r18_gflv1_r101_fpn_coco_1x_20220702_062206-330e6332.pth"
        },
        "ld_r34-gflv1-r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/ld/ld_r34-gflv1-r101_fpn_1x_coco.py",
            "checkpoint": "/ld_r34_gflv1_r101_fpn_coco_1x_20220630_134007-9bc69413.pth"
        },
        "ld_r50-gflv1-r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/ld/ld_r50-gflv1-r101_fpn_1x_coco.py",
            "checkpoint": "/ld_r50_gflv1_r101_fpn_coco_1x_20220629_145355-8dc5bad8.pth"
        },
        "ld_r101-gflv1-r101-dcn_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/ld/ld_r101-gflv1-r101-dcn_fpn_2x_coco.py",
            "checkpoint": "/ld_r101_gflv1_r101dcn_fpn_coco_2x_20220629_185920-9e658426.pth"
        }
    },
    "bytetrack": {},
    "paa": {
        "paa_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/paa/paa_r50_fpn_1x_coco.py",
            "checkpoint": "/paa_r50_fpn_1x_coco_20200821-936edec3.pth"
        },
        "paa_r50_fpn_1.5x_coco": {
            "config_file": pasta_checkpoints + "configs/paa/paa_r50_fpn_1.5x_coco.py",
            "checkpoint": "/paa_r50_fpn_1.5x_coco_20200823-805d6078.pth"
        },
        "paa_r50_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/paa/paa_r50_fpn_2x_coco.py",
            "checkpoint": "/paa_r50_fpn_2x_coco_20200821-c98bfc4e.pth"
        },
        "paa_r50_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/paa/paa_r50_fpn_ms-3x_coco.py",
            "checkpoint": "/paa_r50_fpn_mstrain_3x_coco_20210121_145722-06a6880b.pth"
        },
        "paa_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/paa/paa_r101_fpn_1x_coco.py",
            "checkpoint": "/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth"
        },
        "paa_r101_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/paa/paa_r101_fpn_2x_coco.py",
            "checkpoint": "/paa_r101_fpn_2x_coco_20200821-6829f96b.pth"
        },
        "paa_r101_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/paa/paa_r101_fpn_ms-3x_coco.py",
            "checkpoint": "/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth"
        }
    },
    "reppoints": {
        "reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco": {
            "config_file": pasta_checkpoints + "configs/reppoints/reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco.py",
            "checkpoint": "/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329_145916-0eedf8d1.pth"
        },
        "reppoints-bbox_r50-center_fpn-gn_head-gn-grid_1x_coco": {
            "config_file": pasta_checkpoints + "configs/reppoints/reppoints-bbox_r50-center_fpn-gn_head-gn-grid_1x_coco.py",
            "checkpoint": "/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329_145916-0eedf8d1.pth"
        },
        "reppoints-moment_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/reppoints/reppoints-moment_r50_fpn_1x_coco.py",
            "checkpoint": "/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth"
        },
        "reppoints-moment_r50_fpn-gn_head-gn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/reppoints/reppoints-moment_r50_fpn-gn_head-gn_1x_coco.py",
            "checkpoint": "/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco_20200329_145952-3e51b550.pth"
        },
        "reppoints-moment_r50_fpn-gn_head-gn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/reppoints/reppoints-moment_r50_fpn-gn_head-gn_2x_coco.py",
            "checkpoint": "/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco_20200329-91babaa2.pth"
        },
        "reppoints-moment_r101_fpn-gn_head-gn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/reppoints/reppoints-moment_r101_fpn-gn_head-gn_2x_coco.py",
            "checkpoint": "/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco_20200329-4fbc7310.pth"
        },
        "reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/reppoints/reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py",
            "checkpoint": "/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-3309fbf2.pth"
        },
        "reppoints-moment_x101-dconv-c3-c5_fpn-gn_head-gn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/reppoints/reppoints-moment_x101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py",
            "checkpoint": "/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-f87da1ea.pth"
        }
    },
    "lvis": {
        "mask-rcnn_r50_fpn_sample1e-3_ms-2x_lvis-v0.5": {
            "config_file": pasta_checkpoints + "configs/lvis/mask-rcnn_r50_fpn_sample1e-3_ms-2x_lvis-v0.5.py",
            "checkpoint": "/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis-dbd06831.pth"
        },
        "mask-rcnn_r101_fpn_sample1e-3_ms-2x_lvis-v0.5": {
            "config_file": pasta_checkpoints + "configs/lvis/mask-rcnn_r101_fpn_sample1e-3_ms-2x_lvis-v0.5.py",
            "checkpoint": "/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis-54582ee2.pth"
        },
        "mask-rcnn_x101-32x4d_fpn_sample1e-3_ms-2x_lvis-v0.5": {
            "config_file": pasta_checkpoints + "configs/lvis/mask-rcnn_x101-32x4d_fpn_sample1e-3_ms-2x_lvis-v0.5.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis-3cf55ea2.pth"
        },
        "mask-rcnn_x101-64x4d_fpn_sample1e-3_ms-2x_lvis-v0.5": {
            "config_file": pasta_checkpoints + "configs/lvis/mask-rcnn_x101-64x4d_fpn_sample1e-3_ms-2x_lvis-v0.5.py",
            "checkpoint": "/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis-1c99a5ad.pth"
        },
        "mask-rcnn_r50_fpn_sample1e-3_ms-1x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/lvis/mask-rcnn_r50_fpn_sample1e-3_ms-1x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth"
        },
        "mask-rcnn_r101_fpn_sample1e-3_ms-1x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/lvis/mask-rcnn_r101_fpn_sample1e-3_ms-1x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-ec55ce32.pth"
        },
        "mask-rcnn_x101-32x4d_fpn_sample1e-3_ms-1x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/lvis/mask-rcnn_x101-32x4d_fpn_sample1e-3_ms-1x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth"
        },
        "mask-rcnn_x101-64x4d_fpn_sample1e-3_ms-1x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/lvis/mask-rcnn_x101-64x4d_fpn_sample1e-3_ms-1x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth"
        }
    },
    "simple_copy_paste": {
        "mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_32xb2-ssj-90k_coco": {
            "config_file": pasta_checkpoints + "configs/simple_copy_paste/mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_32xb2-ssj-90k_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco_20220316_181409-f79c84c5.pth"
        },
        "mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_32xb2-ssj-scp-90k_coco": {
            "config_file": pasta_checkpoints + "configs/simple_copy_paste/mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_32xb2-ssj-scp-90k_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco_20220316_181307-6bc5726f.pth"
        },
        "mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_32xb2-ssj-270k_coco": {
            "config_file": pasta_checkpoints + "configs/simple_copy_paste/mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_32xb2-ssj-270k_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco_20220324_182940-33a100c5.pth"
        },
        "mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_32xb2-ssj-scp-270k_coco": {
            "config_file": pasta_checkpoints + "configs/simple_copy_paste/mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_32xb2-ssj-scp-270k_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco_20220324_201229-80ee90b7.pth"
        }
    },
    "cascade_rcnn": {
        "cascade-rcnn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/cascade_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.404_20200504_174853-b857be87.pth"
        },
        "cascade-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"
        },
        "cascade-rcnn_r50_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_r50_fpn_20e_coco.py",
            "checkpoint": "/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth"
        },
        "cascade-rcnn_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/cascade_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.423_20200504_175649-cab8dbd5.pth"
        },
        "cascade-rcnn_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_r101_fpn_1x_coco.py",
            "checkpoint": "/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth"
        },
        "cascade-rcnn_r101_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_r101_fpn_20e_coco.py",
            "checkpoint": "/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth"
        },
        "cascade-rcnn_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316-95c2deb6.pth"
        },
        "cascade-rcnn_x101-32x4d_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_x101-32x4d_fpn_20e_coco.py",
            "checkpoint": "/cascade_rcnn_x101_32x4d_fpn_20e_coco_20200906_134608-9ae0a720.pth"
        },
        "cascade-rcnn_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth"
        },
        "cascade-rcnn_x101_64x4d_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-rcnn_x101_64x4d_fpn_20e_coco.py",
            "checkpoint": "/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth"
        },
        "cascade-mask-rcnn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.412__segm_mAP-0.36_20200504_174659-5004b251.pth"
        },
        "cascade-mask-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
        },
        "cascade-mask-rcnn_r50_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_20e_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r50_fpn_20e_coco_bbox_mAP-0.419__segm_mAP-0.365_20200504_174711-4af8e66e.pth"
        },
        "cascade-mask-rcnn_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.432__segm_mAP-0.376_20200504_174813-5c1e9599.pth"
        },
        "cascade-mask-rcnn_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r101_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth"
        },
        "cascade-mask-rcnn_r101_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r101_fpn_20e_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r101_fpn_20e_coco_bbox_mAP-0.434__segm_mAP-0.378_20200504_174836-005947da.pth"
        },
        "cascade-mask-rcnn_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201-0f411b1f.pth"
        },
        "cascade-mask-rcnn_x101-32x4d_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_x101-32x4d_fpn_20e_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco_20200528_083917-ed1f4751.pth"
        },
        "cascade-mask-rcnn_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth"
        },
        "cascade-mask-rcnn_x101-64x4d_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_x101-64x4d_fpn_20e_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth"
        },
        "cascade-mask-rcnn_r50-caffe_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r50-caffe_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210707_002651-6e29b3a6.pth"
        },
        "cascade-mask-rcnn_r50_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco_20210628_164719-5bdc3824.pth"
        },
        "cascade-mask-rcnn_r101-caffe_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r101-caffe_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r101_caffe_fpn_mstrain_3x_coco_20210707_002620-a5bd2389.pth"
        },
        "cascade-mask-rcnn_r101_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_r101_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco_20210628_165236-51a2d363.pth"
        },
        "cascade-mask-rcnn_x101-32x4d_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_x101-32x4d_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco_20210706_225234-40773067.pth"
        },
        "cascade-mask-rcnn_x101-32x8d_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_x101-32x8d_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x8d_fpn_mstrain_3x_coco_20210719_180640-9ff7e76f.pth"
        },
        "cascade-mask-rcnn_x101-64x4d_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rcnn/cascade-mask-rcnn_x101-64x4d_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth"
        }
    },
    "dcn": {
        "faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth"
        },
        "faster-rcnn_r50_fpn_dpool_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/faster-rcnn_r50_fpn_dpool_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_dpool_1x_coco_20200307-90d3c01d.pth"
        },
        "faster-rcnn_r101-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/faster-rcnn_r101-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth"
        },
        "faster-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/faster-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth"
        },
        "mask-rcnn_r50-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/mask-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203-4d9ad43b.pth"
        },
        "mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth"
        },
        "cascade-rcnn_r50-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/cascade-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth"
        },
        "cascade-rcnn_r101-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/cascade-rcnn_r101-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth"
        },
        "cascade-mask-rcnn_r50-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/cascade-mask-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202-42e767a2.pth"
        },
        "cascade-mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/cascade-mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200204-df0c5f10.pth"
        },
        "cascade-mask-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/cascade-mask-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth"
        },
        "mask-rcnn_r50-dconv-c3-c5_fpn_amp-1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcn/mask-rcnn_r50-dconv-c3-c5_fpn_amp-1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco_20210520_180247-c06429d2.pth"
        }
    },
    "autoassign": {
        "autoassign_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth"
        }
    },
    "mm_grounding_dino": {
        "grounding_dino": {
            "config_file": pasta_checkpoints + "configs/mm_grounding_dino./grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py",
            "checkpoint": "/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
        }
    },
    "tood": {
        "tood_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/tood/tood_r50_fpn_1x_coco.py",
            "checkpoint": "/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth"
        },
        "tood_r50_fpn_anchor-based_1x_coco": {
            "config_file": pasta_checkpoints + "configs/tood/tood_r50_fpn_anchor-based_1x_coco.py",
            "checkpoint": "/tood_r50_fpn_anchor_based_1x_coco_20211214_100105-b776c134.pth"
        },
        "tood_r50_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/tood/tood_r50_fpn_ms-2x_coco.py",
            "checkpoint": "/tood_r50_fpn_mstrain_2x_coco_20211210_144231-3b23174c.pth"
        },
        "tood_r101_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/tood/tood_r101_fpn_ms-2x_coco.py",
            "checkpoint": "/tood_r101_fpn_mstrain_2x_coco_20211210_144232-a18f53c8.pth"
        },
        "tood_r101-dconv-c3-c5_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/tood/tood_r101-dconv-c3-c5_fpn_ms-2x_coco.py",
            "checkpoint": "/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20211210_213728-4a824142.pth"
        },
        "tood_x101-64x4d_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/tood/tood_x101-64x4d_fpn_ms-2x_coco.py",
            "checkpoint": "/tood_x101_64x4d_fpn_mstrain_2x_coco_20211211_003519-a4f36113.pth"
        },
        "tood_x101-64x4d-dconv-c4-c5_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/tood/tood_x101-64x4d-dconv-c4-c5_fpn_ms-2x_coco.py",
            "checkpoint": "/<a href=\""
        }
    },
    "selfsup_pretrain": {
        "mask-rcnn_r50-mocov2-pre_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/selfsup_pretrain/mask-rcnn_r50-mocov2-pre_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_mocov2-pretrain_1x_coco_20210604_114614-a8b63483.pth"
        },
        "mask-rcnn_r50-mocov2-pre_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/selfsup_pretrain/mask-rcnn_r50-mocov2-pre_fpn_ms-2x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_mocov2-pretrain_ms-2x_coco_20210605_163717-d95df20a.pth"
        },
        "mask-rcnn_r50-swav-pre_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/selfsup_pretrain/mask-rcnn_r50-swav-pre_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_swav-pretrain_1x_coco_20210604_114640-7b9baf28.pth"
        },
        "mask-rcnn_r50-swav-pre_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/selfsup_pretrain/mask-rcnn_r50-swav-pre_fpn_ms-2x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_swav-pretrain_ms-2x_coco_20210605_163717-08e26fca.pth"
        }
    },
    "solo": {},
    "v3det": {
        "faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x": {
            "config_file": pasta_checkpoints + "configs/v3det/faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x.py",
            "checkpoint": "/faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x"
        },
        "cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x": {
            "config_file": pasta_checkpoints + "configs/v3det/cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x.py",
            "checkpoint": "/cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x"
        },
        "fcos_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x": {
            "config_file": pasta_checkpoints + "configs/v3det/fcos_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x.py",
            "checkpoint": "/fcos_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x"
        },
        "deformable-detr-refine-twostage_r50_8xb4_sample1e-3_v3det_50e": {
            "config_file": pasta_checkpoints + "configs/v3det/deformable-detr-refine-twostage_r50_8xb4_sample1e-3_v3det_50e.py",
            "checkpoint": "/Deformable_DETR_V3Det_R50"
        },
        "dino-4scale_r50_8xb2_sample1e-3_v3det_36e": {
            "config_file": pasta_checkpoints + "configs/v3det/dino-4scale_r50_8xb2_sample1e-3_v3det_36e.py",
            "checkpoint": "/DINO_V3Det_R50"
        },
        "faster_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x": {
            "config_file": pasta_checkpoints + "configs/v3det/faster_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x.py",
            "checkpoint": "/faster_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x"
        },
        "cascade_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x": {
            "config_file": pasta_checkpoints + "configs/v3det/cascade_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x.py",
            "checkpoint": "/cascade_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x"
        },
        "fcos_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x": {
            "config_file": pasta_checkpoints + "configs/v3det/fcos_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x.py",
            "checkpoint": "/fcos_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x"
        },
        "deformable-detr-refine-twostage_swin_16xb2_sample1e-3_v3det_50e": {
            "config_file": pasta_checkpoints + "configs/v3det/deformable-detr-refine-twostage_swin_16xb2_sample1e-3_v3det_50e.py",
            "checkpoint": "/Deformable_DETR_V3Det_SwinB"
        },
        "dino-4scale_swin_16xb1_sample1e-3_v3det_36e": {
            "config_file": pasta_checkpoints + "configs/v3det/dino-4scale_swin_16xb1_sample1e-3_v3det_36e.py",
            "checkpoint": "/DINO_V3Det_SwinB"
        }
    },
    "foveabox": {
        "fovea_r50_fpn_4xb4-1x_coco": {
            "config_file": pasta_checkpoints + "configs/foveabox/fovea_r50_fpn_4xb4-1x_coco.py",
            "checkpoint": "/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth"
        },
        "fovea_r50_fpn_4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/foveabox/fovea_r50_fpn_4xb4-2x_coco.py",
            "checkpoint": "/fovea_r50_fpn_4x4_2x_coco_20200203-2df792b1.pth"
        },
        "fovea_r50_fpn_gn-head-align_4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/foveabox/fovea_r50_fpn_gn-head-align_4xb4-2x_coco.py",
            "checkpoint": "/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth"
        },
        "fovea_r50_fpn_gn-head-align_ms-640-800-4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/foveabox/fovea_r50_fpn_gn-head-align_ms-640-800-4xb4-2x_coco.py",
            "checkpoint": "/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205-85ce26cb.pth"
        },
        "fovea_r101_fpn_4xb4-1x_coco": {
            "config_file": pasta_checkpoints + "configs/foveabox/fovea_r101_fpn_4xb4-1x_coco.py",
            "checkpoint": "/fovea_r101_fpn_4x4_1x_coco_20200219-05e38f1c.pth"
        },
        "fovea_r101_fpn_4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/foveabox/fovea_r101_fpn_4xb4-2x_coco.py",
            "checkpoint": "/fovea_r101_fpn_4x4_2x_coco_20200208-02320ea4.pth"
        },
        "fovea_r101_fpn_gn-head-align_4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/foveabox/fovea_r101_fpn_gn-head-align_4xb4-2x_coco.py",
            "checkpoint": "/fovea_align_r101_fpn_gn-head_4x4_2x_coco_20200208-c39a027a.pth"
        },
        "fovea_r101_fpn_gn-head-align_ms-640-800-4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/foveabox/fovea_r101_fpn_gn-head-align_ms-640-800-4xb4-2x_coco.py",
            "checkpoint": "/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208-649c5eb6.pth"
        }
    },
    "condinst": {
        "condinst_r50_fpn_ms-poly-90k_coco_instance": {
            "config_file": pasta_checkpoints + "configs/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py",
            "checkpoint": "/condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth"
        }
    },
    "misc": {},
    "nas_fcos": {
        "nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco": {
            "config_file": pasta_checkpoints + "configs/nas_fcos/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco.py",
            "checkpoint": "/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth"
        },
        "nas-fcos_r50-caffe_fpn_fcoshead-gn-head_4xb4-1x_coco": {
            "config_file": pasta_checkpoints + "configs/nas_fcos/nas-fcos_r50-caffe_fpn_fcoshead-gn-head_4xb4-1x_coco.py",
            "checkpoint": "/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200521-7fdcbce0.pth"
        }
    },
    "centernet": {
        "centernet_r18_8xb16-crop512-140e_coco": {
            "config_file": pasta_checkpoints + "configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py",
            "checkpoint": "/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth"
        },
        "centernet_r18-dcnv2_8xb16-crop512-140e_coco": {
            "config_file": pasta_checkpoints + "configs/centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py",
            "checkpoint": "/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
        },
        "centernet-update_r50-caffe_fpn_ms-1x_coco": {
            "config_file": pasta_checkpoints + "configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py",
            "checkpoint": "/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth"
        }
    },
    "libra_rcnn": {
        "libra-faster-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth"
        },
        "libra-faster-rcnn_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/libra_rcnn/libra-faster-rcnn_r101_fpn_1x_coco.py",
            "checkpoint": "/libra_faster_rcnn_r101_fpn_1x_coco_20200203-8dba6a5a.pth"
        },
        "libra-faster-rcnn_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/libra_rcnn/libra-faster-rcnn_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315-3a7d0488.pth"
        },
        "libra-retinanet_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/libra_rcnn/libra-retinanet_r50_fpn_1x_coco.py",
            "checkpoint": "/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth"
        }
    },
    "deformable_detr": {
        "deformable-detr_r50_16xb2-50e_coco": {
            "config_file": pasta_checkpoints + "configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py",
            "checkpoint": "/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth"
        },
        "deformable-detr-refine_r50_16xb2-50e_coco": {
            "config_file": pasta_checkpoints + "configs/deformable_detr/deformable-detr-refine_r50_16xb2-50e_coco.py",
            "checkpoint": "/deformable-detr-refine_r50_16xb2-50e_coco_20221022_225303-844e0f93.pth"
        },
        "deformable-detr-refine-twostage_r50_16xb2-50e_coco": {
            "config_file": pasta_checkpoints + "configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py",
            "checkpoint": "/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth"
        }
    },
    "mask2former_vis": {},
    "soft_teacher": {
        "soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.01-coco": {
            "config_file": pasta_checkpoints + "configs/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.01-coco.py",
            "checkpoint": "/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230330_233412-3c8f6d4a.pth"
        },
        "soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.02-coco": {
            "config_file": pasta_checkpoints + "configs/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.02-coco.py",
            "checkpoint": "/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230331_020244-c0d2c3aa.pth"
        },
        "soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.05-coco": {
            "config_file": pasta_checkpoints + "configs/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.05-coco.py",
            "checkpoint": "/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230331_070656-308798ad.pth"
        },
        "soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco": {
            "config_file": pasta_checkpoints + "configs/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py",
            "checkpoint": "/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230330_232113-b46f78d0.pth"
        }
    },
    "seesaw_loss": {
        "mask-rcnn_r50_fpn_seesaw-loss_random-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/mask-rcnn_r50_fpn_seesaw-loss_random-ms-2x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r50_fpn_random_seesaw_loss_mstrain_2x_lvis_v1-a698dd3d.pth"
        },
        "mask-rcnn_r50_fpn_seesaw-loss-normed-mask_random-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/mask-rcnn_r50_fpn_seesaw-loss-normed-mask_random-ms-2x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-a1c11314.pth"
        },
        "mask-rcnn_r101_fpn_seesaw-loss_random-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/mask-rcnn_r101_fpn_seesaw-loss_random-ms-2x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1-8e6e6dd5.pth"
        },
        "mask-rcnn_r101_fpn_seesaw-loss-normed-mask_random-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/mask-rcnn_r101_fpn_seesaw-loss-normed-mask_random-ms-2x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-a0b59c42.pth"
        },
        "mask-rcnn_r50_fpn_seesaw-loss_sample1e-3-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/mask-rcnn_r50_fpn_seesaw-loss_sample1e-3-ms-2x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1-392a804b.pth"
        },
        "mask-rcnn_r50_fpn_seesaw-loss-normed-mask_sample1e-3-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/mask-rcnn_r50_fpn_seesaw-loss-normed-mask_sample1e-3-ms-2x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-cd0f6a12.pth"
        },
        "mask-rcnn_r101_fpn_seesaw-loss_sample1e-3-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/mask-rcnn_r101_fpn_seesaw-loss_sample1e-3-ms-2x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1-e68eb464.pth"
        },
        "mask-rcnn_r101_fpn_seesaw-loss-normed-mask_sample1e-3-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/mask-rcnn_r101_fpn_seesaw-loss-normed-mask_sample1e-3-ms-2x_lvis-v1.py",
            "checkpoint": "/mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-1d817139.pth"
        },
        "cascade-mask-rcnn_r101_fpn_seesaw-loss_random-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/cascade-mask-rcnn_r101_fpn_seesaw-loss_random-ms-2x_lvis-v1.py",
            "checkpoint": "/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1-71e2215e.pth"
        },
        "cascade-mask-rcnn_r101_fpn_seesaw-loss-normed-mask_random-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/cascade-mask-rcnn_r101_fpn_seesaw-loss-normed-mask_random-ms-2x_lvis-v1.py",
            "checkpoint": "/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-8b5a6745.pth"
        },
        "cascade-mask-rcnn_r101_fpn_seesaw-loss_sample1e-3-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/cascade-mask-rcnn_r101_fpn_seesaw-loss_sample1e-3-ms-2x_lvis-v1.py",
            "checkpoint": "/cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1-5d8ca2a4.pth"
        },
        "cascade-mask-rcnn_r101_fpn_seesaw-loss-normed-mask_sample1e-3-ms-2x_lvis-v1": {
            "config_file": pasta_checkpoints + "configs/seesaw_loss/cascade-mask-rcnn_r101_fpn_seesaw-loss-normed-mask_sample1e-3-ms-2x_lvis-v1.py",
            "checkpoint": "/cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-c8551505.pth"
        }
    },
    "pascal_voc": {
        "faster-rcnn_r50-caffe-c4_ms-18k_voc0712": {
            "config_file": pasta_checkpoints + "configs/pascal_voc/faster-rcnn_r50-caffe-c4_ms-18k_voc0712.py",
            "checkpoint": "/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712_20220314_234327-847a14d2.pth"
        },
        "faster-rcnn_r50_fpn_1x_voc0712": {
            "config_file": pasta_checkpoints + "configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py",
            "checkpoint": "/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth"
        },
        "retinanet_r50_fpn_1x_voc0712": {
            "config_file": pasta_checkpoints + "configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py",
            "checkpoint": "/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth"
        },
        "ssd300_voc0712": {
            "config_file": pasta_checkpoints + "configs/pascal_voc/ssd300_voc0712.py",
            "checkpoint": "/ssd300_voc0712_20220320_194658-17edda1b.pth"
        },
        "ssd512_voc0712": {
            "config_file": pasta_checkpoints + "configs/pascal_voc/ssd512_voc0712.py",
            "checkpoint": "/ssd512_voc0712_20220320_194717-03cefefe.pth"
        }
    },
    "centripetalnet": {
        "centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco": {
            "config_file": pasta_checkpoints + "configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py",
            "checkpoint": "/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth"
        }
    },
    "atss": {
        "atss_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/atss/atss_r50_fpn_1x_coco.py",
            "checkpoint": "/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth"
        },
        "atss_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/atss/atss_r101_fpn_1x_coco.py",
            "checkpoint": "/atss_r101_fpn_1x_20200825-dfcadd6f.pth"
        }
    },
    "gcnet": {
        "mask-rcnn_r50-gcb-r16-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r50-gcb-r16-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco_20200515_211915-187da160.pth"
        },
        "mask-rcnn_r50-gcb-r4-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r50-gcb-r4-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth"
        },
        "mask-rcnn_r101-gcb-r16-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r101-gcb-r16-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco_20200205-e58ae947.pth"
        },
        "mask-rcnn_r101-gcb-r4-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r101-gcb-r4-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco_20200206-af22dc9d.pth"
        },
        "mask-rcnn_r50-syncbn_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r50-syncbn_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco_20200202-bb3eb55c.pth"
        },
        "mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202-587b99aa.pth"
        },
        "mask-rcnn_r50-syncbn-gcb-r4-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r50-syncbn-gcb-r4-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202-50b90e5c.pth"
        },
        "mask-rcnn_r101-syncbn_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r101-syncbn_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco_20200210-81658c8a.pth"
        },
        "mask-rcnn_r101-syncbn-gcb-r16-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r101-syncbn-gcb-r16-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200207-945e77ca.pth"
        },
        "mask-rcnn_r101-syncbn-gcb-r4-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_r101-syncbn-gcb-r4-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206-8407a3f0.pth"
        },
        "mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200211-7584841c.pth"
        },
        "mask-rcnn_x101-32x4d-syncbn-gcb-r16-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_x101-32x4d-syncbn-gcb-r16-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211-cbed3d2c.pth"
        },
        "mask-rcnn_x101-32x4d-syncbn-gcb-r4-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/mask-rcnn_x101-32x4d-syncbn-gcb-r4-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200212-68164964.pth"
        },
        "cascade-mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/cascade-mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200310-d5ad2a5e.pth"
        },
        "cascade-mask-rcnn_x101-32x4d-syncbn-r16-gcb-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/cascade-mask-rcnn_x101-32x4d-syncbn-r16-gcb-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211-10bf2463.pth"
        },
        "cascade-mask-rcnn_x101-32x4d-syncbn-r4-gcb-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/cascade-mask-rcnn_x101-32x4d-syncbn-r4-gcb-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200703_180653-ed035291.pth"
        },
        "cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco_20210615_211019-abbc39ea.pth"
        },
        "cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r16-gcb-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r16-gcb-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco_20210615_215648-44aa598a.pth"
        },
        "cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r4-gcb-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/gcnet/cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r4-gcb-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20210615_161851-720338ec.pth"
        }
    },
    "maskformer": {
        "maskformer_r50_ms-16xb1-75e_coco": {
            "config_file": pasta_checkpoints + "configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py",
            "checkpoint": "/maskformer_r50_ms-16xb1-75e_coco_20230116_095226-baacd858.pth"
        },
        "maskformer_swin-l-p4-w12_64xb1-ms-300e_coco": {
            "config_file": pasta_checkpoints + "configs/maskformer/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py",
            "checkpoint": "/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth"
        }
    },
    "free_anchor": {
        "freeanchor_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/free_anchor/freeanchor_r50_fpn_1x_coco.py",
            "checkpoint": "/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth"
        },
        "freeanchor_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/free_anchor/freeanchor_r101_fpn_1x_coco.py",
            "checkpoint": "/retinanet_free_anchor_r101_fpn_1x_coco_20200130-358324e6.pth"
        },
        "freeanchor_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/free_anchor/freeanchor_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130-d4846968.pth"
        }
    },
    "guided_anchoring": {
        "ga-rpn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-rpn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/ga_rpn_r50_caffe_fpn_1x_coco_20200531-899008a6.pth"
        },
        "ga-rpn_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-rpn_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/ga_rpn_r101_caffe_fpn_1x_coco_20200531-ca9ba8fb.pth"
        },
        "ga-rpn_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-rpn_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth"
        },
        "ga-rpn_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-rpn_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/ga_rpn_x101_64x4d_fpn_1x_coco_20200225-3c6e1aa2.pth"
        },
        "ga-faster-rcnn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-faster-rcnn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718-a11ccfe6.pth"
        },
        "ga-faster-rcnn_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-faster-rcnn_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/ga_faster_r101_caffe_fpn_1x_coco_bbox_mAP-0.415_20200505_115528-fb82e499.pth"
        },
        "ga-faster-rcnn_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-faster-rcnn_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/ga_faster_x101_32x4d_fpn_1x_coco_20200215-1ded9da3.pth"
        },
        "ga-faster-rcnn_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-faster-rcnn_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/ga_faster_x101_64x4d_fpn_1x_coco_20200215-0fa7bde7.pth"
        },
        "ga-retinanet_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-retinanet_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth"
        },
        "ga-retinanet_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-retinanet_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/ga_retinanet_r101_caffe_fpn_1x_coco_20200531-6266453c.pth"
        },
        "ga-retinanet_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-retinanet_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219-40c56caa.pth"
        },
        "ga-retinanet_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/guided_anchoring/ga-retinanet_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226-ef9f7f1f.pth"
        }
    },
    "dino": {
        "dino-4scale_r50_8xb2-12e_coco": {
            "config_file": pasta_checkpoints + "configs/dino/dino-4scale_r50_8xb2-12e_coco.py",
            "checkpoint": "/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"
        },
        "dino-4scale_r50_improved_8xb2-12e_coco": {
            "config_file": pasta_checkpoints + "configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py",
            "checkpoint": "/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"
        },
        "dino-5scale_swin-l_8xb2-12e_coco": {
            "config_file": pasta_checkpoints + "configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py",
            "checkpoint": "/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"
        },
        "dino-5scale_swin-l_8xb2-36e_coco": {
            "config_file": pasta_checkpoints + "configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py",
            "checkpoint": "/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
        }
    },
    "retinanet": {
        "retinanet_r18_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r18_fpn_1x_coco.py",
            "checkpoint": "/retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth"
        },
        "retinanet_r18_fpn_1xb8-1x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r18_fpn_1xb8-1x_coco.py",
            "checkpoint": "/retinanet_r18_fpn_1x8_1x_coco_20220407_171255-4ea310d7.pth"
        },
        "retinanet_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth"
        },
        "retinanet_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
            "checkpoint": "/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"
        },
        "retinanet_r50_fpn_amp-1x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r50_fpn_amp-1x_coco.py",
            "checkpoint": "/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth"
        },
        "retinanet_r50_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r50_fpn_2x_coco.py",
            "checkpoint": "/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
        },
        "retinanet_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/retinanet_r101_caffe_fpn_1x_coco_20200531-b428fa0f.pth"
        },
        "retinanet_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r101_fpn_1x_coco.py",
            "checkpoint": "/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"
        },
        "retinanet_r101_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r101_fpn_2x_coco.py",
            "checkpoint": "/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth"
        },
        "retinanet_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth"
        },
        "retinanet_x101-32x4d_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_x101-32x4d_fpn_2x_coco.py",
            "checkpoint": "/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth"
        },
        "retinanet_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
        },
        "retinanet_x101-64x4d_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_x101-64x4d_fpn_2x_coco.py",
            "checkpoint": "/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth"
        },
        "retinanet_r50_fpn_ms-640-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r50_fpn_ms-640-800-3x_coco.py",
            "checkpoint": "/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth"
        },
        "retinanet_r101-caffe_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r101-caffe_fpn_ms-3x_coco.py",
            "checkpoint": "/retinanet_r101_caffe_fpn_mstrain_3x_coco_20210721_063439-88a8a944.pth"
        },
        "retinanet_r101_fpn_ms-640-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_r101_fpn_ms-640-800-3x_coco.py",
            "checkpoint": "/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.pth"
        },
        "retinanet_x101-64x4d_fpn_ms-640-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/retinanet/retinanet_x101-64x4d_fpn_ms-640-800-3x_coco.py",
            "checkpoint": "/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth"
        }
    },
    "pisa": {
        "faster-rcnn_r50_fpn_pisa_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pisa/faster-rcnn_r50_fpn_pisa_1x_coco.py",
            "checkpoint": "/pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth"
        },
        "faster-rcnn_x101-32x4d_fpn_pisa_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pisa/faster-rcnn_x101-32x4d_fpn_pisa_1x_coco.py",
            "checkpoint": "/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco-e4accec4.pth"
        },
        "mask-rcnn_r50_fpn_pisa_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pisa/mask-rcnn_r50_fpn_pisa_1x_coco.py",
            "checkpoint": "/pisa_mask_rcnn_r50_fpn_1x_coco-dfcedba6.pth"
        },
        "retinanet-r50_fpn_pisa_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pisa/retinanet-r50_fpn_pisa_1x_coco.py",
            "checkpoint": "/pisa_retinanet_r50_fpn_1x_coco-76409952.pth"
        },
        "retinanet_x101-32x4d_fpn_pisa_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pisa/retinanet_x101-32x4d_fpn_pisa_1x_coco.py",
            "checkpoint": "/pisa_retinanet_x101_32x4d_fpn_1x_coco-a0c13c73.pth"
        },
        "ssd300_pisa_coco": {
            "config_file": pasta_checkpoints + "configs/pisa/ssd300_pisa_coco.py",
            "checkpoint": "/pisa_ssd300_coco-710e3ac9.pth"
        },
        "ssd512_pisa_coco": {
            "config_file": pasta_checkpoints + "configs/pisa/ssd512_pisa_coco.py",
            "checkpoint": "/pisa_ssd512_coco-247addee.pth"
        }
    },
    "timm_example": {
        "retinanet_timm-tv-resnet50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/timm_example/retinanet_timm-tv-resnet50_fpn_1x_coco.py",
            "checkpoint": "/"
        },
        "retinanet_timm-efficientnet-b1_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/timm_example/retinanet_timm-efficientnet-b1_fpn_1x_coco.py",
            "checkpoint": "/"
        }
    },
    "yolact": {
        "yolact_r50_1xb8-55e_coco": {
            "config_file": pasta_checkpoints + "configs/yolact/yolact_r50_1xb8-55e_coco.py",
            "checkpoint": "/yolact_r50_1x8_coco_20200908-f38d58df.pth"
        },
        "yolact_r50_8xb8-55e_coco": {
            "config_file": pasta_checkpoints + "configs/yolact/yolact_r50_8xb8-55e_coco.py",
            "checkpoint": "/yolact_r50_8x8_coco_20200908-ca34f5db.pth"
        },
        "yolact_r101_1xb8-55e_coco": {
            "config_file": pasta_checkpoints + "configs/yolact/yolact_r101_1xb8-55e_coco.py",
            "checkpoint": "/yolact_r101_1x8_coco_20200908-4cbe9101.pth"
        }
    },
    "deepsort": {},
    "ddod": {
        "ddod_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/ddod/ddod_r50_fpn_1x_coco.py",
            "checkpoint": "/ddod_r50_fpn_1x_coco_20220523_223737-29b2fc67.pth"
        }
    },
    "vfnet": {
        "vfnet_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/vfnet/vfnet_r50_fpn_1x_coco.py",
            "checkpoint": "/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth"
        },
        "vfnet_r50_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/vfnet/vfnet_r50_fpn_ms-2x_coco.py",
            "checkpoint": "/vfnet_r50_fpn_mstrain_2x_coco_20201027-7cc75bd2.pth"
        },
        "vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/vfnet/vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco.py",
            "checkpoint": "/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth"
        },
        "vfnet_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/vfnet/vfnet_r101_fpn_1x_coco.py",
            "checkpoint": "/vfnet_r101_fpn_1x_coco_20201027pth-c831ece7.pth"
        },
        "vfnet_r101_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/vfnet/vfnet_r101_fpn_ms-2x_coco.py",
            "checkpoint": "/vfnet_r101_fpn_mstrain_2x_coco_20201027pth-4a5d53f1.pth"
        },
        "vfnet_r101-mdconv-c3-c5_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/vfnet/vfnet_r101-mdconv-c3-c5_fpn_ms-2x_coco.py",
            "checkpoint": "/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth"
        },
        "vfnet_x101-32x4d-mdconv-c3-c5_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/vfnet/vfnet_x101-32x4d-mdconv-c3-c5_fpn_ms-2x_coco.py",
            "checkpoint": "/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-d300a6fc.pth"
        },
        "vfnet_x101-64x4d-mdconv-c3-c5_fpn_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/vfnet/vfnet_x101-64x4d-mdconv-c3-c5_fpn_ms-2x_coco.py",
            "checkpoint": "/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth"
        }
    },
    "convnext": {
        "mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco": {
            "config_file": pasta_checkpoints + "configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py",
            "checkpoint": "/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth"
        },
        "cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco": {
            "config_file": pasta_checkpoints + "configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth"
        },
        "cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco": {
            "config_file": pasta_checkpoints + "configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth"
        }
    },
    "scratch": {
        "faster-rcnn_r50-scratch_fpn_gn-all_6x_coco": {
            "config_file": pasta_checkpoints + "configs/scratch/faster-rcnn_r50-scratch_fpn_gn-all_6x_coco.py",
            "checkpoint": "/scratch_faster_rcnn_r50_fpn_gn_6x_bbox_mAP-0.407_20200201_193013-90813d01.pth"
        },
        "mask-rcnn_r50-scratch_fpn_gn-all_6x_coco": {
            "config_file": pasta_checkpoints + "configs/scratch/mask-rcnn_r50-scratch_fpn_gn-all_6x_coco.py",
            "checkpoint": "/scratch_mask_rcnn_r50_fpn_gn_6x_bbox_mAP-0.412__segm_mAP-0.374_20200201_193051-1e190a40.pth"
        }
    },
    "conditional_detr": {
        "conditional-detr_r50_8xb2-50e_coco": {
            "config_file": pasta_checkpoints + "configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py",
            "checkpoint": "/conditional-detr_r50_8xb2-50e_coco_20221121_180202-c83a1dc0.pth"
        }
    },
    "res2net": {
        "faster-rcnn_res2net-101_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/res2net/faster-rcnn_res2net-101_fpn_2x_coco.py",
            "checkpoint": "/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth"
        },
        "mask-rcnn_res2net-101_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/res2net/mask-rcnn_res2net-101_fpn_2x_coco.py",
            "checkpoint": "/mask_rcnn_r2_101_fpn_2x_coco-17f061e8.pth"
        },
        "cascade-rcnn_res2net-101_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/res2net/cascade-rcnn_res2net-101_fpn_20e_coco.py",
            "checkpoint": "/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth"
        },
        "cascade-mask-rcnn_res2net-101_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/res2net/cascade-mask-rcnn_res2net-101_fpn_20e_coco.py",
            "checkpoint": "/cascade_mask_rcnn_r2_101_fpn_20e_coco-8a7b41e1.pth"
        },
        "htc_res2net-101_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/res2net/htc_res2net-101_fpn_20e_coco.py",
            "checkpoint": "/htc_r2_101_fpn_20e_coco-3a8d2112.pth"
        }
    },
    "objects365": {},
    "empirical_attention": {
        "faster-rcnn_r50-attn1111_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/empirical_attention/faster-rcnn_r50-attn1111_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth"
        },
        "faster-rcnn_r50-attn0010_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/empirical_attention/faster-rcnn_r50-attn0010_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_attention_0010_1x_coco_20200130-7cb0c14d.pth"
        },
        "faster-rcnn_r50-attn1111-dcn_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/empirical_attention/faster-rcnn_r50-attn1111-dcn_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco_20200130-8b2523a6.pth"
        },
        "faster-rcnn_r50-attn0010-dcn_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/empirical_attention/faster-rcnn_r50-attn0010-dcn_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130-1a2e831d.pth"
        }
    },
    "fsaf": {
        "fsaf_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/fsaf/fsaf_r50_fpn_1x_coco.py",
            "checkpoint": "/fsaf_r50_fpn_1x_coco-94ccc51f.pth"
        },
        "fsaf_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/fsaf/fsaf_r101_fpn_1x_coco.py",
            "checkpoint": "/fsaf_r101_fpn_1x_coco-9e71098f.pth"
        },
        "fsaf_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/fsaf/fsaf_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/fsaf_x101_64x4d_fpn_1x_coco-e3f6e6fd.pth"
        }
    },
    "dyhead": {
        "atss_r50-caffe_fpn_dyhead_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dyhead/atss_r50-caffe_fpn_dyhead_1x_coco.py",
            "checkpoint": "/atss_r50_fpn_dyhead_for_reproduction_4x4_1x_coco_20220107_213939-162888e6.pth"
        },
        "atss_r50_fpn_dyhead_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py",
            "checkpoint": "/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth"
        },
        "atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco": {
            "config_file": pasta_checkpoints + "configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco.py",
            "checkpoint": "/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth"
        }
    },
    "fast_rcnn": {},
    "hrnet": {
        "faster-rcnn_hrnetv2p-w18-1x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/faster-rcnn_hrnetv2p-w18-1x_coco.py",
            "checkpoint": "/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth"
        },
        "faster-rcnn_hrnetv2p-w18-2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/faster-rcnn_hrnetv2p-w18-2x_coco.py",
            "checkpoint": "/faster_rcnn_hrnetv2p_w18_2x_coco_20200702_085731-a4ec0611.pth"
        },
        "faster-rcnn_hrnetv2p-w32-1x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/faster-rcnn_hrnetv2p-w32-1x_coco.py",
            "checkpoint": "/faster_rcnn_hrnetv2p_w32_1x_coco_20200130-6e286425.pth"
        },
        "faster-rcnn_hrnetv2p-w32_2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/faster-rcnn_hrnetv2p-w32_2x_coco.py",
            "checkpoint": "/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927-976a9c15.pth"
        },
        "faster-rcnn_hrnetv2p-w40-1x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/faster-rcnn_hrnetv2p-w40-1x_coco.py",
            "checkpoint": "/faster_rcnn_hrnetv2p_w40_1x_coco_20200210-95c1f5ce.pth"
        },
        "faster-rcnn_hrnetv2p-w40_2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/faster-rcnn_hrnetv2p-w40_2x_coco.py",
            "checkpoint": "/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033-0f236ef4.pth"
        },
        "mask-rcnn_hrnetv2p-w18-1x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/mask-rcnn_hrnetv2p-w18-1x_coco.py",
            "checkpoint": "/mask_rcnn_hrnetv2p_w18_1x_coco_20200205-1c3d78ed.pth"
        },
        "mask-rcnn_hrnetv2p-w18-2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/mask-rcnn_hrnetv2p-w18-2x_coco.py",
            "checkpoint": "/mask_rcnn_hrnetv2p_w18_2x_coco_20200212-b3c825b1.pth"
        },
        "mask-rcnn_hrnetv2p-w32-1x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/mask-rcnn_hrnetv2p-w32-1x_coco.py",
            "checkpoint": "/mask_rcnn_hrnetv2p_w32_1x_coco_20200207-b29f616e.pth"
        },
        "mask-rcnn_hrnetv2p-w32-2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/mask-rcnn_hrnetv2p-w32-2x_coco.py",
            "checkpoint": "/mask_rcnn_hrnetv2p_w32_2x_coco_20200213-45b75b4d.pth"
        },
        "mask-rcnn_hrnetv2p-w40_1x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/mask-rcnn_hrnetv2p-w40_1x_coco.py",
            "checkpoint": "/mask_rcnn_hrnetv2p_w40_1x_coco_20200511_015646-66738b35.pth"
        },
        "mask-rcnn_hrnetv2p-w40-2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/mask-rcnn_hrnetv2p-w40-2x_coco.py",
            "checkpoint": "/mask_rcnn_hrnetv2p_w40_2x_coco_20200512_163732-aed5e4ab.pth"
        },
        "cascade-rcnn_hrnetv2p-w18-20e_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/cascade-rcnn_hrnetv2p-w18-20e_coco.py",
            "checkpoint": "/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210-434be9d7.pth"
        },
        "cascade-rcnn_hrnetv2p-w32-20e_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/cascade-rcnn_hrnetv2p-w32-20e_coco.py",
            "checkpoint": "/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208-928455a4.pth"
        },
        "cascade-rcnn_hrnetv2p-w40-20e_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/cascade-rcnn_hrnetv2p-w40-20e_coco.py",
            "checkpoint": "/cascade_rcnn_hrnetv2p_w40_20e_coco_20200512_161112-75e47b04.pth"
        },
        "cascade-mask-rcnn_hrnetv2p-w18_20e_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/cascade-mask-rcnn_hrnetv2p-w18_20e_coco.py",
            "checkpoint": "/cascade_mask_rcnn_hrnetv2p_w18_20e_coco_20200210-b543cd2b.pth"
        },
        "cascade-mask-rcnn_hrnetv2p-w32_20e_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/cascade-mask-rcnn_hrnetv2p-w32_20e_coco.py",
            "checkpoint": "/cascade_mask_rcnn_hrnetv2p_w32_20e_coco_20200512_154043-39d9cf7b.pth"
        },
        "cascade-mask-rcnn_hrnetv2p-w40-20e_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/cascade-mask-rcnn_hrnetv2p-w40-20e_coco.py",
            "checkpoint": "/cascade_mask_rcnn_hrnetv2p_w40_20e_coco_20200527_204922-969c4610.pth"
        },
        "htc_hrnetv2p-w18_20e_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/htc_hrnetv2p-w18_20e_coco.py",
            "checkpoint": "/htc_hrnetv2p_w18_20e_coco_20200210-b266988c.pth"
        },
        "htc_hrnetv2p-w32_20e_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/htc_hrnetv2p-w32_20e_coco.py",
            "checkpoint": "/htc_hrnetv2p_w32_20e_coco_20200207-7639fa12.pth"
        },
        "htc_hrnetv2p-w40_20e_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/htc_hrnetv2p-w40_20e_coco.py",
            "checkpoint": "/htc_hrnetv2p_w40_20e_coco_20200529_183411-417c4d5b.pth"
        },
        "fcos_hrnetv2p-w18-gn-head_4xb4-1x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/fcos_hrnetv2p-w18-gn-head_4xb4-1x_coco.py",
            "checkpoint": "/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20201212_100710-4ad151de.pth"
        },
        "fcos_hrnetv2p-w18-gn-head_4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/fcos_hrnetv2p-w18-gn-head_4xb4-2x_coco.py",
            "checkpoint": "/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20201212_101110-5c575fa5.pth"
        },
        "fcos_hrnetv2p-w32-gn-head_4xb4-1x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/fcos_hrnetv2p-w32-gn-head_4xb4-1x_coco.py",
            "checkpoint": "/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20201211_134730-cb8055c0.pth"
        },
        "fcos_hrnetv2p-w32-gn-head_4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/fcos_hrnetv2p-w32-gn-head_4xb4-2x_coco.py",
            "checkpoint": "/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20201212_112133-77b6b9bb.pth"
        },
        "fcos_hrnetv2p-w18-gn-head_ms-640-800-4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/fcos_hrnetv2p-w18-gn-head_ms-640-800-4xb4-2x_coco.py",
            "checkpoint": "/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20201212_111651-441e9d9f.pth"
        },
        "fcos_hrnetv2p-w32-gn-head_ms-640-800-4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/fcos_hrnetv2p-w32-gn-head_ms-640-800-4xb4-2x_coco.py",
            "checkpoint": "/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20201212_090846-b6f2b49f.pth"
        },
        "fcos_hrnetv2p-w40-gn-head_ms-640-800-4xb4-2x_coco": {
            "config_file": pasta_checkpoints + "configs/hrnet/fcos_hrnetv2p-w40-gn-head_ms-640-800-4xb4-2x_coco.py",
            "checkpoint": "/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20201212_124752-f22d2ce5.pth"
        }
    },
    "resnet_strikes_back": {},
    "queryinst": {
        "queryinst_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/queryinst/queryinst_r50_fpn_1x_coco.py",
            "checkpoint": "/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth"
        },
        "queryinst_r50_fpn_ms-480-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/queryinst/queryinst_r50_fpn_ms-480-800-3x_coco.py",
            "checkpoint": "/queryinst_r50_fpn_mstrain_480-800_3x_coco_20210901_103643-7837af86.pth"
        },
        "queryinst_r50_fpn_300-proposals_crop-ms-480-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/queryinst/queryinst_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py",
            "checkpoint": "/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_101802-85cffbd8.pth"
        },
        "queryinst_r101_fpn_ms-480-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/queryinst/queryinst_r101_fpn_ms-480-800-3x_coco.py",
            "checkpoint": "/queryinst_r101_fpn_mstrain_480-800_3x_coco_20210904_104048-91f9995b.pth"
        },
        "queryinst_r101_fpn_300-proposals_crop-ms-480-800-3x_coco": {
            "config_file": pasta_checkpoints + "configs/queryinst/queryinst_r101_fpn_300-proposals_crop-ms-480-800-3x_coco.py",
            "checkpoint": "/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth"
        }
    },
    "ms_rcnn": {
        "ms-rcnn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth"
        },
        "ms-rcnn_r50-caffe_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_2x_coco.py",
            "checkpoint": "/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth"
        },
        "ms-rcnn_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/ms_rcnn/ms-rcnn_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/ms_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.404__segm_mAP-0.376_20200506_004755-b9b12a37.pth"
        },
        "ms-rcnn_r101-caffe_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/ms_rcnn/ms-rcnn_r101-caffe_fpn_2x_coco.py",
            "checkpoint": "/ms_rcnn_r101_caffe_fpn_2x_coco_bbox_mAP-0.411__segm_mAP-0.381_20200506_011134-5f3cc74f.pth"
        },
        "ms-rcnn_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/ms_rcnn/ms-rcnn_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth"
        },
        "ms-rcnn_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/ms_rcnn/ms-rcnn_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth"
        },
        "ms-rcnn_x101-64x4d_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/ms_rcnn/ms-rcnn_x101-64x4d_fpn_2x_coco.py",
            "checkpoint": "/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308-02a445e2.pth"
        }
    },
    "qdtrack": {},
    "gn": {
        "mask-rcnn_r50_fpn_gn-all_2x_coco": {
            "config_file": pasta_checkpoints + "configs/gn/mask-rcnn_r50_fpn_gn-all_2x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth"
        },
        "mask-rcnn_r50_fpn_gn-all_3x_coco": {
            "config_file": pasta_checkpoints + "configs/gn/mask-rcnn_r50_fpn_gn-all_3x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_gn-all_3x_coco_20200214-8b23b1e5.pth"
        },
        "mask-rcnn_r101_fpn_gn-all_2x_coco": {
            "config_file": pasta_checkpoints + "configs/gn/mask-rcnn_r101_fpn_gn-all_2x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_gn-all_2x_coco_20200205-d96b1b50.pth"
        },
        "mask-rcnn_r101_fpn_gn-all_3x_coco": {
            "config_file": pasta_checkpoints + "configs/gn/mask-rcnn_r101_fpn_gn-all_3x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609-0df864f4.pth"
        },
        "mask-rcnn_r50-contrib_fpn_gn-all_2x_coco": {
            "config_file": pasta_checkpoints + "configs/gn/mask-rcnn_r50-contrib_fpn_gn-all_2x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco_20200207-20d3e849.pth"
        },
        "mask-rcnn_r50-contrib_fpn_gn-all_3x_coco": {
            "config_file": pasta_checkpoints + "configs/gn/mask-rcnn_r50-contrib_fpn_gn-all_3x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco_20200225-542aefbc.pth"
        }
    },
    "legacy_1.x": {
        "mask-rcnn_r50_fpn_1x_coco_v1": {
            "config_file": pasta_checkpoints + "configs/legacy_1.x/mask-rcnn_r50_fpn_1x_coco_v1.py",
            "checkpoint": "/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth"
        },
        "retinanet_r50-caffe_fpn_1x_coco_v1": {
            "config_file": pasta_checkpoints + "configs/legacy_1.x/retinanet_r50-caffe_fpn_1x_coco_v1.py",
            "checkpoint": "/"
        },
        "retinanet_r50_fpn_1x_coco_v1": {
            "config_file": pasta_checkpoints + "configs/legacy_1.x/retinanet_r50_fpn_1x_coco_v1.py",
            "checkpoint": "/retinanet_r50_fpn_1x_20181125-7b0c2548.pth"
        },
        "cascade-mask-rcnn_r50_fpn_1x_coco_v1": {
            "config_file": pasta_checkpoints + "configs/legacy_1.x/cascade-mask-rcnn_r50_fpn_1x_coco_v1.py",
            "checkpoint": "/cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth"
        },
        "ssd300_coco_v1": {
            "config_file": pasta_checkpoints + "configs/legacy_1.x/ssd300_coco_v1.py",
            "checkpoint": "/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth"
        }
    },
    "regnet": {
        "mask_rcnn": {
            "config_file": pasta_checkpoints + "configs/regnet./mask_rcnn/mask-rcnn_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth"
        },
        "mask-rcnn_regnetx-3.2GF_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-3.2GF_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141-2a9d1814.pth"
        },
        "mask-rcnn_regnetx-4GF_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-4GF_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-4GF_fpn_1x_coco_20200517_180217-32e9c92d.pth"
        },
        "mask-rcnn_regnetx-6.4GF_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-6.4GF_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-6.4GF_fpn_1x_coco_20200517_180439-3a7aae83.pth"
        },
        "mask-rcnn_regnetx-8GF_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-8GF_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-8GF_fpn_1x_coco_20200517_180515-09daa87e.pth"
        },
        "mask-rcnn_regnetx-12GF_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-12GF_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-12GF_fpn_1x_coco_20200517_180552-b538bd8b.pth"
        },
        "mask-rcnn_regnetx-3.2GF-mdconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-3.2GF-mdconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco_20200520_172726-75f40794.pth"
        },
        "faster_rcnn": {
            "config_file": pasta_checkpoints + "configs/regnet./faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        },
        "faster-rcnn_regnetx-3.2GF_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/faster-rcnn_regnetx-3.2GF_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_regnetx-3.2GF_fpn_1x_coco_20200517_175927-126fd9bf.pth"
        },
        "faster-rcnn_regnetx-3.2GF_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/faster-rcnn_regnetx-3.2GF_fpn_2x_coco.py",
            "checkpoint": "/faster_rcnn_regnetx-3.2GF_fpn_2x_coco_20200520_223955-e2081918.pth"
        },
        "retinanet": {
            "config_file": pasta_checkpoints + "configs/regnet./retinanet/retinanet_r50_fpn_1x_coco.py",
            "checkpoint": "/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"
        },
        "retinanet_regnetx-800MF_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/retinanet_regnetx-800MF_fpn_1x_coco.py",
            "checkpoint": "/retinanet_regnetx-800MF_fpn_1x_coco_20200517_191403-f6f91d10.pth"
        },
        "retinanet_regnetx-1.6GF_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/retinanet_regnetx-1.6GF_fpn_1x_coco.py",
            "checkpoint": "/retinanet_regnetx-1.6GF_fpn_1x_coco_20200517_191403-37009a9d.pth"
        },
        "retinanet_regnetx-3.2GF_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py",
            "checkpoint": "/retinanet_regnetx-3.2GF_fpn_1x_coco_20200520_163141-cb1509e8.pth"
        },
        "faster-rcnn_regnetx-400MF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/faster-rcnn_regnetx-400MF_fpn_ms-3x_coco.py",
            "checkpoint": "/faster_rcnn_regnetx-400MF_fpn_mstrain_3x_coco_20210526_095112-e1967c37.pth"
        },
        "faster-rcnn_regnetx-800MF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/faster-rcnn_regnetx-800MF_fpn_ms-3x_coco.py",
            "checkpoint": "/faster_rcnn_regnetx-800MF_fpn_mstrain_3x_coco_20210526_095118-a2c70b20.pth"
        },
        "faster-rcnn_regnetx-1.6GF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/faster-rcnn_regnetx-1.6GF_fpn_ms-3x_coco.py",
            "checkpoint": "/faster_rcnn_regnetx-1_20210526_095325-94aa46cc.pth"
        },
        "faster-rcnn_regnetx-3.2GF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/faster-rcnn_regnetx-3.2GF_fpn_ms-3x_coco.py",
            "checkpoint": "/faster_rcnn_regnetx-3_20210526_095152-e16a5227.pth"
        },
        "faster-rcnn_regnetx-4GF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/faster-rcnn_regnetx-4GF_fpn_ms-3x_coco.py",
            "checkpoint": "/faster_rcnn_regnetx-4GF_fpn_mstrain_3x_coco_20210526_095201-65eaf841.pth"
        },
        "mask-rcnn_regnetx-400MF_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-400MF_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-400MF_fpn_mstrain-poly_3x_coco_20210601_235443-8aac57a4.pth"
        },
        "mask-rcnn_regnetx-800MF_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-800MF_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-800MF_fpn_mstrain-poly_3x_coco_20210602_210641-715d51f5.pth"
        },
        "mask-rcnn_regnetx-1.6GF_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-1.6GF_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-1_20210602_210641-6764cff5.pth"
        },
        "mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200521_202221-99879813.pth"
        },
        "mask-rcnn_regnetx-4GF_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/mask-rcnn_regnetx-4GF_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_regnetx-4GF_fpn_mstrain-poly_3x_coco_20210602_032621-00f0331c.pth"
        },
        "cascade-mask-rcnn_regnetx-400MF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/cascade-mask-rcnn_regnetx-400MF_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco_20210715_211619-5142f449.pth"
        },
        "cascade-mask-rcnn_regnetx-800MF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/cascade-mask-rcnn_regnetx-800MF_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco_20210715_211616-dcbd13f4.pth"
        },
        "cascade-mask-rcnn_regnetx-1.6GF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/cascade-mask-rcnn_regnetx-1.6GF_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_regnetx-1_20210715_211616-75f29a61.pth"
        },
        "cascade-mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/cascade-mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_regnetx-3_20210715_211616-b9c2c58b.pth"
        },
        "cascade-mask-rcnn_regnetx-4GF_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/regnet/cascade-mask-rcnn_regnetx-4GF_fpn_ms-3x_coco.py",
            "checkpoint": "/cascade_mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco_20210715_212034-cbb1be4c.pth"
        }
    },
    "openimages": {
        "faster-rcnn_r50_fpn_32xb2-1x_openimages": {
            "config_file": pasta_checkpoints + "configs/openimages/faster-rcnn_r50_fpn_32xb2-1x_openimages.py",
            "checkpoint": "/faster_rcnn_r50_fpn_32x2_1x_openimages_20211130_231159-e87ab7ce.pth"
        },
        "faster-rcnn_r50_fpn_32xb2-cas-1x_openimages": {
            "config_file": pasta_checkpoints + "configs/openimages/faster-rcnn_r50_fpn_32xb2-cas-1x_openimages.py",
            "checkpoint": "/faster_rcnn_r50_fpn_32x2_cas_1x_openimages_20220306_202424-98c630e5.pth"
        },
        "faster-rcnn_r50_fpn_32xb2-1x_openimages-challenge": {
            "config_file": pasta_checkpoints + "configs/openimages/faster-rcnn_r50_fpn_32xb2-1x_openimages-challenge.py",
            "checkpoint": "/faster_rcnn_r50_fpn_32x2_1x_openimages_challenge_20220114_045100-0e79e5df.pth"
        },
        "faster-rcnn_r50_fpn_32xb2-cas-1x_openimages-challenge": {
            "config_file": pasta_checkpoints + "configs/openimages/faster-rcnn_r50_fpn_32xb2-cas-1x_openimages-challenge.py",
            "checkpoint": "/faster_rcnn_r50_fpn_32x2_cas_1x_openimages_challenge_20220221_192021-34c402d9.pth"
        },
        "retinanet_r50_fpn_32xb2-1x_openimages": {
            "config_file": pasta_checkpoints + "configs/openimages/retinanet_r50_fpn_32xb2-1x_openimages.py",
            "checkpoint": "/retinanet_r50_fpn_32x2_1x_openimages_20211223_071954-d2ae5462.pth"
        },
        "ssd300_32xb8-36e_openimages": {
            "config_file": pasta_checkpoints + "configs/openimages/ssd300_32xb8-36e_openimages.py",
            "checkpoint": "/ssd300_32x8_36e_openimages_20211224_000232-dce93846.pth"
        }
    },
    "detectors": {
        "cascade-rcnn_r50-rfp_1x_coco": {
            "config_file": pasta_checkpoints + "configs/detectors/cascade-rcnn_r50-rfp_1x_coco.py",
            "checkpoint": "/cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth"
        },
        "cascade-rcnn_r50-sac_1x_coco": {
            "config_file": pasta_checkpoints + "configs/detectors/cascade-rcnn_r50-sac_1x_coco.py",
            "checkpoint": "/cascade_rcnn_r50_sac_1x_coco-24bfda62.pth"
        },
        "detectors_cascade-rcnn_r50_1x_coco": {
            "config_file": pasta_checkpoints + "configs/detectors/detectors_cascade-rcnn_r50_1x_coco.py",
            "checkpoint": "/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth"
        },
        "htc_r50-rfp_1x_coco": {
            "config_file": pasta_checkpoints + "configs/detectors/htc_r50-rfp_1x_coco.py",
            "checkpoint": "/htc_r50_rfp_1x_coco-8ff87c51.pth"
        },
        "htc_r50-sac_1x_coco": {
            "config_file": pasta_checkpoints + "configs/detectors/htc_r50-sac_1x_coco.py",
            "checkpoint": "/htc_r50_sac_1x_coco-bfa60c54.pth"
        },
        "detectors_htc-r50_1x_coco": {
            "config_file": pasta_checkpoints + "configs/detectors/detectors_htc-r50_1x_coco.py",
            "checkpoint": "/detectors_htc_r50_1x_coco-329b1453.pth"
        },
        "detectors_htc-r101_20e_coco": {
            "config_file": pasta_checkpoints + "configs/detectors/detectors_htc-r101_20e_coco.py",
            "checkpoint": "/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth"
        }
    },
    "cityscapes": {
        "faster-rcnn_r50_fpn_1x_cityscapes": {
            "config_file": pasta_checkpoints + "configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes.py",
            "checkpoint": "/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
        },
        "mask-rcnn_r50_fpn_1x_cityscapes": {
            "config_file": pasta_checkpoints + "configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes.py",
            "checkpoint": "/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth"
        }
    },
    "strongsort": {},
    "yolox": {
        "yolox_tiny_8xb8-300e_coco": {
            "config_file": pasta_checkpoints + "configs/yolox/yolox_tiny_8xb8-300e_coco.py",
            "checkpoint": "/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
        },
        "yolox_s_8xb8-300e_coco": {
            "config_file": pasta_checkpoints + "configs/yolox/yolox_s_8xb8-300e_coco.py",
            "checkpoint": "/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
        },
        "yolox_l_8xb8-300e_coco": {
            "config_file": pasta_checkpoints + "configs/yolox/yolox_l_8xb8-300e_coco.py",
            "checkpoint": "/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
        },
        "yolox_x_8xb8-300e_coco": {
            "config_file": pasta_checkpoints + "configs/yolox/yolox_x_8xb8-300e_coco.py",
            "checkpoint": "/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
        }
    },
    "common": {},
    "ddq": {
        "ddq-detr-4scale_r50_8xb2-12e_coco": {
            "config_file": pasta_checkpoints + "configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py",
            "checkpoint": "/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth"
        },
        "ddq-detr-5scale_r50_8xb2-12e_coco": {
            "config_file": pasta_checkpoints + "configs/ddq/ddq-detr-5scale_r50_8xb2-12e_coco.py",
            "checkpoint": "/ddq_detr_5scale_coco_1x.pth"
        },
        "ddq-detr-4scale_swinl_8xb2-30e_coco": {
            "config_file": pasta_checkpoints + "configs/ddq/ddq-detr-4scale_swinl_8xb2-30e_coco.py",
            "checkpoint": "/ddq_detr_swinl_30e.pth"
        }
    },
    "grounding_dino": {},
    "deepfashion": {
        "mask-rcnn_r50_fpn_15e_deepfashion": {
            "config_file": pasta_checkpoints + "configs/deepfashion/mask-rcnn_r50_fpn_15e_deepfashion.py",
            "checkpoint": "/mask_rcnn_r50_fpn_15e_deepfashion_20200329_192752.pth"
        }
    },
    "dcnv2": {
        "faster-rcnn_r50-mdconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcnv2/faster-rcnn_r50-mdconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200130-d099253b.pth"
        },
        "faster-rcnn_r50-mdconv-group4-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcnv2/faster-rcnn_r50-mdconv-group4-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130-01262257.pth"
        },
        "faster-rcnn_r50_fpn_mdpool_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcnv2/faster-rcnn_r50_fpn_mdpool_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307-c0df27ff.pth"
        },
        "mask-rcnn_r50-mdconv-c3-c5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcnv2/mask-rcnn_r50-mdconv-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203-ad97591f.pth"
        },
        "mask-rcnn_r50-mdconv-c3-c5_fpn_amp-1x_coco": {
            "config_file": pasta_checkpoints + "configs/dcnv2/mask-rcnn_r50-mdconv-c3-c5_fpn_amp-1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco_20210520_180434-cf8fefa5.pth"
        }
    },
    "grid_rcnn": {
        "grid-rcnn_r50_fpn_gn-head_2x_coco": {
            "config_file": pasta_checkpoints + "configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py",
            "checkpoint": "/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth"
        },
        "grid-rcnn_r101_fpn_gn-head_2x_coco": {
            "config_file": pasta_checkpoints + "configs/grid_rcnn/grid-rcnn_r101_fpn_gn-head_2x_coco.py",
            "checkpoint": "/grid_rcnn_r101_fpn_gn-head_2x_coco_20200309-d6eca030.pth"
        },
        "grid-rcnn_x101-32x4d_fpn_gn-head_2x_coco": {
            "config_file": pasta_checkpoints + "configs/grid_rcnn/grid-rcnn_x101-32x4d_fpn_gn-head_2x_coco.py",
            "checkpoint": "/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco_20200130-d8f0e3ff.pth"
        },
        "grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco": {
            "config_file": pasta_checkpoints + "configs/grid_rcnn/grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco.py",
            "checkpoint": "/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204-ec76a754.pth"
        }
    },
    "sort": {},
    "panoptic_fpn": {
        "panoptic-fpn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py",
            "checkpoint": "/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth"
        },
        "panoptic-fpn_r50_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/panoptic_fpn/panoptic-fpn_r50_fpn_ms-3x_coco.py",
            "checkpoint": "/panoptic_fpn_r50_fpn_mstrain_3x_coco_20210824_171155-5650f98b.pth"
        },
        "panoptic-fpn_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/panoptic_fpn/panoptic-fpn_r101_fpn_1x_coco.py",
            "checkpoint": "/panoptic_fpn_r101_fpn_1x_coco_20210820_193950-ab9157a2.pth"
        },
        "panoptic-fpn_r101_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/panoptic_fpn/panoptic-fpn_r101_fpn_ms-3x_coco.py",
            "checkpoint": "/panoptic_fpn_r101_fpn_mstrain_3x_coco_20210823_114712-9c99acc4.pth"
        }
    },
    "mask_rcnn": {
        "mask-rcnn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth"
        },
        "mask-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
        },
        "mask-rcnn_r50_fpn_amp-1x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r50_fpn_amp-1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_fp16_1x_coco_20200205-59faf7e4.pth"
        },
        "mask-rcnn_r50_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth"
        },
        "mask-rcnn_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth"
        },
        "mask-rcnn_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth"
        },
        "mask-rcnn_r101_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r101_fpn_2x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth"
        },
        "mask-rcnn_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth"
        },
        "mask-rcnn_x101-32x4d_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_x101-32x4d_fpn_2x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth"
        },
        "mask-rcnn_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth"
        },
        "mask-rcnn_x101-64x4d_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_x101-64x4d_fpn_2x_coco.py",
            "checkpoint": "/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208-39d6f70c.pth"
        },
        "mask-rcnn_x101-32x8d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_x101-32x8d_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x8d_fpn_1x_coco_20220630_173841-0aaf329e.pth"
        },
        "mask-rcnn_r50-caffe_fpn_ms-poly-2x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-2x_coco.py",
            "checkpoint": "/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco_bbox_mAP-0.403__segm_mAP-0.365_20200504_231822-a75c98ce.pth"
        },
        "mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
        },
        "mask-rcnn_r50_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r50_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
        },
        "mask-rcnn_r101-caffe_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r101-caffe_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco_20210526_132339-3c33ce02.pth"
        },
        "mask-rcnn_r101_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_r101_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth"
        },
        "mask-rcnn_x101-32x4d_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_x101-32x4d_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x4d_fpn_mstrain-poly_3x_coco_20210524_201410-abcd7859.pth"
        },
        "mask-rcnn_x101-32x8d_fpn_ms-poly-1x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_x101-32x8d_fpn_ms-poly-1x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x8d_fpn_mstrain-poly_1x_coco_20220630_170346-b4637974.pth"
        },
        "mask-rcnn_x101-32x8d_fpn_ms-poly-3x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_x101-32x8d_fpn_ms-poly-3x_coco.py",
            "checkpoint": "/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco_20210607_161042-8bd2c639.pth"
        },
        "mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco": {
            "config_file": pasta_checkpoints + "configs/mask_rcnn/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py",
            "checkpoint": "/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth"
        }
    },
    "swin": {
        "mask-rcnn_swin-t-p4-w7_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth"
        },
        "mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco": {
            "config_file": pasta_checkpoints + "configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py",
            "checkpoint": "/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth"
        },
        "mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco": {
            "config_file": pasta_checkpoints + "configs/swin/mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py",
            "checkpoint": "/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth"
        },
        "mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco": {
            "config_file": pasta_checkpoints + "configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py",
            "checkpoint": "/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
        }
    },
    "rtmdet": {
        "rtmdet_tiny_8xb32-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
            "checkpoint": "/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
        },
        "rtmdet_s_8xb32-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
            "checkpoint": "/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth"
        },
        "rtmdet_m_8xb32-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet_m_8xb32-300e_coco.py",
            "checkpoint": "/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
        },
        "rtmdet_l_8xb32-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet_l_8xb32-300e_coco.py",
            "checkpoint": "/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
        },
        "rtmdet_x_8xb32-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet_x_8xb32-300e_coco.py",
            "checkpoint": "/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
        },
        "rtmdet_x_p6_4xb8-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet_x_p6_4xb8-300e_coco.py",
            "checkpoint": "/rtmdet_x_p6_4xb8-300e_coco-bf32be58.pth"
        },
        "rtmdet_l_convnext_b_4xb32-100e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet_l_convnext_b_4xb32-100e_coco.py",
            "checkpoint": "/rtmdet_l_convnext_b_4xb32-100e_coco-d4731b3d.pth"
        },
        "rtmdet_l_swin_b_4xb32-100e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet_l_swin_b_4xb32-100e_coco.py",
            "checkpoint": "/rtmdet_l_swin_b_4xb32-100e_coco-0828ce5d.pth"
        },
        "rtmdet_l_swin_b_p6_4xb16-100e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py",
            "checkpoint": "/rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth"
        },
        "rtmdet-ins_tiny_8xb32-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py",
            "checkpoint": "/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
        },
        "rtmdet-ins_s_8xb32-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py",
            "checkpoint": "/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth"
        },
        "rtmdet-ins_m_8xb32-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet-ins_m_8xb32-300e_coco.py",
            "checkpoint": "/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth"
        },
        "rtmdet-ins_l_8xb32-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py",
            "checkpoint": "/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth"
        },
        "rtmdet-ins_x_8xb16-300e_coco": {
            "config_file": pasta_checkpoints + "configs/rtmdet/rtmdet-ins_x_8xb16-300e_coco.py",
            "checkpoint": "/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth"
        }
    },
    "ocsort": {},
    "sabl": {
        "sabl-faster-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-faster-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth"
        },
        "sabl-faster-rcnn_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-faster-rcnn_r101_fpn_1x_coco.py",
            "checkpoint": "/sabl_faster_rcnn_r101_fpn_1x_coco-f804c6c1.pth"
        },
        "sabl-cascade-rcnn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-cascade-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/sabl_cascade_rcnn_r50_fpn_1x_coco-e1748e5e.pth"
        },
        "sabl-cascade-rcnn_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-cascade-rcnn_r101_fpn_1x_coco.py",
            "checkpoint": "/sabl_cascade_rcnn_r101_fpn_1x_coco-2b83e87c.pth"
        },
        "sabl-retinanet_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-retinanet_r50_fpn_1x_coco.py",
            "checkpoint": "/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth"
        },
        "sabl-retinanet_r50-gn_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-retinanet_r50-gn_fpn_1x_coco.py",
            "checkpoint": "/sabl_retinanet_r50_fpn_gn_1x_coco-e16dfcf1.pth"
        },
        "sabl-retinanet_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-retinanet_r101_fpn_1x_coco.py",
            "checkpoint": "/sabl_retinanet_r101_fpn_1x_coco-42026904.pth"
        },
        "sabl-retinanet_r101-gn_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-retinanet_r101-gn_fpn_1x_coco.py",
            "checkpoint": "/sabl_retinanet_r101_fpn_gn_1x_coco-40a893e8.pth"
        },
        "sabl-retinanet_r101-gn_fpn_ms-640-800-2x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-retinanet_r101-gn_fpn_ms-640-800-2x_coco.py",
            "checkpoint": "/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco-1e63382c.pth"
        },
        "sabl-retinanet_r101-gn_fpn_ms-480-960-2x_coco": {
            "config_file": pasta_checkpoints + "configs/sabl/sabl-retinanet_r101-gn_fpn_ms-480-960-2x_coco.py",
            "checkpoint": "/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco-5342f857.pth"
        }
    },
    "groie": {
        "faster_rcnn": {
            "config_file": pasta_checkpoints + "configs/groie./faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        },
        "faste-rcnn_r50_fpn_groie_1x_coco": {
            "config_file": pasta_checkpoints + "configs/groie/faste-rcnn_r50_fpn_groie_1x_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715-66ee9516.pth"
        },
        "grid-rcnn_r50_fpn_gn-head-groie_1x_coco": {
            "config_file": pasta_checkpoints + "configs/groie/grid-rcnn_r50_fpn_gn-head-groie_1x_coco.py",
            "checkpoint": "/"
        },
        "mask_rcnn": {
            "config_file": pasta_checkpoints + "configs/groie./mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
        },
        "mask-rcnn_r50_fpn_groie_1x_coco": {
            "config_file": pasta_checkpoints + "configs/groie/mask-rcnn_r50_fpn_groie_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_groie_1x_coco_20200604_211715-50d90c74.pth"
        },
        "gcnet": {
            "config_file": pasta_checkpoints + "configs/groie./gcnet/mask-rcnn_r101-syncbn-gcb-r4-c3-c5_fpn_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206-8407a3f0.pth"
        },
        "mask-rcnn_r50_fpn_syncbn-r4-gcb-c3-c5-groie_1x_coco": {
            "config_file": pasta_checkpoints + "configs/groie/mask-rcnn_r50_fpn_syncbn-r4-gcb-c3-c5-groie_1x_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth"
        },
        "mask-rcnn_r101_fpn_syncbn-r4-gcb_c3-c5-groie_1x_coco": {
            "config_file": pasta_checkpoints + "configs/groie/mask-rcnn_r101_fpn_syncbn-r4-gcb_c3-c5-groie_1x_coco.py",
            "checkpoint": "/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507-8daae01c.pth"
        }
    },
    "masktrack_rcnn": {},
    "nas_fpn": {
        "retinanet_r50_fpn_crop640-50e_coco": {
            "config_file": pasta_checkpoints + "configs/nas_fpn/retinanet_r50_fpn_crop640-50e_coco.py",
            "checkpoint": "/retinanet_r50_fpn_crop640_50e_coco-9b953d76.pth"
        },
        "retinanet_r50_nasfpn_crop640-50e_coco": {
            "config_file": pasta_checkpoints + "configs/nas_fpn/retinanet_r50_nasfpn_crop640-50e_coco.py",
            "checkpoint": "/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth"
        }
    },
    "point_rend": {
        "point-rend_r50-caffe_fpn_ms-1x_coco": {
            "config_file": pasta_checkpoints + "configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py",
            "checkpoint": "/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth"
        },
        "point-rend_r50-caffe_fpn_ms-3x_coco": {
            "config_file": pasta_checkpoints + "configs/point_rend/point-rend_r50-caffe_fpn_ms-3x_coco.py",
            "checkpoint": "/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth"
        }
    },
    "boxinst": {
        "boxinst_r50_fpn_ms-90k_coco": {
            "config_file": pasta_checkpoints + "configs/boxinst/boxinst_r50_fpn_ms-90k_coco.py",
            "checkpoint": "/boxinst_r50_fpn_ms-90k_coco_20221228_163052-6add751a.pth"
        },
        "boxinst_r101_fpn_ms-90k_coco": {
            "config_file": pasta_checkpoints + "configs/boxinst/boxinst_r101_fpn_ms-90k_coco.py",
            "checkpoint": "/boxinst_r101_fpn_ms-90k_coco_20221229_145106-facf375b.pth"
        }
    },
    "cascade_rpn": {
        "cascade-rpn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rpn/cascade-rpn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/cascade_rpn_r50_caffe_fpn_1x_coco-7aa93cef.pth"
        },
        "cascade-rpn_fast-rcnn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rpn/cascade-rpn_fast-rcnn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/crpn_fast_rcnn_r50_caffe_fpn_1x_coco-cb486e66.pth"
        },
        "cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth"
        }
    },
    "strong_baselines": {
        "mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-50e_coco": {
            "config_file": pasta_checkpoints + "configs/strong_baselines/mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-50e_coco.py",
            "checkpoint": "/<a href=\""
        },
        "mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco": {
            "config_file": pasta_checkpoints + "configs/strong_baselines/mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco.py",
            "checkpoint": "/<a href=\""
        },
        "mask-rcnn_r50-caffe_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco": {
            "config_file": pasta_checkpoints + "configs/strong_baselines/mask-rcnn_r50-caffe_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco.py",
            "checkpoint": "/<a href=\""
        },
        "mask-rcnn_r50-caffe_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-400e_coco": {
            "config_file": pasta_checkpoints + "configs/strong_baselines/mask-rcnn_r50-caffe_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-400e_coco.py",
            "checkpoint": "/<a href=\""
        }
    },
    "fpg": {
        "faster-rcnn_r50_fpg_crop640-50e_coco": {
            "config_file": pasta_checkpoints + "configs/fpg/faster-rcnn_r50_fpg_crop640-50e_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpg_crop640_50e_coco_20220311_011856-74109f42.pth"
        },
        "faster-rcnn_r50_fpg-chn128_crop640-50e_coco": {
            "config_file": pasta_checkpoints + "configs/fpg/faster-rcnn_r50_fpg-chn128_crop640-50e_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpg-chn128_crop640_50e_coco_20220311_011857-9376aa9d.pth"
        },
        "faster-rcnn_r50_fpn_crop640-50e_coco": {
            "config_file": pasta_checkpoints + "configs/fpg/faster-rcnn_r50_fpn_crop640-50e_coco.py",
            "checkpoint": "/faster_rcnn_r50_fpn_crop640_50e_coco_20220311_011857-be7c9f42.pth"
        },
        "mask-rcnn_r50_fpg_crop640-50e_coco": {
            "config_file": pasta_checkpoints + "configs/fpg/mask-rcnn_r50_fpg_crop640-50e_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpg_crop640_50e_coco_20220311_011857-233b8334.pth"
        },
        "mask-rcnn_r50_fpg-chn128_crop640-50e_coco": {
            "config_file": pasta_checkpoints + "configs/fpg/mask-rcnn_r50_fpg-chn128_crop640-50e_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpg-chn128_crop640_50e_coco_20220311_011859-043c9b4e.pth"
        },
        "mask-rcnn_r50_fpn_crop640-50e_coco": {
            "config_file": pasta_checkpoints + "configs/fpg/mask-rcnn_r50_fpn_crop640-50e_coco.py",
            "checkpoint": "/mask_rcnn_r50_fpn_crop640_50e_coco_20220311_011855-a756664a.pth"
        },
        "retinanet_r50_fpg_crop640_50e_coco": {
            "config_file": pasta_checkpoints + "configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py",
            "checkpoint": "/retinanet_r50_fpg_crop640_50e_coco_20220311_110809-b0bcf5f4.pth"
        },
        "retinanet_r50_fpg-chn128_crop640_50e_coco": {
            "config_file": pasta_checkpoints + "configs/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco.py",
            "checkpoint": "/retinanet_r50_fpg-chn128_crop640_50e_coco_20220313_104829-ee99a686.pth"
        }
    },
    "glip": {},
    "cornernet": {
        "cornernet_hourglass104_10xb5-crop511-210e-mstest_coco": {
            "config_file": pasta_checkpoints + "configs/cornernet/cornernet_hourglass104_10xb5-crop511-210e-mstest_coco.py",
            "checkpoint": "/cornernet_hourglass104_mstest_10x5_210e_coco_20200824_185720-5fefbf1c.pth"
        },
        "cornernet_hourglass104_8xb6-210e-mstest_coco": {
            "config_file": pasta_checkpoints + "configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py",
            "checkpoint": "/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth"
        },
        "cornernet_hourglass104_32xb3-210e-mstest_coco": {
            "config_file": pasta_checkpoints + "configs/cornernet/cornernet_hourglass104_32xb3-210e-mstest_coco.py",
            "checkpoint": "/cornernet_hourglass104_mstest_32x3_210e_coco_20200819_203110-1efaea91.pth"
        }
    },
    "rpn": {
        "rpn_r50-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_r50-caffe_fpn_1x_coco.py",
            "checkpoint": "/rpn_r50_caffe_fpn_1x_coco_20200531-5b903a37.pth"
        },
        "rpn_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_r50_fpn_1x_coco.py",
            "checkpoint": "/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth"
        },
        "rpn_r50_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_r50_fpn_2x_coco.py",
            "checkpoint": "/rpn_r50_fpn_2x_coco_20200131-0728c9b3.pth"
        },
        "rpn_r101-caffe_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_r101-caffe_fpn_1x_coco.py",
            "checkpoint": "/rpn_r101_caffe_fpn_1x_coco_20200531-0629a2e2.pth"
        },
        "rpn_r101_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_r101_fpn_1x_coco.py",
            "checkpoint": "/rpn_r101_fpn_1x_coco_20200131-2ace2249.pth"
        },
        "rpn_r101_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_r101_fpn_2x_coco.py",
            "checkpoint": "/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth"
        },
        "rpn_x101-32x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_x101-32x4d_fpn_1x_coco.py",
            "checkpoint": "/rpn_x101_32x4d_fpn_1x_coco_20200219-b02646c6.pth"
        },
        "rpn_x101-32x4d_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_x101-32x4d_fpn_2x_coco.py",
            "checkpoint": "/rpn_x101_32x4d_fpn_2x_coco_20200208-d22bd0bb.pth"
        },
        "rpn_x101-64x4d_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_x101-64x4d_fpn_1x_coco.py",
            "checkpoint": "/rpn_x101_64x4d_fpn_1x_coco_20200208-cde6f7dd.pth"
        },
        "rpn_x101-64x4d_fpn_2x_coco": {
            "config_file": pasta_checkpoints + "configs/rpn/rpn_x101-64x4d_fpn_2x_coco.py",
            "checkpoint": "/rpn_x101_64x4d_fpn_2x_coco_20200208-c65f524f.pth"
        }
    },
    "pvt": {
        "retinanet_pvt-t_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pvt/retinanet_pvt-t_fpn_1x_coco.py",
            "checkpoint": "/retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth"
        },
        "retinanet_pvt-s_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pvt/retinanet_pvt-s_fpn_1x_coco.py",
            "checkpoint": "/retinanet_pvt-s_fpn_1x_coco_20210906_142921-b6c94a5b.pth"
        },
        "retinanet_pvt-m_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pvt/retinanet_pvt-m_fpn_1x_coco.py",
            "checkpoint": "/retinanet_pvt-m_fpn_1x_coco_20210831_103243-55effa1b.pth"
        },
        "retinanet_pvtv2-b0_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pvt/retinanet_pvtv2-b0_fpn_1x_coco.py",
            "checkpoint": "/retinanet_pvtv2-b0_fpn_1x_coco_20210831_103157-13e9aabe.pth"
        },
        "retinanet_pvtv2-b1_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pvt/retinanet_pvtv2-b1_fpn_1x_coco.py",
            "checkpoint": "/retinanet_pvtv2-b1_fpn_1x_coco_20210831_103318-7e169a7d.pth"
        },
        "retinanet_pvtv2-b2_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pvt/retinanet_pvtv2-b2_fpn_1x_coco.py",
            "checkpoint": "/retinanet_pvtv2-b2_fpn_1x_coco_20210901_174843-529f0b9a.pth"
        },
        "retinanet_pvtv2-b3_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pvt/retinanet_pvtv2-b3_fpn_1x_coco.py",
            "checkpoint": "/retinanet_pvtv2-b3_fpn_1x_coco_20210903_151512-8357deff.pth"
        },
        "retinanet_pvtv2-b4_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pvt/retinanet_pvtv2-b4_fpn_1x_coco.py",
            "checkpoint": "/retinanet_pvtv2-b4_fpn_1x_coco_20210901_170151-83795c86.pth"
        },
        "retinanet_pvtv2-b5_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/pvt/retinanet_pvtv2-b5_fpn_1x_coco.py",
            "checkpoint": "/retinanet_pvtv2-b5_fpn_1x_coco_20210902_201800-3420eb57.pth"
        }
    },
    "wider_face": {},
    "scnet": {
        "scnet_r50_fpn_1x_coco": {
            "config_file": pasta_checkpoints + "configs/scnet/scnet_r50_fpn_1x_coco.py",
            "checkpoint": "/scnet_r50_fpn_1x_coco-c3f09857.pth"
        },
        "scnet_r50_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/scnet/scnet_r50_fpn_20e_coco.py",
            "checkpoint": "/scnet_r50_fpn_20e_coco-a569f645.pth"
        },
        "scnet_r101_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/scnet/scnet_r101_fpn_20e_coco.py",
            "checkpoint": "/scnet_r101_fpn_20e_coco-294e312c.pth"
        },
        "scnet_x101-64x4d_fpn_20e_coco": {
            "config_file": pasta_checkpoints + "configs/scnet/scnet_x101-64x4d_fpn_20e_coco.py",
            "checkpoint": "/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth"
        }
    },
    "reid": {},
    "lad": {
        "paa": {
            "config_file": pasta_checkpoints + "configs/lad./paa/paa_r101_fpn_1x_coco.py",
            "checkpoint": "/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth"
        },
        "lad_r50-paa-r101_fpn_2xb8_coco_1x": {
            "config_file": pasta_checkpoints + "configs/lad/lad_r50-paa-r101_fpn_2xb8_coco_1x.py",
            "checkpoint": "/lad_r50_paa_r101_fpn_coco_1x_20220708_124246-74c76ff0.pth"
        },
        "lad_r101-paa-r50_fpn_2xb8_coco_1x": {
            "config_file": pasta_checkpoints + "configs/lad/lad_r101-paa-r50_fpn_2xb8_coco_1x.py",
            "checkpoint": "/lad_r101_paa_r50_fpn_coco_1x_20220708_124357-9407ac54.pth"
        }
    }
}