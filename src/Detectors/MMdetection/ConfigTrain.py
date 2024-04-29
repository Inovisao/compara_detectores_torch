import json
import random
# Função que ira fazer a configuração do treino
def configTrain(model, checkpoint, path):
    JsonData = '../dataset/all/train/_annotations.coco.json'
    with open(JsonData) as f:
        data = json.load(f)
    ann_ids = []
    for anotation in data["annotations"]:
        if anotation["category_id"] not in ann_ids:
            ann_ids.append(anotation["category_id"])
    Classe = ()
    for category in data["categories"]:
        if category["id"] in ann_ids:
            Classe += (category["name"],)
    paletaCor = []
    for i in range(0,len(Classe)):
        Cor1, Cor2, Cor3 = random.randint(0,255) , random.randint(0,255), random.randint(0,255)
        cores = (Cor1,Cor2,Cor3)
        paletaCor.append(cores)

    config_Train = f"""
# Inherit and overwrite part of the config based on this config
_base_ = './{model}.py'

data_root = '../dataset/all/' # dataset root

train_batch_size_per_gpu = 4 # Tamanho do lote da imagem

train_num_workers = 1

max_epochs = 10 # Quantidade de Epocas
stage2_num_epochs = 5
base_lr = 0.0001 # Taxa de aprendizagem 

metainfo = {{
    'classes': {Classe},
    'palette':{paletaCor}
}}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='filesJSON/fold_1_train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='filesJSON/fold_1_val.json'))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='filesJSON/fold_1_test.json'))

val_evaluator = dict(ann_file=data_root + 'filesJSON/fold_1_val.json')

test_evaluator = dict(ann_file=data_root + 'filesJSON/fold_1_test.json')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10),
    dict(
        # use cosine lr from 10 to 20 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# optimizer
# Onde Ira alterar o optmizador (SGD Adam AdamW RMSprop SGDP AdamP LARS LAMB)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05), # Troce o type='' Pelo optmizador de sua escolha
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# load COCO pre-trained weight
load_from = f'{checkpoint}'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
"""

    with open(f'{path}/train.py', 'w') as f:
        f.write(config_Train)
