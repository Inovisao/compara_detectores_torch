import argparse
import os
import sys
import traceback
import yaml
import torch
import numpy as np
import torchinfo

from config import (
    BATCH_SIZE as CFG_BATCH_SIZE, RESIZE_TO as CFG_RESIZE_TO, NUM_EPOCHS as CFG_NUM_EPOCHS,
    NUM_WORKERS as CFG_NUM_WORKERS, LR as CFG_LR, DEVICE as CFG_DEVICE, ROOT_DATA_DIR,
    TRAIN_DIR, VALID_DIR, CLASSES as CFG_CLASSES, NUM_CLASSES as CFG_NUM_CLASSES,
    VISUALIZE_TRANSFORMED_IMAGES as CFG_VIS, OUT_DIR as CFG_OUT_DIR, DATA_PATH as CFG_DATA_PATH
)
from utils.detection.datasets import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)
from utils.detection.detr.engine import train, evaluate
from detection.detr.model import DETRModel
from utils.detection.detr.matcher import HungarianMatcher
from utils.detection.detr.detr import SetCriterion, PostProcess
from torch.utils.data import distributed, RandomSampler, SequentialSampler
from utils.detection.detr.general import (
    SaveBestModel, init_seeds, set_training_dir, save_model_state, save_mAP, show_tranformed_image
)
from utils.detection.detr.logging import set_log, coco_log

RANK = int(os.getenv('RANK', -1))
np.random.seed(42)

def error(msg: str, exc: Exception | None = None, exit_code: int = 1):
    """Loga erro padronizado e encerra o processo."""
    print(f"[ERRO] {msg}", file=sys.stderr)
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    sys.exit(exit_code)

def warn(msg: str):
    print(f"[AVISO] {msg}")

def info(msg: str):
    print(f"[INFO] {msg}")

def validate_positive_int(name, value):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} deve ser inteiro positivo (>0), recebido: {value}")

def validate_non_negative_float(name, value):
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} deve ser float não negativo (>=0), recebido: {value}")

def validate_file_exists(path, name="arquivo"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} não encontrado: {path}")

def validate_dir_exists(path, name="diretório"):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} não encontrado: {path}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=CFG_NUM_EPOCHS, type=int)
    parser.add_argument('--model', default='detr_resnet50', help='name of the model')
    parser.add_argument('--data', default=CFG_DATA_PATH, help='path to the data config file')
    parser.add_argument('-d', '--device', default='cuda', help='training device (cuda/cpu)')
    parser.add_argument('--name', default=CFG_OUT_DIR, type=str, help='training result dir name')
    parser.add_argument('--imgsz', '--img-size', dest='imgsz', default=CFG_RESIZE_TO, type=int)
    parser.add_argument('--batch', default=CFG_BATCH_SIZE, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-st', '--square-training', dest='square_training', action='store_true')
    parser.add_argument('-uta', '--use-train-aug', dest='use_train_aug', action='store_true')
    parser.add_argument('--mosaic', default=0.0, type=float)
    parser.add_argument('-vt', '--vis-transformed', dest='vis_transformed', action='store_true')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=5e-5)
    parser.add_argument('-lrb', '--lr-backbone', dest='lr_backbone', type=float, default=1e-6)
    parser.add_argument('--weight-decay', dest='weight_decay', default=1e-4, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--no-lrs', dest='no_lrs', action='store_true')
    parser.add_argument('--weights', default=None, help='path to weights if resuming training')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    return args

def load_data_config(path):
    try:
        validate_file_exists(path, "Arquivo de configuração de dados")
        with open(path, 'r') as f:
            data_configs = yaml.safe_load(f)
        required_keys = ["TRAIN_DIR_IMAGES", "TRAIN_DIR_LABELS", "VALID_DIR_IMAGES", "VALID_DIR_LABELS", "CLASSES", "NC"]
        missing = [k for k in required_keys if k not in data_configs]
        if missing:
            raise KeyError(f"Chaves ausentes no data config: {missing}")
        return data_configs
    except Exception as e:
        error(f"Falha ao carregar config de dados em {path}.", e)

def check_device(device_arg: str) -> str:
    dev = device_arg.lower()
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            warn("CUDA não disponível. Alternando para CPU.")
            return "cpu"
        # se usuário passou 'cuda:1', validar índice
        if ":" in dev:
            try:
                idx = int(dev.split(":")[1])
                n = torch.cuda.device_count()
                if idx < 0 or idx >= n:
                    warn(f"Índice de GPU inválido ({idx}). Usando cuda:0.")
                    return "cuda:0"
                return f"cuda:{idx}"
            except ValueError:
                warn(f"Formato de device '{device_arg}' inválido. Usando cuda:0.")
                return "cuda:0"
        return "cuda:0"
    elif dev == "cpu":
        return "cpu"
    else:
        warn(f"Device '{device_arg}' não reconhecido. Usando CPU.")
        return "cpu"

def safe_create_datasets(data_configs, image_size, classes, use_train_aug, mosaic, square_training):
    try:
        TRAIN_DIR_IMAGES = os.path.normpath(data_configs['TRAIN_DIR_IMAGES'])
        TRAIN_DIR_LABELS = os.path.normpath(data_configs['TRAIN_DIR_LABELS'])
        VALID_DIR_IMAGES = os.path.normpath(data_configs['VALID_DIR_IMAGES'])
        VALID_DIR_LABELS = os.path.normpath(data_configs['VALID_DIR_LABELS'])

        validate_dir_exists(TRAIN_DIR_IMAGES, "TRAIN_DIR_IMAGES")
        validate_dir_exists(TRAIN_DIR_LABELS, "TRAIN_DIR_LABELS")
        validate_dir_exists(VALID_DIR_IMAGES, "VALID_DIR_IMAGES")
        validate_dir_exists(VALID_DIR_LABELS, "VALID_DIR_LABELS")

        train_dataset = create_train_dataset(
            TRAIN_DIR_IMAGES, TRAIN_DIR_LABELS, image_size, classes,
            use_train_aug=use_train_aug, mosaic=mosaic, square_training=True if square_training else False
        )
        valid_dataset = create_valid_dataset(
            VALID_DIR_IMAGES, VALID_DIR_LABELS, image_size, classes,
            square_training=True if square_training else False
        )

        if len(train_dataset) == 0:
            warn("Dataset de treino vazio.")
        if len(valid_dataset) == 0:
            warn("Dataset de validação vazio.")

        return train_dataset, valid_dataset
    except Exception as e:
        error("Falha ao criar datasets (confira caminhos e formato dos dados).", e)

def safe_create_loaders(train_dataset, valid_dataset, batch_size, num_workers, is_distributed=False):
    try:
        if is_distributed:
            train_sampler = distributed.DistributedSampler(train_dataset)
            valid_sampler = distributed.DistributedSampler(valid_dataset, shuffle=False)
        else:
            train_sampler = RandomSampler(train_dataset)
            valid_sampler = SequentialSampler(valid_dataset)

        train_loader = create_train_loader(train_dataset, batch_size, num_workers, batch_sampler=train_sampler)
        valid_loader = create_valid_loader(valid_dataset, batch_size, num_workers, batch_sampler=valid_sampler)
        return train_loader, valid_loader
    except Exception as e:
        error("Falha ao criar data loaders (verifique batch_size/workers/memória).", e)

def load_weights_if_any(model, weights_path, device):
    if weights_path is None:
        return model
    try:
        validate_file_exists(weights_path, "Arquivo de pesos")
        map_location = device if device == "cpu" else None
        ckpt = torch.load(weights_path, map_location=map_location)
        state = ckpt.get('model_state_dict', ckpt)  # suporta checkpoints simples ou dict
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            warn(f"Parâmetros ausentes ao carregar pesos: {missing}")
        if unexpected:
            warn(f"Parâmetros inesperados no checkpoint: {unexpected}")
        info(f"Pesos carregados de {weights_path}")
        return model
    except Exception as e:
        error(f"Falha ao carregar pesos de {weights_path}.", e)

def summarize_model(model, device, batch_size, image_size):
    try:
        torchinfo.summary(
            model,
            device=device,
            input_size=(batch_size, 3, image_size, image_size),
            row_settings=["var_names"],
            col_names=["input_size", "output_size", "num_params"],
        )
    except Exception as e:
        warn(f"torchinfo.summary falhou, imprimindo estrutura básica. Detalhe: {e}")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info(f"{total_params:,} parâmetros no total, {total_trainable_params:,} treináveis.")

def build_optimizer(model, base_lr, weight_decay):
    try:
        lr_dict = {'backbone': 0.1, 'transformer': 1, 'embed': 1, 'final': 5}
        params = []
        groups = model.parameter_groups() if hasattr(model, "parameter_groups") else {"default": list(model.parameters())}
        for k, v in groups.items():
            lr = lr_dict.get(k, 1) * base_lr
            params.append({'params': v, 'lr': lr})
        optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
        return optimizer
    except Exception as e:
        error("Falha ao construir otimizador (confira parameter_groups do modelo).", e)

def main(args):
    # Validações iniciais de argumentos
    try:
        validate_positive_int("epochs", args.epochs)
        validate_positive_int("imgsz", args.imgsz)
        validate_positive_int("batch", args.batch)
        if args.workers < 0:
            raise ValueError(f"workers deve ser >=0, recebido: {args.workers}")
        validate_non_negative_float("learning_rate", args.learning_rate)
        validate_non_negative_float("weight_decay", args.weight_decay)
        validate_non_negative_float("eos_coef", args.eos_coef)
        validate_non_negative_float("mosaic", args.mosaic)
    except Exception as e:
        error("Parâmetros inválidos.", e)

    # Carregar config de dados
    data_configs = load_data_config(args.data)

    # Dispositivo
    DEVICE = check_device(args.device)
    info(f"Usando dispositivo: {DEVICE}")

    # Seeds determinísticos
    try:
        init_seeds(args.seed + 1 + RANK, deterministic=True)
    except Exception as e:
        warn(f"Falha ao inicializar seeds de forma determinística: {e}")

    # Constantes derivadas
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = len(CLASSES)
    IMAGE_SIZE = args.imgsz
    BATCH_SIZE = args.batch
    EPOCHS = args.epochs
    LR = args.learning_rate
    NUM_WORKERS = args.workers
    VISUALIZE_TRANSFORMED_IMAGES = args.vis_transformed
    IS_DISTRIBUTED = False

    # Diretório de saída e logging
    try:
        OUT_DIR = set_training_dir(args.name)
        set_log(OUT_DIR)
        info(f"Resultados serão salvos em: {OUT_DIR}")
    except Exception as e:
        error("Falha ao configurar diretório/log de treinamento.", e)

    # Datasets
    train_dataset, valid_dataset = safe_create_datasets(
        data_configs, IMAGE_SIZE, CLASSES, args.use_train_aug, args.mosaic, args.square_training
    )

    # Loaders
    train_loader, valid_loader = safe_create_loaders(
        train_dataset, valid_dataset, BATCH_SIZE, NUM_WORKERS, IS_DISTRIBUTED
    )

    # Visualização (opcional)
    if VISUALIZE_TRANSFORMED_IMAGES:
        try:
            COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
            show_tranformed_image(train_loader, DEVICE, CLASSES, COLORS)
        except Exception as e:
            warn(f"Falha ao visualizar imagens transformadas: {e}")

    # Modelo
    try:
        matcher = HungarianMatcher(cost_giou=2, cost_class=1, cost_bbox=5)
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        losses = ['labels', 'boxes', 'cardinality']
        model = DETRModel(num_classes=NUM_CLASSES, model=args.model)
    except Exception as e:
        error("Falha ao instanciar o modelo DETR.", e)

    # Pesos
    model = load_weights_if_any(model, args.weights, DEVICE)

    # Movendo para device
    try:
        model = model.to(DEVICE)
    except Exception as e:
        error("Falha ao mover o modelo para o dispositivo.", e)

    # Sumário do modelo
    summarize_model(model, DEVICE, BATCH_SIZE, IMAGE_SIZE)

    # Criterion
    try:
        criterion = SetCriterion(
            NUM_CLASSES - 1,
            matcher,
            weight_dict,
            eos_coef=args.eos_coef,
            losses=losses
        ).to(DEVICE)
    except Exception as e:
        error("Falha ao criar o critério de loss.", e)

    # Otimizador e LR scheduler
    optimizer = build_optimizer(model, LR, args.weight_decay)
    try:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[EPOCHS // 2, int(EPOCHS // 1.333)], gamma=0.5
        )
    except Exception as e:
        error("Falha ao criar o scheduler de LR.", e)

    save_best_model = SaveBestModel()

    val_map_05, val_map = [], []

    # Loop de treino
    try:
        for epoch in range(EPOCHS):
            info(f"Epoch {epoch+1}/{EPOCHS}")
            try:
                train_loss = train(
                    train_loader, model, criterion, optimizer, DEVICE, epoch=epoch
                )
            except RuntimeError as re:
                # Típicos erros: CUDA OOM
                warn(f"Falha no treino na epoch {epoch}: {re}")
                if "CUDA out of memory" in str(re):
                    warn("Sugestões: reduza batch_size, use grad_accumulation ou imagens menores.")
                raise

            # if not args.no_lrs:
            #     lr_scheduler.step()

            try:
                stats, coco_evaluator = evaluate(
                    model=model,
                    criterion=criterion,
                    postprocessors={'bbox': PostProcess()},
                    data_loader=valid_loader,
                    device=DEVICE,
                    output_dir='outputs'
                )
            except Exception as e:
                warn(f"Falha na avaliação da epoch {epoch}: {e}")
                # Decide: continuar mesmo sem avaliação? Aqui continuamos.
                continue

            try:
                coco_log(OUT_DIR, stats)
            except Exception as e:
                warn(f"Falha ao registrar COCO log: {e}")

            try:
                val_map_05.append(stats['coco_eval_bbox'][1])
                val_map.append(stats['coco_eval_bbox'])
            except Exception as e:
                warn(f"Formato inesperado de stats em avaliação: {e}")

            # Salvamentos
            try:
                save_mAP(OUT_DIR, val_map_05, val_map)
            except Exception as e:
                warn(f"Falha ao salvar gráfico de mAP: {e}")

            try:
                save_model_state(model, OUT_DIR, data_configs, args.model)
            except Exception as e:
                warn(f"Falha ao salvar estado do modelo da epoch {epoch}: {e}")

            try:
                if len(val_map) > 0:
                    save_best_model(model, val_map[-1], epoch, OUT_DIR, data_configs, args.model)
            except Exception as e:
                warn(f"Falha ao salvar melhor modelo: {e}")

    except KeyboardInterrupt:
        warn("Treinamento interrompido manualmente (KeyboardInterrupt). Salvando checkpoint...")
        try:
            # Salva um checkpoint de emergência
            emergency_ckpt = os.path.join(OUT_DIR, "interrupt_checkpoint.pt")
            torch.save({'model_state_dict': model.state_dict()}, emergency_ckpt)
            info(f"Checkpoint salvo em: {emergency_ckpt}")
        except Exception as e:
            warn(f"Falha ao salvar checkpoint de emergência: {e}")
        finally:
            sys.exit(130)
    except Exception as e:
        error("Erro não tratado durante o loop de treinamento.", e)

if __name__ == '__main__':
    try:
        args = parse_opt()
        main(args)
    except SystemExit as e:
        # argparse ou error() podem chamar sys.exit: deixe propagar o código.
        raise
    except Exception as e:
        error("Falha inesperada na execução principal.", e)
