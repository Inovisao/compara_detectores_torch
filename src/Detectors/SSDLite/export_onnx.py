"""
Exporta o melhor checkpoint SSDLite para ONNX.

Uso:
    python src/Detectors/SSDLite/export_onnx.py \
        --checkpoint model_checkpoints/fold_1/SSDLite/best.pth \
        --output ssdlite_mobilenetv2.onnx

Arquivos gerados:
    ssdlite_mobilenetv2.onnx        — grafo ONNX (backbone + head)
    ssdlite_mobilenetv2.anchors.npy — âncoras padrão (3234, 4) para decodificar caixas

O modelo exportado recebe um tensor 1×3×320×320 com valores em [0, 1]
(resultado de cv2 → float32 / 255) e devolve:
    - bbox_regression : (1, 3234, 4)            offsets de regressão por âncora
    - cls_logits      : (1, 3234, num_classes)   logits brutos por classe

A normalização ImageNet está embutida no grafo — não é necessário aplicá-la
antes de chamar o modelo.  NMS e decodificação das âncoras ficam em
onnx_predict.py (use SSDLitePredictor para inferência no app).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Garante que os módulos do projeto são encontráveis
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Detectors.SSDLite.RunSSDLite import _build_model
from Detectors.SSDLite.config import get_config


class _SSDLiteONNXWrapper(nn.Module):
    """
    Wrapper backbone + head com normalização ImageNet embutida.

    Entrada : tensor (1, 3, 320, 320) com valores em [0, 1]
    Saídas  : bbox_regression  (1, 3234, 4)
              cls_logits        (1, 3234, num_classes)
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, ssd_model: nn.Module) -> None:
        super().__init__()
        self.backbone = ssd_model.backbone
        self.head = ssd_model.head
        self.register_buffer(
            "mean", torch.tensor(self._MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self._STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        features = list(self.backbone(x).values())
        outputs = self.head(features)
        return outputs["bbox_regression"], outputs["cls_logits"]


def _load_ssd_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = int(checkpoint["num_classes"])
    cfg = get_config()
    if checkpoint.get("backbone"):
        cfg = replace(cfg, backbone=checkpoint["backbone"])
    model = _build_model(num_classes, cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _save_anchors(ssd_model: nn.Module, output_path: Path) -> None:
    """Gera e salva as âncoras padrão usadas pelo modelo."""
    from torchvision.models.detection.image_list import ImageList

    dummy = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        features = list(ssd_model.backbone(dummy).values())
        image_list = ImageList(dummy, [(320, 320)])
        anchors = ssd_model.anchor_generator(image_list, features)

    anchors_np = anchors[0].numpy()  # (3234, 4) como (cx, cy, w, h) normalizado
    anchor_path = output_path.parent / (output_path.stem + ".anchors.npy")
    np.save(str(anchor_path), anchors_np)
    print(f"[export] âncoras salvas: {anchor_path}  shape={anchors_np.shape}")


def export(checkpoint_path: Path, output_path: Path, opset: int = 17) -> None:
    print(f"[export] carregando checkpoint: {checkpoint_path}")
    ssd_model = _load_ssd_model(checkpoint_path)
    wrapper = _SSDLiteONNXWrapper(ssd_model)

    dummy_input = torch.zeros(1, 3, 320, 320)  # [0, 1] — normalização é interna

    print(f"[export] exportando para ONNX (opset {opset}): {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["image"],
        output_names=["bbox_regression", "cls_logits"],
        dynamic_axes={
            "image":          {0: "batch"},
            "bbox_regression":{0: "batch"},
            "cls_logits":     {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print("[export] ONNX concluído.")

    _save_anchors(ssd_model, output_path)
    _verify(output_path, dummy_input)


def _verify(onnx_path: Path, dummy_input: torch.Tensor) -> None:
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"image": dummy_input.numpy()})
        print(f"[verify] bbox_regression : {out[0].shape}")
        print(f"[verify] cls_logits      : {out[1].shape}")
        print("[verify] inferência ONNX Runtime OK.")
    except ImportError:
        print("[verify] onnxruntime não instalado — pulando verificação.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta SSDLite para ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Caminho para best.pth")
    parser.add_argument("--output", default=Path("ssdlite_mobilenetv2.onnx"), type=Path,
                        help="Arquivo .onnx de saída")
    parser.add_argument("--opset", default=17, type=int,
                        help="Versão do opset ONNX (padrão: 17)")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.opset)
