# Contrato de dataset tileado

Este repositório consome datasets de detecção em COCO com folds explícitos. Para
datasets tileados gerados pelo `slice_inference_api`, use:

```bash
DATASET_ROOT=dataset/asahi_rect python src/main.py
```

O contrato principal é cross-fold. O layout achatado `train/`, `val/`, `test/` na
raiz é aceito apenas como fallback legado do módulo de treinamento.

## Layout principal

```text
dataset/asahi_rect/
├── dataset_manifest.json
├── filesJSON/
│   ├── fold_1_train.json
│   ├── fold_1_val.json
│   ├── fold_1_test.json
│   └── ...
├── filesJSON_infos/
│   ├── fold_1.yaml
│   ├── fold_1_stats.json
│   └── ...
├── fold_1/
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   ├── val/labels/
│   ├── test/images/
│   └── test/labels/
├── fold_2/
│   └── ...
└── fold_5/
    └── ...
```

## Regra de resolução

Cada arquivo em `filesJSON/` é COCO e deve ser resolvido contra o split do fold
correspondente:

- `filesJSON/fold_N_train.json` referencia imagens em `fold_N/train/images/`
- `filesJSON/fold_N_val.json` referencia imagens em `fold_N/val/images/`
- `filesJSON/fold_N_test.json` referencia imagens em `fold_N/test/images/`

Regra geral:

```text
annotation_path = DATASET_ROOT/filesJSON/{fold}_{split}.json
images_dir      = DATASET_ROOT/{fold}/{split}/images
labels_dir      = DATASET_ROOT/{fold}/{split}/labels
```

Exemplo:

```text
DATASET_ROOT=dataset/asahi_rect

fold_1_train.json -> dataset/asahi_rect/fold_1/train/images
fold_1_val.json   -> dataset/asahi_rect/fold_1/val/images
fold_1_test.json  -> dataset/asahi_rect/fold_1/test/images
```

## Fonte bruta

A fonte original do dataset fica fora dos modos de recorte:

```text
dataset/all/
├── imagens originais
├── _annotations.coco.json
└── _annotations_clean.coco.json
```

Experimentos tileados devem consumir `dataset/<modo>`, não `dataset/all`.

## Manifesto recomendado para asahi_rect

```json
{
  "contract_version": "1.0",
  "dataset_name": "asahi_rect",
  "dataset_type": "tiled_detection_crossfold",
  "annotation_format": "coco",
  "category_id_base": 1,
  "splits": ["train", "val", "test"],
  "folds": ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
  "layout": {
    "annotations_dir": "filesJSON",
    "annotation_pattern": "{fold}_{split}.json",
    "image_dir_pattern": "{fold}/{split}/images",
    "label_dir_pattern": "{fold}/{split}/labels",
    "fold_info_dir": "filesJSON_infos",
    "fold_yaml_pattern": "{fold}.yaml",
    "fold_stats_pattern": "{fold}_stats.json"
  },
  "tiling": {
    "mode": "asahi_rect",
    "evaluation_mode": "basic",
    "tile_shape": "rectangular",
    "variable_tile_size": true,
    "tiles_are_primary_samples": true,
    "requires_reconstruction": false
  },
  "classes": [
    {
      "id": 1,
      "name": "insect"
    }
  ]
}
```

Para `asahi_rect`, `images[].width` e `images[].height` em cada COCO JSON são a
fonte da verdade. Não assuma tamanho fixo global.

## Regras COCO exigidas

- `images[].id` deve ser único dentro do JSON.
- `images[].file_name`, `images[].width` e `images[].height` são obrigatórios.
- `images[].file_name` deve ser relativo a `DATASET_ROOT/{fold}/{split}/images`.
- `images[].width` e `images[].height` devem ser positivos.
- `annotations[].id` deve ser único dentro do JSON.
- `annotations[].image_id` deve existir em `images[]`.
- `annotations[].category_id` deve existir em `categories[]`.
- `annotations[].bbox` deve estar em COCO absoluto `[x, y, width, height]`.
- `bbox` deve ter origem não negativa, dimensões positivas e ficar dentro da imagem.
- `categories[].id` deve ser positivo.

## Modos de avaliação

Para os datasets gerados pelo `slice_inference_api`, o modo recomendado é:

```json
"evaluation_mode": "basic"
```

Isso significa que tiles/imagens do split são avaliados como amostras primárias.
Não exigir reconstrução/agregação para `asahi_rect` neste contrato.

## Fallback legado

O layout abaixo não é o contrato principal:

```text
dataset/asahi_rect/
├── train/
├── val/
└── test/
```

O módulo de treinamento ainda tenta esse layout como fallback quando não encontra
`DATASET_ROOT/{fold}/{split}/images`.

## Validação

Rode:

```bash
python scripts/validate_dataset_contract.py --root dataset/asahi_rect
```
