import json
import argparse
import os
import sys
import shutil
import glob
import math
import cv2


def confirmar(prompt):
    print(prompt, file=sys.stderr)
    resp = input()
    return resp.strip().lower() == "sim"


def cli():
    parser = argparse.ArgumentParser(
        description="Recorta imagens em tiles 640x640 IN-PLACE em dataset/all/train/ "
                    "e reescreve o _annotations.coco.json correspondente."
    )
    parser.add_argument('-annotations',
                        default='../dataset/all/train/_annotations.coco.json',
                        help='Caminho para o JSON COCO original.')
    parser.add_argument('-images',
                        default='../dataset/all/train/',
                        help='Pasta com as imagens a serem recortadas (in-place).')
    parser.add_argument('-crop-size', dest='crop_size', type=int, default=640,
                        help='Tamanho do lado do tile (default: 640).')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Pula o prompt de confirmacao.')
    parser.add_argument('--keep-originals', action='store_true',
                        help='Move os originais para train/_originais/ antes de recortar.')
    parser.add_argument('-folds-dir', dest='folds_dir',
                        default='../dataset/all/filesJSON/',
                        help='Pasta com os arquivos fold_*_*.json para atualizar apos o recorte.')
    parser.add_argument('--no-folds', dest='no_folds', action='store_true',
                        help='Nao atualizar os arquivos de dobra em -folds-dir.')
    return parser.parse_args()


def extensao_valida(nome):
    return nome.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff'))


def carregar_coco(caminho):
    with open(caminho, 'rt', encoding='utf-8') as f:
        return json.load(f)


def salvar_coco(caminho, data):
    with open(caminho, 'wt', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=False)


def distribuir_anotacao(ann, tiles, crop_size):
    """Divide uma anotacao COCO original pelos seus tiles, clipando nas bordas.

    Retorna uma lista de dicts de anotacao SEM o campo 'id' (o caller deve
    atribuir os ids sequencialmente).
    """
    bx, by, bw, bh = ann['bbox']
    categorias_id = ann['category_id']
    out = []
    for (r, c, x0, y0, novo_id) in tiles:
        x = bx - x0
        y = by - y0
        x_cl = max(0.0, min(float(x), float(crop_size)))
        y_cl = max(0.0, min(float(y), float(crop_size)))
        w_cl = max(0.0, min(float(x + bw), float(crop_size)) - x_cl)
        h_cl = max(0.0, min(float(y + bh), float(crop_size)) - y_cl)
        if w_cl > 0 and h_cl > 0:
            out.append({
                'image_id': novo_id,
                'category_id': categorias_id,
                'bbox': [x_cl, y_cl, w_cl, h_cl],
                'area': w_cl * h_cl,
                'iscrowd': ann.get('iscrowd', 0),
                'segmentation': ann.get('segmentation', []),
            })
    return out


def atualizar_folds(fold_dir, tile_id_por_orig, img_entry_por_novo_id, crop_size):
    """Reescreve cada fold_*_*.json expandindo imagens e anotacoes para tiles.

    Mantem o particionamento original: uma imagem original que estava em um
    conjunto (train/val/test) continua nesse conjunto, e todos os seus tiles
    vao junto. Os IDs das imagens/tiles correspondem aos do _annotations.coco.json
    recortado. Os IDs das anotacoes sao renumerados a partir de 1 dentro de cada
    arquivo de dobra.
    """
    if not os.path.isdir(fold_dir):
        print(f"Aviso: pasta de dobras nao encontrada: {fold_dir}. Dobras nao atualizadas.")
        return
    fold_files = sorted(glob.glob(os.path.join(fold_dir, 'fold_*_*.json')))
    if not fold_files:
        print(f"Aviso: nenhum arquivo fold_*_*.json encontrado em {fold_dir}. Dobras nao atualizadas.")
        return

    for fold_path in fold_files:
        coco = carregar_coco(fold_path)
        imgs_fold = coco.get('images', [])
        anns_fold = coco.get('annotations', [])

        novas_imgs = []
        novas_anns = []
        prox_id_ann = 1

        for img in imgs_fold:
            tiles = tile_id_por_orig.get(img['id'], [])
            for (r, c, x0, y0, novo_id) in tiles:
                entry = img_entry_por_novo_id.get(novo_id)
                if entry is None:
                    continue
                novas_imgs.append(dict(entry))

        for ann in anns_fold:
            tiles = tile_id_por_orig.get(ann['image_id'], [])
            for nova in distribuir_anotacao(ann, tiles, crop_size):
                nova['id'] = prox_id_ann
                prox_id_ann += 1
                novas_anns.append(nova)

        coco['images'] = novas_imgs
        coco['annotations'] = novas_anns

        salvar_coco(fold_path, coco)
        print(f"Dobra atualizada: {os.path.basename(fold_path)} "
              f"(imgs: {len(novas_imgs)}, anns: {len(novas_anns)})")


def recortar_e_reescrever(args):
    crop_size = args.crop_size
    annotations_path = args.annotations
    images_dir = args.images

    if not os.path.isfile(annotations_path):
        print(f"Erro: anotacoes nao encontradas: {annotations_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(images_dir):
        print(f"Erro: pasta de imagens nao existe: {images_dir}", file=sys.stderr)
        sys.exit(1)

    coco = carregar_coco(annotations_path)
    imagens_originais = coco['images']
    anotacoes_originais = coco['annotations']
    categories = coco.get('categories', [])
    info = coco.get('info', {})
    licenses = coco.get('licenses', [])

    # Validar consistencia entre dimensoes do JSON e o tamanho real das imagens.
    for img in imagens_originais:
        path = os.path.join(images_dir, img['file_name'])
        if not os.path.isfile(path):
            print(f"Erro: imagem declarada no JSON nao existe no disco: {path}", file=sys.stderr)
            sys.exit(1)
        real = cv2.imread(path)
        if real is None:
            print(f"Erro: falha ao ler a imagem: {path}", file=sys.stderr)
            sys.exit(1)
        h_real, w_real = real.shape[:2]
        if w_real != img.get('width') or h_real != img.get('height'):
            print(
                f"Erro: dimensoes declaradas ({img.get('width')}x{img.get('height')}) "
                f" != reais ({w_real}x{h_real}) para {img['file_name']}. "
                f"Rode antes com JSON atualizado.", file=sys.stderr
            )
            sys.exit(1)

    # Pasta de backup opcional
    backup_dir = os.path.join(images_dir, '_originais')
    if args.keep_originals:
        os.makedirs(backup_dir, exist_ok=True)

    novas_imagens = []
    novas_anotacoes = []
    prox_id_img = 1
    prox_id_ann = 1
    total_tiles = 0

    # Mapeia (id_original, row, col) -> novo_id
    tile_id_por_orig = {}
    # Mapeia novo_id -> entrada da imagem (memoria) para reuso nos folds
    img_entry_por_novo_id = {}

    for img in imagens_originais:
        fname = img['file_name']
        path = os.path.join(images_dir, fname)
        matriz = cv2.imread(path)
        h, w = matriz.shape[:2]
        cols = math.ceil(w / crop_size)
        rows = math.ceil(h / crop_size)
        padded_w = cols * crop_size
        padded_h = rows * crop_size
        if padded_w != w or padded_h != h:
            matriz = cv2.copyMakeBorder(
                matriz,
                0, padded_h - h,
                0, padded_w - w,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            h, w = padded_h, padded_w

        base, ext = os.path.splitext(fname)

        for r in range(rows):
            for c in range(cols):
                x0 = c * crop_size
                y0 = r * crop_size
                tile = matriz[y0:y0 + crop_size, x0:x0 + crop_size]
                nome_tile = f"{base}_x{c}_y{r}{ext}"
                path_tile = os.path.join(images_dir, nome_tile)
                # Salvar antes de apagar original (nomes sao diferentes, nao collide)
                cv2.imwrite(path_tile, tile)

                novo_id = prox_id_img
                prox_id_img += 1
                entry = {
                    'id': novo_id,
                    'file_name': nome_tile,
                    'width': crop_size,
                    'height': crop_size,
                }
                novas_imagens.append(entry)
                img_entry_por_novo_id[novo_id] = entry
                tile_id_por_orig.setdefault(img['id'], []).append((r, c, x0, y0, novo_id))
                total_tiles += 1

        # Descartar ou arquivar o original
        if args.keep_originals:
            destino = os.path.join(backup_dir, fname)
            shutil.move(path, destino)
        else:
            os.remove(path)

    # Redistribuir anotacoes pelos tiles (clipando nas bordas)
    for ann in anotacoes_originais:
        tiles = tile_id_por_orig.get(ann['image_id'], [])
        for nova in distribuir_anotacao(ann, tiles, crop_size):
            nova['id'] = prox_id_ann
            prox_id_ann += 1
            novas_anotacoes.append(nova)

    novo_coco = {
        'info': info,
        'licenses': licenses,
        'images': novas_imagens,
        'annotations': novas_anotacoes,
        'categories': categories,
    }
    backup_json = annotations_path + '.bak'
    shutil.copy2(annotations_path, backup_json)
    salvar_coco(annotations_path, novo_coco)
    print(f"Backup do JSON salvo em: {backup_json}")

    print(f"Imagens originais: {len(imagens_originais)}")
    print(f"Tiles gerados:     {total_tiles}")
    print(f"Anotacoes originais: {len(anotacoes_originais)}")
    print(f"Anotacoes finais:    {len(novas_anotacoes)}")
    if args.keep_originals:
        print(f"Originais preservados em: {backup_dir}")

    if not args.no_folds:
        print(f"Atualizando dobras em: {args.folds_dir}")
        atualizar_folds(args.folds_dir, tile_id_por_orig, img_entry_por_novo_id, crop_size)


def main():
    args = cli()
    crop_size = args.crop_size
    annotations_path = args.annotations
    images_dir = args.images

    if not args.yes:
        msg = (
            "AVISO: esta operacao vai PERMANENTEMENTE recortar e substituir\n"
            f"as imagens em {images_dir} por tiles de {crop_size}x{crop_size}.\n"
            "Os arquivos originais serao perdidos"
            + (" (movidos para _originais/)" if args.keep_originals else " (apagados).") + "\n"
            f"O {os.path.basename(annotations_path)} tambem sera reescrito.\n"
        )
        if not args.no_folds:
            msg += f"Os arquivos fold_*_*.json em {args.folds_dir} tambem serao reescritos (backup .bak).\n"
        msg += 'Prosseguir? (digite "sim" para continuar): '
        if not confirmar(msg.strip()):
            print("Operacao cancelada.", file=sys.stderr)
            sys.exit(1)

    recortar_e_reescrever(args)


if __name__ == "__main__":
    main()