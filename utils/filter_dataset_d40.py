from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_class_name(name: str | None) -> str:
    return (name or "").strip().casefold()


def keep_class(name: str | None, target_class: str) -> bool:
    return normalize_class_name(name) == normalize_class_name(target_class)


def indent_xml(tree: ET.ElementTree) -> None:
    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t")


def filter_xml_file(
    xml_path: Path,
    target_class: str,
    dry_run: bool,
) -> tuple[int, int, Counter[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    kept = 0
    removed = 0
    removed_by_class: Counter[str] = Counter()

    for obj in list(root.findall("object")):
        name_node = obj.find("name")
        class_name = name_node.text if name_node is not None else ""

        if keep_class(class_name, target_class):
            kept += 1
            continue

        removed += 1
        removed_by_class[(class_name or "<sem_nome>").strip() or "<sem_nome>"] += 1
        root.remove(obj)

    if removed and not dry_run:
        indent_xml(tree)
        tree.write(xml_path, encoding="utf-8", xml_declaration=False)

    return kept, removed, removed_by_class


def read_yolo_names(yaml_path: Path) -> list[str]:
    if not yaml_path.exists():
        return []

    names: list[str] = []
    in_names_block = False

    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("names:"):
            in_names_block = True
            inline_value = line.split(":", 1)[1].strip()
            if inline_value.startswith("[") and inline_value.endswith("]"):
                return [
                    item.strip().strip("'\"")
                    for item in inline_value[1:-1].split(",")
                    if item.strip()
                ]
            continue

        if in_names_block:
            if line.startswith("-"):
                names.append(line[1:].strip().strip("'\""))
                continue
            if not raw_line.startswith((" ", "\t")):
                break

    return names


def filter_yolo_file(
    label_path: Path,
    target_class_ids: set[int],
    dry_run: bool,
) -> tuple[int, int, Counter[str]]:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    kept_lines: list[str] = []
    kept = 0
    removed = 0
    removed_by_class: Counter[str] = Counter()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        try:
            class_id = int(parts[0])
        except (IndexError, ValueError):
            removed += 1
            removed_by_class["<linha_invalida>"] += 1
            continue

        if class_id in target_class_ids:
            kept += 1
            kept_lines.append(line)
            continue

        removed += 1
        removed_by_class[str(class_id)] += 1

    if removed and not dry_run:
        content = "\n".join(kept_lines)
        if content:
            content += "\n"
        label_path.write_text(content, encoding="utf-8")

    return kept, removed, removed_by_class


def filter_dataset(dataset_root: Path, target_class: str, dry_run: bool) -> None:
    xml_files = sorted((dataset_root / "annotations" / "xmls").glob("*.xml"))
    yolo_label_files = sorted(dataset_root.glob("**/labels/*.txt"))

    total_xml_kept = 0
    total_xml_removed = 0
    total_yolo_kept = 0
    total_yolo_removed = 0
    removed_by_class: Counter[str] = Counter()
    changed_files = 0

    for xml_path in xml_files:
        kept, removed, file_removed_by_class = filter_xml_file(
            xml_path=xml_path,
            target_class=target_class,
            dry_run=dry_run,
        )
        total_xml_kept += kept
        total_xml_removed += removed
        removed_by_class.update(file_removed_by_class)
        if removed:
            changed_files += 1

    yaml_path = dataset_root / "data_yolo26.yaml"
    names = read_yolo_names(yaml_path)
    target_class_ids = {
        idx for idx, name in enumerate(names) if keep_class(name, target_class)
    }

    if yolo_label_files and not target_class_ids:
        print(
            f"[WARN] Labels YOLO encontradas, mas '{target_class}' nao foi encontrado em {yaml_path}."
        )
        print("[WARN] Labels YOLO nao foram alteradas para evitar remover anotacoes corretas.")
    else:
        for label_path in yolo_label_files:
            kept, removed, file_removed_by_class = filter_yolo_file(
                label_path=label_path,
                target_class_ids=target_class_ids,
                dry_run=dry_run,
            )
            total_yolo_kept += kept
            total_yolo_removed += removed
            removed_by_class.update(file_removed_by_class)
            if removed:
                changed_files += 1

    mode = "simulacao" if dry_run else "aplicado"
    print(f"[filter_dataset_d40] Modo: {mode}")
    print(f"Dataset: {dataset_root}")
    print(f"Classe mantida: {target_class}")
    print(f"XMLs encontrados: {len(xml_files)}")
    print(f"Labels YOLO encontrados: {len(yolo_label_files)}")
    print(f"Arquivos alterados: {changed_files}")
    print(f"Objetos XML mantidos: {total_xml_kept}")
    print(f"Objetos XML removidos: {total_xml_removed}")
    print(f"Objetos YOLO mantidos: {total_yolo_kept}")
    print(f"Objetos YOLO removidos: {total_yolo_removed}")

    if removed_by_class:
        print("Removidos por classe/id:")
        for class_name, count in removed_by_class.most_common():
            print(f"  {class_name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove anotacoes que nao pertencem a classe alvo no dataset/all."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/all"),
        help="Raiz do dataset. Padrao: dataset/all",
    )
    parser.add_argument(
        "--class-name",
        default="D40",
        help="Classe que deve ser mantida. Padrao: D40",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria removido sem alterar arquivos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_root}")

    filter_dataset(
        dataset_root=dataset_root,
        target_class=args.class_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
