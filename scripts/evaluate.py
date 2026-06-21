"""
avalia predicoes de pegadas de predios contra ground truth.

calcula metricas a nivel de instancia: IoU por predio e F1-score
(precisao/recall) onde um match eh contado quando IoU > 0.5 entre
um poligono predito e um ground truth.

ground truth pode ser um diretorio de arquivos GeoJSON por tile (formato
SpaceNet 2) ou um unico arquivo GeoJSON.

Usage:
    python evaluate.py \
        --predictions predictions.geojson \
        --ground_truth /path/to/spacenet/train/geojson/buildings/ \
        --output results.json
"""

import argparse
import glob
import json
import os
import re

import geopandas as gpd
import numpy as np
from shapely.geometry import shape


def extract_image_number(filename):
    """extrai o ID numerico do tile de um nome de arquivo SpaceNet.

    Args:
        filename: string do nome do arquivo.

    Returns:
        string do numero da imagem, ou None se nao encontrado.
    """
    match = re.search(r'img(\d+)', filename)
    if match:
        return match.group(1)
    return None


def compute_iou(polygon1, polygon2):
    """calcula intersection over union entre dois poligonos.

    Args:
        polygon1: primeiro poligono shapely.
        polygon2: segundo poligono shapely.

    Returns:
        score IoU (float entre 0 e 1).
    """
    if polygon1.is_empty or polygon2.is_empty:
        return 0.0

    try:
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        if union == 0:
            return 0.0
        return intersection / union
    except Exception:
        return 0.0


def match_predictions_to_ground_truth(pred_polygons, gt_polygons,
                                       iou_threshold=0.5):
    """faz matching dos poligonos preditos com ground truth usando IoU.

    usa matching guloso: iterativamente pega o par com maior IoU
    acima do limiar.

    Args:
        pred_polygons: lista de poligonos shapely preditos.
        gt_polygons: lista de poligonos shapely ground truth.
        iou_threshold: IoU minimo pra considerar um match.

    Returns:
        dicionario com resultados do matching:
        - tp: numero de verdadeiros positivos
        - fp: numero de falsos positivos
        - fn: numero de falsos negativos
        - ious: lista de valores IoU dos pares matcheados
        - matches: lista de tuplas (gt_idx, pred_idx, iou)
    """
    if not gt_polygons and not pred_polygons:
        return {"tp": 0, "fp": 0, "fn": 0, "ious": [], "matches": []}

    if not gt_polygons:
        return {
            "tp": 0,
            "fp": len(pred_polygons),
            "fn": 0,
            "ious": [],
            "matches": [],
        }

    if not pred_polygons:
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(gt_polygons),
            "ious": [],
            "matches": [],
        }

    # calcula matriz de IoU
    iou_matrix = np.zeros((len(gt_polygons), len(pred_polygons)))
    for i, gt in enumerate(gt_polygons):
        for j, pred in enumerate(pred_polygons):
            iou_matrix[i, j] = compute_iou(gt, pred)

    # matching: atribui melhor predicao pra cada ground truth
    matched_gt = set()
    matched_pred = set()
    matches = []
    ious = []

    # ordena por IoU decrescente pro matching guloso
    while True:
        if iou_matrix.size == 0:
            break

        best_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        best_iou = iou_matrix[best_idx]

        if best_iou < iou_threshold:
            break

        gt_idx, pred_idx = best_idx
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            matches.append((int(gt_idx), int(pred_idx), float(best_iou)))
            ious.append(float(best_iou))

        # marca linha e coluna como usadas
        iou_matrix[gt_idx, :] = 0
        iou_matrix[:, pred_idx] = 0

    tp = len(matches)
    fp = len(pred_polygons) - tp
    fn = len(gt_polygons) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ious": ious,
        "matches": matches,
    }


def load_ground_truth(gt_path):
    """carrega poligonos ground truth de arquivo(s) GeoJSON.

    suporta:
    - arquivo geoJSON unico
    - diretorio de arquivos GeoJSON por tile (formato SpaceNet 2)

    as chaves sao derivadas do padrao img{N} nos nomes de arquivo pra
    fazer matching com os image_ids das predicoes.

    Args:
        gt_path: caminho pra um arquivo GeoJSON ou diretorio de GeoJSONs.

    Returns:
        dicionario mapeando chaves de imagem pra listas de poligonos.
    """
    gt_polygons = {}

    if os.path.isfile(gt_path):
        gdf = gpd.read_file(gt_path)
        polys = [
            geom for geom in gdf.geometry
            if geom is not None and geom.is_valid and not geom.is_empty
        ]
        key = os.path.splitext(os.path.basename(gt_path))[0]
        gt_polygons[key] = polys

    elif os.path.isdir(gt_path):
        geojson_files = sorted(glob.glob(os.path.join(gt_path, "*.geojson")))
        for f in geojson_files:
            try:
                gdf = gpd.read_file(f)
                polys = [
                    geom for geom in gdf.geometry
                    if geom is not None and geom.is_valid and not geom.is_empty
                ]
                # usa o numero da img como chave pro matching
                img_num = extract_image_number(os.path.basename(f))
                if img_num:
                    key = img_num
                else:
                    key = os.path.splitext(os.path.basename(f))[0]
                gt_polygons[key] = polys
            except Exception as e:
                print(f"    AVISO: nao consegui ler {f}: {e}")
    else:
        raise FileNotFoundError(f"caminho do ground truth nao encontrado: {gt_path}")

    return gt_polygons


def load_predictions(pred_path):
    """carrega poligonos preditos de um arquivo GeoJSON.

    agrupa predicoes por image_id. o image_id deve conter o padrao
    img{N} pra fazer matching com ground truth.

    Args:
        pred_path: caminho pro arquivo GeoJSON de predicoes.

    Returns:
        dicionario mapeando chaves de imagem pra listas de poligonos.
    """
    gdf = gpd.read_file(pred_path)

    pred_polygons = {}
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        image_id = row.get("image_id", "unknown")
        # extrai numero da img pro matching com GT
        img_num = extract_image_number(str(image_id))
        key = img_num if img_num else image_id

        if key not in pred_polygons:
            pred_polygons[key] = []
        pred_polygons[key].append(geom)

    return pred_polygons


def evaluate(predictions_path, ground_truth_path, iou_threshold=0.5,
             split_filter=None):
    """roda avaliacao completa das predicoes contra ground truth.

    Args:
        predictions_path: caminho pro GeoJSON de predicoes.
        ground_truth_path: caminho pro arquivo ou diretorio GeoJSON de ground truth.
        iou_threshold: limiar de IoU pro matching.
        split_filter: conjunto opcional de strings de image ID. quando informado,
            so tiles GT cuja chave esta nesse conjunto sao avaliados.

    Returns:
        dicionario com resultados da avaliacao.
    """
    print(f"carregando predicoes de: {predictions_path}")
    pred_by_image = load_predictions(predictions_path)
    total_preds = sum(len(v) for v in pred_by_image.values())
    print(f"    total de predios preditos: {total_preds}")
    print(f"    imagens com predicoes: {len(pred_by_image)}")

    print(f"carregando ground truth de: {ground_truth_path}")
    gt_by_image = load_ground_truth(ground_truth_path)

    if split_filter is not None:
        gt_by_image = {k: v for k, v in gt_by_image.items() if k in split_filter}
    total_gt = sum(len(v) for v in gt_by_image.values())
    print(f"    total de predios ground truth: {total_gt}")
    print(f"    imagens com ground truth: {len(gt_by_image)}")

    # agrega resultados
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []
    per_image_results = {}

    # faz matching por chave de imagem (numero da img)
    all_keys = sorted(
        set(list(pred_by_image.keys()) + list(gt_by_image.keys()))
    )

    for key in all_keys:
        gt_polys = gt_by_image.get(key, [])
        pred_polys = pred_by_image.get(key, [])

        result = match_predictions_to_ground_truth(
            pred_polys, gt_polys, iou_threshold
        )

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        all_ious.extend(result["ious"])

        per_image_results[key] = {
            "gt_count": len(gt_polys),
            "pred_count": len(pred_polys),
            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],
            "mean_iou": (
                float(np.mean(result["ious"])) if result["ious"] else 0.0
            ),
        }

    # calcula metricas agregadas
    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )
    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    results = {
        "iou_threshold": iou_threshold,
        "total_ground_truth": total_gt,
        "total_predictions": total_preds,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_iou": mean_iou,
        "per_image": per_image_results,
    }

    return results


def print_report(results):
    """imprime relatorio formatado da avaliacao.

    Args:
        results: dicionario com resultados da avaliacao.
    """
    print("\n" + "=" * 60)
    print("RELATORIO DE AVALIACAO - SpaceNet 2 AOI_3_Paris")
    print("=" * 60)
    print(f"    limiar IoU: {results['iou_threshold']}")
    print(f"    predios ground truth: {results['total_ground_truth']}")
    print(f"    predios preditos: {results['total_predictions']}")
    print(f"    verdadeiros positivos: {results['true_positives']}")
    print(f"    falsos positivos: {results['false_positives']}")
    print(f"    falsos negativos: {results['false_negatives']}")
    print(f"    precisao: {results['precision']:.4f}")
    print(f"    recall: {results['recall']:.4f}")
    print(f"    F1-score: {results['f1_score']:.4f}")
    print(f"    IoU medio (matcheados): {results['mean_iou']:.4f}")
    print("=" * 60)

    # resumo por imagem (top 10)
    per_image = results.get("per_image", {})
    if per_image:
        print("\nresultados por imagem (primeiras 10):")
        for i, (key, img_res) in enumerate(sorted(per_image.items())):
            if i >= 10:
                print(f"  ... e mais {len(per_image) - 10} imagens")
                break
            print(
                f"  img{key}: GT={img_res['gt_count']}, "
                f"Pred={img_res['pred_count']}, "
                f"TP={img_res['tp']}, "
                f"IoU={img_res['mean_iou']:.3f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="avalia predicoes de pegadas de predios (IoU a nivel "
                    "de instancia + F1-score)."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="caminho pro arquivo GeoJSON de predicoes.",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="caminho pro arquivo ou diretorio GeoJSON de ground truth "
             "(ex: spacenet/train/geojson/buildings/).",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="limiar de IoU pro matching (padrao: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="arquivo JSON de saida pros resultados.",
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default=None,
        help="caminho pro splits.json. com --split, so tiles GT daquele "
             "split sao avaliados (deve bater com o split do postprocess).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="qual split avaliar (requer --splits_json).",
    )
    args = parser.parse_args()

    # filtra GT pra um split especifico se solicitado (deve bater com o split do postprocess)
    gt_path = args.ground_truth
    split_filter = None
    if args.splits_json and args.split:
        with open(args.splits_json) as f:
            split_filter = set(str(i) for i in json.load(f).get(args.split, []))
        print(f"  restringindo GT ao split '{args.split}': {len(split_filter)} tiles")

    results = evaluate(
        args.predictions,
        args.ground_truth,
        args.iou_threshold,
        split_filter=split_filter,
    )

    print_report(results)

    # salva resultados (sem per_image pra json mais limpo)
    results_save = {k: v for k, v in results.items() if k != "per_image"}
    results_save["num_images_evaluated"] = len(results.get("per_image", {}))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results_save, f, indent=2)

    print(f"\nresultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
