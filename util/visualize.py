"""gera figuras pro relatorio e video.

tres tipos de visualizacao:
  overlay  - predicao vs ground truth sobreposta na imagem de satelite
  curves   - curvas de loss e IoU do treino a partir do training_history.json
  grid     - tiles de amostra mostrando variedade do dataset (vazio vs denso)

Usage:
    python visualize.py overlay \
        --data_dir /path/to/spacenet \
        --predictions predictions.geojson \
        --splits_json splits.json --split test \
        --output_dir figures/

    python visualize.py curves \
        --history training_history.json \
        --output_dir figures/

    python visualize.py grid \
        --data_dir /path/to/spacenet \
        --output_dir figures/
"""

import argparse
import json
import os
import re

import cv2
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from shapely.geometry import shape


def normalize_percentile(image, low=2, high=98):
    img = image.astype(np.float32)
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    if p_high > p_low:
        img = np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        img = np.zeros_like(img)
    return img


def load_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        n = min(3, src.count)
        bands = [src.read(i + 1) for i in range(n)]
        while len(bands) < 3:
            bands.append(bands[-1])
        img = np.stack(bands, axis=-1)
    return normalize_percentile(img)


def extract_image_number(filename):
    m = re.search(r'img(\d+)', filename)
    return m.group(1) if m else None


def draw_polygons(ax, polygons, color, alpha=0.3, linewidth=1.5):
    for poly in polygons:
        if poly.is_empty:
            continue
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=alpha, fc=color, ec=color, linewidth=linewidth)


def make_overlay(image, pred_polys=None, gt_polys=None, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1: so a imagem
    axes[0].imshow(image)
    axes[0].set_title("Imagem de satelite")
    axes[0].axis("off")

    # 2: overlay de predicao
    axes[1].imshow(image)
    if pred_polys:
        draw_polygons(axes[1], pred_polys, color="red", alpha=0.4)
    axes[1].set_title(f"Predicao ({len(pred_polys or [])} predios)")
    axes[1].axis("off")

    # 3: comparacao (pred=vermelho, gt=verde, overlap=amarelo)
    axes[2].imshow(image)
    if gt_polys:
        draw_polygons(axes[2], gt_polys, color="lime", alpha=0.3)
    if pred_polys:
        draw_polygons(axes[2], pred_polys, color="red", alpha=0.3)
    legend_items = []
    if pred_polys:
        legend_items.append(mpatches.Patch(color="red", alpha=0.5, label="Predicao"))
    if gt_polys:
        legend_items.append(mpatches.Patch(color="lime", alpha=0.5, label="Ground Truth"))
    if legend_items:
        axes[2].legend(handles=legend_items, loc="upper right", fontsize=9)
    axes[2].set_title("Predicao vs Ground Truth" if gt_polys else "Predicao")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def geo_to_pixel(polygon, transform):
    """
    converte poligono geografico pra coordenadas de pixel usando transformada inversa.

    trata coordenadas 2D e 3D - poligonos GT do SpaceNet carregam um Z=0
    que precisa ser ignorado (senao desempacotar 'lon, lat' da erro).
    """
    inv = ~transform
    coords = []
    for c in polygon.exterior.coords:
        col, row = inv * (c[0], c[1])
        coords.append((col, row))
    from shapely.geometry import Polygon as ShapelyPolygon
    return ShapelyPolygon(coords)


def cmd_overlay(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # carrega predicoes
    pred_gdf = gpd.read_file(args.predictions)

    # agrupa predicoes por image_id
    pred_by_img = {}
    for _, row in pred_gdf.iterrows():
        img_id = extract_image_number(str(row.get("image_id", "")))
        if img_id and row.geometry and not row.geometry.is_empty:
            pred_by_img.setdefault(img_id, []).append(row.geometry)

    # determina quais tiles visualizar
    if args.tile_ids:
        tile_ids = args.tile_ids
    elif args.splits_json and args.split:
        with open(args.splits_json) as f:
            splits = json.load(f)
        split_ids = [str(i) for i in splits.get(args.split, [])]
        # pega tiles com mais predicoes (os mais interessantes)
        scored = [(tid, len(pred_by_img.get(tid, []))) for tid in split_ids]
        scored.sort(key=lambda x: -x[1])
        tile_ids = [t[0] for t in scored[:args.num_tiles]]
    else:
        tile_ids = list(pred_by_img.keys())[:args.num_tiles]

    img_dir = os.path.join(args.data_dir, "train", "RGB-PanSharpen")
    gt_dir = os.path.join(args.data_dir, "train", "geojson", "buildings")

    for tid in tile_ids:
        tif_path = os.path.join(img_dir, f"RGB-PanSharpen_AOI_3_Paris_img{tid}.tif")
        if not os.path.exists(tif_path):
            print(f"  img{tid}: .tif nao encontrado, pulando")
            continue

        image = load_rgb(tif_path)

        # pega transformada pra conversao geo->pixel
        with rasterio.open(tif_path) as src:
            transform = src.transform

        # predicoes (geo->pixel)
        pred_polys_px = []
        for poly in pred_by_img.get(tid, []):
            try:
                pred_polys_px.append(geo_to_pixel(poly, transform))
            except Exception:
                pass

        # ground truth (geo->pixel)
        gt_polys_px = []
        gt_path = os.path.join(gt_dir, f"buildings_AOI_3_Paris_img{tid}.geojson")
        if os.path.exists(gt_path):
            gt_gdf = gpd.read_file(gt_path)
            with rasterio.open(tif_path) as src:
                gt_crs = gt_gdf.crs
                tif_crs = src.crs
                if gt_crs and tif_crs and str(gt_crs) != str(tif_crs):
                    try:
                        gt_gdf = gt_gdf.to_crs(tif_crs)
                    except Exception:
                        pass
            for geom in gt_gdf.geometry:
                if geom and geom.is_valid and not geom.is_empty:
                    try:
                        gt_polys_px.append(geo_to_pixel(geom, transform))
                    except Exception:
                        pass

        title = f"img{tid} - Pred: {len(pred_polys_px)}, GT: {len(gt_polys_px)}"
        fig = make_overlay(image, pred_polys_px, gt_polys_px or None, title)

        out_path = os.path.join(args.output_dir, f"overlay_img{tid}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  img{tid}: {len(pred_polys_px)} pred, {len(gt_polys_px)} gt -> {out_path}")



def cmd_curves(args):
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.history) as f:
        h = json.load(f)

    epochs = range(1, len(h["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # loss
    ax1.plot(epochs, h["loss"], "b-", label="Treino")
    ax1.plot(epochs, h["val_loss"], "r-", label="Validacao")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss (BCE + Dice)")
    ax1.set_title("Loss por epoca")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IoU
    ax2.plot(epochs, h["iou_metric"], "b-", label="Treino")
    ax2.plot(epochs, h["val_iou_metric"], "r-", label="Validacao")
    best_epoch = h["val_iou_metric"].index(max(h["val_iou_metric"])) + 1
    best_iou = max(h["val_iou_metric"])
    ax2.axvline(best_epoch, color="green", linestyle="--", alpha=0.5,
                label=f"Melhor: {best_iou:.4f} (epoca {best_epoch})")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("IoU")
    ax2.set_title("IoU por epoca")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Curvas de Treinamento - U-Net ResNet34", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  curvas salvas em: {out_path}")



def cmd_grid(args):
    os.makedirs(args.output_dir, exist_ok=True)

    img_dir = os.path.join(args.data_dir, "train", "RGB-PanSharpen")
    gt_dir = os.path.join(args.data_dir, "train", "geojson", "buildings")

    # conta predios por tile
    tiles = []
    for f in sorted(os.listdir(gt_dir)):
        if not f.endswith(".geojson"):
            continue
        tid = extract_image_number(f)
        if not tid:
            continue
        try:
            gdf = gpd.read_file(os.path.join(gt_dir, f))
            n = len([g for g in gdf.geometry if g and g.is_valid and not g.is_empty])
        except Exception:
            n = 0
        tiles.append((tid, n))

    # escolhe: 3 vazios, 3 medios, 3 densos
    tiles.sort(key=lambda x: x[1])
    empty = [t for t in tiles if t[1] == 0][:3]
    dense = sorted(tiles, key=lambda x: -x[1])[:3]
    mid = len(tiles) // 2
    sparse = tiles[mid:mid + 3]
    selection = empty + sparse + dense

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    labels = (["Vazio"] * 3 + ["Medio"] * 3 + ["Denso"] * 3)

    for idx, ((tid, n_buildings), label) in enumerate(zip(selection, labels)):
        ax = axes[idx // 3][idx % 3]
        tif = os.path.join(img_dir, f"RGB-PanSharpen_AOI_3_Paris_img{tid}.tif")
        if os.path.exists(tif):
            img = load_rgb(tif)
            ax.imshow(img)
        ax.set_title(f"{label} - img{tid} ({n_buildings} predios)", fontsize=10)
        ax.axis("off")

    fig.suptitle("Diversidade do Dataset - SpaceNet 2 AOI_3 Paris",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, "dataset_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  grid salvo em: {out_path}")



def main():
    parser = argparse.ArgumentParser(
        description="gera figuras pro relatorio e video."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # overlay
    p_ov = sub.add_parser("overlay", help="predicao vs GT na imagem de satelite")
    p_ov.add_argument("--data_dir", required=True)
    p_ov.add_argument("--predictions", required=True, help="predictions.geojson")
    p_ov.add_argument("--output_dir", default="figures")
    p_ov.add_argument("--splits_json", default=None)
    p_ov.add_argument("--split", default=None)
    p_ov.add_argument("--num_tiles", type=int, default=8)
    p_ov.add_argument("--tile_ids", nargs="*", default=None,
                       help="IDs especificos de tiles pra visualizar")

    # curves
    p_cv = sub.add_parser("curves", help="curvas de loss/IoU do treino")
    p_cv.add_argument("--history", required=True, help="training_history.json")
    p_cv.add_argument("--output_dir", default="figures")

    # grid
    p_gr = sub.add_parser("grid", help="grid de amostras do dataset (vazio/medio/denso)")
    p_gr.add_argument("--data_dir", required=True)
    p_gr.add_argument("--output_dir", default="figures")

    args = parser.parse_args()

    if args.command == "overlay":
        cmd_overlay(args)
    elif args.command == "curves":
        cmd_curves(args)
    elif args.command == "grid":
        cmd_grid(args)


if __name__ == "__main__":
    main()
