"""gera previews PNG 8-bit de todos os tiles GeoTIFF 16-bit.

os arquivos .tif crus aparecem pretos em visualizadores de imagem normais
porque sao 16-bit com valores concentrados num range estreito. esse script
aplica normalizacao percentil 2-98 e salva PNGs 8-bit visiveis.

Usage:
    python generate_previews.py --data_dir /path/to/spacenet
    # cria spacenet/train/previews/preview_img485.png etc.
"""

import argparse
import os
import re

import numpy as np
from PIL import Image


def normalize_percentile(image, low=2, high=98):
    img = image.astype(np.float32)
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    if p_high > p_low:
        img = np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        img = np.zeros_like(img)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="gera previews PNG 8-bit dos tiles GeoTIFF 16-bit."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="diretorio raiz do dataset SpaceNet.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="onde salvar os previews. padrao: <data_dir>/train/previews.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recria previews que ja existem.",
    )
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, "train", "RGB-PanSharpen")
    output_dir = args.output_dir or os.path.join(args.data_dir, "train", "previews")
    os.makedirs(output_dir, exist_ok=True)

    tifs = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")])
    print(f"gerando previews de {len(tifs)} tiles -> {output_dir}")

    try:
        import tifffile
        use_tifffile = True
    except ImportError:
        import rasterio
        use_tifffile = False

    created = 0
    skipped = 0

    for i, f in enumerate(tifs):
        num = re.search(r'img(\d+)', f)
        name = f"preview_img{num.group(1)}.png" if num else f.replace(".tif", ".png")
        out_path = os.path.join(output_dir, name)

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        if use_tifffile:
            img = tifffile.imread(os.path.join(img_dir, f))
        else:
            with rasterio.open(os.path.join(img_dir, f)) as src:
                bands = [src.read(j + 1) for j in range(min(3, src.count))]
                while len(bands) < 3:
                    bands.append(bands[-1])
                img = np.stack(bands, axis=-1)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        img = normalize_percentile(img)
        img_u8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_u8).save(out_path)
        created += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(tifs)}] ...")

    print(f"\npronto: {created} criados, {skipped} pulados (ja existiam)")
    print(f"salvos em: {output_dir}")


if __name__ == "__main__":
    main()
