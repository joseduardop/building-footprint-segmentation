"""pre-rasteriza mascaras de predios em disco pra que o treino nao
re-rasterize a cada epoca.

cada GeoJSON de tile rotulado eh rasterizado uma vez numa mascara binaria
alinhada ao grid de pixels do tile, e salvo como PNG (0/255). o treino
entao carrega essas mascaras em cache em vez de ler + rasterizar o GeoJSON
a cada epoca, o que eh bem mais rapido.

as mascaras sao um cache derivado - podem ser regeneradas a partir do dataset
a qualquer momento, entao NAO precisam ser commitadas no git.

Usage:
    python create_masks.py --data_dir /path/to/spacenet
    # mascaras salvas em /path/to/spacenet/train/masks/mask_<id>.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
from PIL import Image

from dataset import discover_tiles, rasterize_mask


def main():
    parser = argparse.ArgumentParser(
        description="pre-rasteriza mascaras de predios em disco (cache unico "
                    "pra treino rapido)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="diretorio raiz do dataset SpaceNet 2.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="onde salvar os PNGs de mascara. padrao: <data_dir>/train/masks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recria mascaras que ja existem (padrao: pula elas).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        args.data_dir, "train", "masks"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("SpaceNet 2 AOI_3_Paris - pre-rasterizando mascaras")
    print("=" * 50)
    print(f"    data dir: {args.data_dir}")
    print(f"    output dir: {output_dir}")

    tiles = discover_tiles(args.data_dir)
    print(f"    tiles: {len(tiles)}")

    created = 0
    skipped = 0
    empty = 0
    total_building_px = 0

    for i, tile in enumerate(tiles):
        out_path = os.path.join(output_dir, f"mask_{tile['image_id']}.png")

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        mask = rasterize_mask(tile["image_path"], tile["geojson_path"])
        n_px = int(mask.sum())
        total_building_px += n_px
        if n_px == 0:
            empty += 1

        # salva como PNG 0/255 (compacto + visualizavel pra sanity checks)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
        created += 1

        if (i + 1) % 100 == 0:
            print(f"    [{i + 1}/{len(tiles)}] processados...")

    print("\n" + "=" * 50)
    print("RESUMO")
    print("=" * 50)
    print(f"    mascaras criadas: {created}")
    print(f"    puladas (ja existiam): {skipped}")
    print(f"    mascaras vazias (s/ pr): {empty}")
    print(f"    total pixels de predio: {total_building_px}")
    print(f"    salvas em: {output_dir}")


if __name__ == "__main__":
    main()
