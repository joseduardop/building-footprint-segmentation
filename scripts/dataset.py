"""
carregador de dataset spacenet 2 aoi_3_paris com pipeline tf.data.

le tiles rgb-pansharpen geotiff (650x650, 16-bit, 3 bandas) e rasteriza
as anotacoes geojson de edificacoes em mascaras binarias. aplica normalizacao
percentil 2-98, crops aleatorios 256x256 pro treino e augmentation.

os 1148 tiles rotulados sao divididos 70/15/15 em train/val/test,
com o split salvo em splits.json pra reprodutibilidade.

uso:
    python dataset.py --data_dir /caminho/spacenet --batch_size 8
"""

import argparse
import json
import os
import random
import re

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize as rasterio_rasterize
import geopandas as gpd

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    tf = None
    HAS_TF = False

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def extract_image_number(filename):
    """extrai o numero do tile de um nome de arquivo spacenet."""
    match = re.search(r'img(\d+)', filename)
    if match:
        return match.group(1)
    return None


def discover_tiles(data_dir):
    """descobre todos os tiles rotulados no dataset spacenet 2 aoi_3_paris.

    busca pares de .tif (rgb-pansharpen) e .geojson (buildings) que casam
    pelo numero do tile (img{N}).

    retorna lista de dicts com image_id, image_path, geojson_path.
    """
    image_dir = os.path.join(data_dir, "train", "RGB-PanSharpen")
    geojson_dir = os.path.join(data_dir, "train", "geojson", "buildings")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"pasta RGB-PanSharpen nao encontrada: {image_dir}"
        )
    if not os.path.isdir(geojson_dir):
        raise FileNotFoundError(
            f"pasta geojson/buildings nao encontrada: {geojson_dir}"
        )

    images = {}
    for f in os.listdir(image_dir):
        if f.endswith(".tif"):
            img_num = extract_image_number(f)
            if img_num:
                images[img_num] = os.path.join(image_dir, f)

    geojsons = {}
    for f in os.listdir(geojson_dir):
        if f.endswith(".geojson"):
            img_num = extract_image_number(f)
            if img_num:
                geojsons[img_num] = os.path.join(geojson_dir, f)

    tiles = []
    for img_id in sorted(images.keys(), key=lambda x: int(x)):
        if img_id in geojsons:
            tiles.append({
                "image_id": img_id,
                "image_path": images[img_id],
                "geojson_path": geojsons[img_id],
            })

    return tiles


def create_splits(tiles, train_ratio=0.70, val_ratio=0.15, seed=42):
    """divide os tiles em subconjuntos train/val/test."""
    rng = random.Random(seed)
    indices = list(range(len(tiles)))
    rng.shuffle(indices)

    n_train = int(len(tiles) * train_ratio)
    n_val = int(len(tiles) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": [tiles[i] for i in sorted(train_idx)],
        "val": [tiles[i] for i in sorted(val_idx)],
        "test": [tiles[i] for i in sorted(test_idx)],
    }


def save_splits_json(splits, output_path):
    """salva a divisao de splits em json pra reprodutibilidade."""
    splits_ids = {
        split_name: [t["image_id"] for t in tile_list]
        for split_name, tile_list in splits.items()
    }
    splits_ids["counts"] = {
        k: len(v) for k, v in splits_ids.items() if k != "counts"
    }
    with open(output_path, "w") as f:
        json.dump(splits_ids, f, indent=2)
    print(f"splits salvos em: {output_path}")


def load_splits_json(splits_path, tiles):
    """carrega um split salvo anteriormente do splits.json."""
    with open(splits_path, "r") as f:
        splits_ids = json.load(f)

    tile_map = {t["image_id"]: t for t in tiles}
    result = {}
    for split_name in ["train", "val", "test"]:
        ids = splits_ids.get(split_name, [])
        result[split_name] = [tile_map[i] for i in ids if i in tile_map]

    return result


def normalize_percentile(image, low=2, high=98):
    """normaliza imagem 16-bit pra [0, 1] usando clipping por percentil."""
    img = image.astype(np.float32)
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    if p_high > p_low:
        img = np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        img = np.zeros_like(img)
    return img


def rasterize_mask(image_path, geojson_path):
    """rasteriza poligonos de edificacoes em mascara binaria alinhada ao tile.

    usa o transform afim e crs do .tif de referencia pra desenhar os poligonos
    (em lat/lon) nos pixels corretos. tiles sem edificacoes produzem mascara
    toda zerada.
    """
    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception:
        gdf = gpd.GeoDataFrame()

    if gdf.empty or len(gdf) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    # reprojetar se o crs nao bater
    if gdf.crs and crs and str(gdf.crs) != str(crs):
        try:
            gdf = gdf.to_crs(crs)
        except Exception:
            pass

    valid_geoms = [
        geom for geom in gdf.geometry
        if geom is not None and geom.is_valid and not geom.is_empty
    ]
    if not valid_geoms:
        return np.zeros((height, width), dtype=np.uint8)

    shapes = [(geom, 1) for geom in valid_geoms]
    return rasterio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def load_tile(image_path, geojson_path, mask_path=None):
    """carrega um tile 650x650 (imagem + mascara binaria).

    le o geotiff rgb-pansharpen (3 bandas, 16-bit) e normaliza.
    a mascara vem do cache png se existir (rapido), senao rasteriza
    do geojson na hora (fallback lento).
    """
    with rasterio.open(image_path) as src:
        n_bands = min(3, src.count)
        bands = [src.read(i + 1) for i in range(n_bands)]
        while len(bands) < 3:
            bands.append(bands[-1])
        image = np.stack(bands, axis=-1)

    # normaliza 16-bit pra [0, 1] com percentil 2-98
    image = normalize_percentile(image, low=2, high=98)

    # mascara: usa cache png se disponivel, senao rasteriza na hora
    if mask_path and os.path.exists(mask_path):
        from PIL import Image
        cached = np.array(Image.open(mask_path))
        if cached.ndim == 3:
            cached = cached[..., 0]
        mask = (cached > 127).astype(np.float32)
    else:
        mask = rasterize_mask(image_path, geojson_path).astype(np.float32)

    return image, mask


def random_crop(image, mask, crop_size=256):
    """recorta um pedaco aleatorio do par imagem/mascara."""
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
        h, w = image.shape[:2]

    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)

    image_crop = image[top:top + crop_size, left:left + crop_size]
    mask_crop = mask[top:top + crop_size, left:left + crop_size]

    return image_crop, mask_crop


def fixed_crop(image, mask, top, left, crop_size=256):
    """recorta na posicao (top, left) fixa, pra cobertura completa do tile."""
    h, w = image.shape[:2]
    top = min(max(0, top), max(0, h - crop_size))
    left = min(max(0, left), max(0, w - crop_size))

    img = image[top:top + crop_size, left:left + crop_size]
    m = mask[top:top + crop_size, left:left + crop_size]

    if img.shape[0] < crop_size or img.shape[1] < crop_size:
        ph = crop_size - img.shape[0]
        pw = crop_size - img.shape[1]
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        m = np.pad(m, ((0, ph), (0, pw)), mode="reflect")

    return img, m


def resize_pair(image, mask, size=256):
    """redimensiona o par imagem/mascara pro tamanho (size, size).

    usado no modo resize: o tile inteiro 650x650 e encolhido pro tamanho
    de entrada do modelo. bilinear pra imagem, nearest pra mascara.
    """
    img = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, m


def patch_positions(h, w, patch, stride=None):
    """posicoes (top, left) dos patches que cobrem uma imagem (h, w).

    o ultimo patch de cada eixo e puxado pra dentro pra garantir
    que a borda e coberta.
    """
    stride = stride or patch
    ys = list(range(0, max(1, h - patch + 1), stride))
    if not ys or ys[-1] + patch < h:
        ys.append(max(0, h - patch))
    xs = list(range(0, max(1, w - patch + 1), stride))
    if not xs or xs[-1] + patch < w:
        xs.append(max(0, w - patch))
    return [(y, x) for y in sorted(set(ys)) for x in sorted(set(xs))]


def get_augmentation_pipeline():
    """cria pipeline de augmentation com albumentations."""
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
    ])


def augment_tf_fallback(image, mask):
    """augmentation nativo do tf (fallback quando nao tem albumentations)."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def create_dataset(tiles, batch_size=8, crop_size=256, shuffle=True,
                   augment=False, seed=42, mask_dir=None,
                   mode="tiling", full_coverage=False):
    """cria um tf.data.Dataset a partir de uma lista de tiles.

    modos de entrada (experimento tiling vs resize):
      - "tiling": cada tile contribui um crop 256x256 na resolucao original.
        treino usa um crop aleatorio por tile; full_coverage=True (validacao)
        emite TODOS os patches que cobrem o tile, nao so o centro.
      - "resize": o tile inteiro e redimensionado pra crop_size x crop_size
        (uma amostra por tile, igual no treino e validacao).
    """
    if not tiles:
        raise ValueError("nenhum tile fornecido pro create_dataset")

    image_paths = [t["image_path"] for t in tiles]
    geojson_paths = [t["geojson_path"] for t in tiles]
    image_ids = [t["image_id"] for t in tiles]

    # cada amostra e (tile_idx, top, left):
    #   top == -2 -> resize do tile inteiro
    #   top == -1 -> crop aleatorio (treino, modo tiling)
    #   top >= 0  -> crop fixo em (top, left) (validacao com cobertura completa)
    if mode == "resize":
        samples = [(i, -2, -2) for i in range(len(tiles))]
    elif full_coverage:
        with rasterio.open(image_paths[0]) as src:
            full_h, full_w = src.height, src.width
        positions = patch_positions(full_h, full_w, crop_size)
        samples = [
            (i, y, x) for i in range(len(tiles)) for (y, x) in positions
        ]
    else:
        samples = [(i, -1, -1) for i in range(len(tiles))]

    aug_pipeline = get_augmentation_pipeline() if augment else None

    def load_fn(s_idx):
        """carrega uma amostra (crop ou resize do tile) e aplica augmentation."""
        tile_idx, top, left = samples[int(s_idx.numpy())]
        mask_path = None
        if mask_dir is not None:
            mask_path = os.path.join(
                mask_dir, f"mask_{image_ids[tile_idx]}.png"
            )
        image, mask = load_tile(
            image_paths[tile_idx], geojson_paths[tile_idx], mask_path=mask_path
        )

        if top == -2:
            image, mask = resize_pair(image, mask, crop_size)
        elif top == -1:
            image, mask = random_crop(image, mask, crop_size)
        else:
            image, mask = fixed_crop(image, mask, top, left, crop_size)

        if augment and aug_pipeline is not None:
            augmented = aug_pipeline(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        return image, mask[..., np.newaxis]

    def tf_load_fn(idx):
        """wrapper tf pra load_fn com py_function."""
        image, mask = tf.py_function(
            load_fn,
            [idx],
            [tf.float32, tf.float32],
        )
        image.set_shape([crop_size, crop_size, 3])
        mask.set_shape([crop_size, crop_size, 1])
        return image, mask

    indices = tf.data.Dataset.range(len(samples))

    if shuffle:
        indices = indices.shuffle(
            buffer_size=len(samples), seed=seed, reshuffle_each_iteration=True
        )

    dataset = indices.map(tf_load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # fallback de augmentation nativo do tf se nao tiver albumentations
    if augment and aug_pipeline is None:
        dataset = dataset.map(
            augment_tf_fallback, num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():
    """teste do carregador de dataset via cli.""" # usado só pra debug
    parser = argparse.ArgumentParser(
        description="carregador de dataset spacenet 2 aoi_3_paris. "
                    "descobre tiles, cria splits 70/15/15 e testa o pipeline tf.data."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="diretorio raiz do dataset spacenet 2.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="tamanho do batch.")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="tamanho do crop aleatorio (quadrado).")
    parser.add_argument("--splits_json", type=str, default="splits.json",
                        help="caminho pra salvar/carregar o splits.json.")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed aleatoria pra reprodutibilidade do split.")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="pasta com mascaras png cacheadas (do create_masks.py).")
    parser.add_argument("--mode", type=str, default="tiling",
                        choices=["tiling", "resize"],
                        help="modo de entrada: tiling (crops 256) ou resize (tile inteiro).")
    args = parser.parse_args()

    print(f"spacenet 2 aoi_3_paris - carregador de dataset")
    print(f"  data dir: {args.data_dir}")
    print(f"  batch: {args.batch_size}")
    print(f"  crop: {args.crop_size}x{args.crop_size}")
    print(f"  seed: {args.seed}")

    print("\nbuscando tiles...")
    tiles = discover_tiles(args.data_dir)
    print(f"  {len(tiles)} tiles rotulados encontrados")

    if os.path.exists(args.splits_json):
        print(f"\ncarregando splits existentes de {args.splits_json}")
        splits = load_splits_json(args.splits_json, tiles)
    else:
        print(f"\ncriando splits 70/15/15 (seed={args.seed})...")
        splits = create_splits(tiles, seed=args.seed)
        save_splits_json(splits, args.splits_json)

    for split_name, split_tiles in splits.items():
        print(f"  {split_name}: {len(split_tiles)} tiles")

    for split_name in ["train", "val", "test"]:
        split_tiles = splits[split_name]
        if not split_tiles:
            print(f"\n  {split_name}: vazio")
            continue

        print(f"\n  testando split {split_name} ({len(split_tiles)} tiles)...")
        try:
            ds = create_dataset(
                split_tiles,
                batch_size=args.batch_size,
                crop_size=args.crop_size,
                shuffle=(split_name == "train"),
                augment=(split_name == "train"),
                seed=args.seed,
                mask_dir=args.mask_dir,
                mode=args.mode,
                full_coverage=(split_name != "train"),
            )
            for batch_images, batch_masks in ds.take(1):
                print(f"    shape imagens: {batch_images.shape}")
                print(f"    shape mascaras: {batch_masks.shape}")
                print(f"    range imagem: [{batch_images.numpy().min():.3f}, "
                      f"{batch_images.numpy().max():.3f}]")
                print(f"    valores unicos mascara: "
                      f"{np.unique(batch_masks.numpy())}")
        except Exception as e:
            print(f"    erro: {e}")

    print("\nteste do carregador completo.")


if __name__ == "__main__":
    main()
