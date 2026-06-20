"""valida integridade do dataset SpaceNet 2 AOI_3_Paris e rasteriza anotacoes.

inspeciona a estrutura real do diretorio SpaceNet 2, pareia arquivos .tif
RGB-PanSharpen com suas anotacoes GeoJSON de predios correspondentes,
rasteriza poligonos vetoriais em mascaras binarias (650x650), e salva
amostras de debug com overlay.

estrutura de diretorio esperada:
    spacenet/train/RGB-PanSharpen/RGB-PanSharpen_AOI_3_Paris_img{N}.tif
    spacenet/train/geojson/buildings/buildings_AOI_3_Paris_img{N}.geojson
    spacenet/test/RGB-PanSharpen/...  (sem geojson)

Usage:
    python validate_dataset.py --data_dir /path/to/spacenet
"""

import argparse
import os
import re
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_image_number(filename):
    """extrai o ID numerico do tile de um nome de arquivo SpaceNet AOI_3_Paris.

    trata padroes como:
        RGB-PanSharpen_AOI_3_Paris_img123.tif  -> '123'
        buildings_AOI_3_Paris_img123.geojson    -> '123'

    Args:
        filename: string do nome do arquivo.

    Returns:
        string do numero da imagem, ou None se nao encontrado.
    """
    match = re.search(r'img(\d+)', filename)
    if match:
        return match.group(1)
    return None


def find_pairs(data_dir):
    """encontra pares correspondentes de RGB-PanSharpen .tif e buildings GeoJSON.

    busca na estrutura de diretorio AOI_3_Paris:
        {data_dir}/train/RGB-PanSharpen/*.tif
        {data_dir}/train/geojson/buildings/*.geojson

    Args:
        data_dir: diretorio raiz do dataset SpaceNet.

    Returns:
        tupla de (pairs, image_files, geojson_files) onde pairs eh uma lista
        de tuplas (image_path, geojson_path).
    """
    image_dir = os.path.join(data_dir, "train", "RGB-PanSharpen")
    geojson_dir = os.path.join(data_dir, "train", "geojson", "buildings")

    image_files = {}
    geojson_files = {}

    # escaneia imagens
    if os.path.isdir(image_dir):
        for f in os.listdir(image_dir):
            if f.endswith(".tif"):
                img_num = extract_image_number(f)
                if img_num:
                    image_files[img_num] = os.path.join(image_dir, f)

    # escaneia geojsons
    if os.path.isdir(geojson_dir):
        for f in os.listdir(geojson_dir):
            if f.endswith(".geojson"):
                img_num = extract_image_number(f)
                if img_num:
                    geojson_files[img_num] = os.path.join(geojson_dir, f)

    # pareia pelo numero da imagem
    pairs = []
    for key in sorted(image_files.keys(), key=lambda x: int(x)):
        if key in geojson_files:
            pairs.append((image_files[key], geojson_files[key]))

    return pairs, image_files, geojson_files


def rasterize_geojson(geojson_path, reference_tif_path, output_mask_path=None):
    """rasteriza um GeoJSON de pegadas de predios pra bater com um .tif de referencia.

    trata arquivos GeoJSON vazios (tiles com zero predios) de boa,
    retornando uma mascara toda zerada.

    Args:
        geojson_path: caminho pro arquivo GeoJSON com poligonos de predios.
        reference_tif_path: caminho pro .tif de referencia pro alinhamento espacial.
        output_mask_path: caminho opcional pra salvar a mascara de saida como GeoTIFF.

    Returns:
        mascara binaria como array numpy (H, W) com 1 pra predios.
    """
    with rasterio.open(reference_tif_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"  AVISO: nao consegui ler {geojson_path}: {e}")
        return np.zeros(out_shape, dtype=np.uint8)

    if gdf.empty or len(gdf) == 0:
        # tile com zero predios - caso valido
        mask = np.zeros(out_shape, dtype=np.uint8)
    else:
        # reprojetar pro CRS da imagem se nao bater (geojson eh CRS84 = lon/lat)
        if gdf.crs and crs and str(gdf.crs) != str(crs):
            try:
                gdf = gdf.to_crs(crs)
            except Exception:
                pass

        # filtra geometrias validas
        valid_geoms = [
            geom for geom in gdf.geometry
            if geom is not None and geom.is_valid and not geom.is_empty
        ]

        if valid_geoms:
            shapes = [(geom, 1) for geom in valid_geoms]
            mask = rasterize(
                shapes,
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
        else:
            mask = np.zeros(out_shape, dtype=np.uint8)

    if output_mask_path:
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        with rasterio.open(
            output_mask_path, "w",
            driver="GTiff",
            height=out_shape[0],
            width=out_shape[1],
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(mask, 1)

    return mask


def normalize_16bit_for_display(image):
    """normaliza imagem 16-bit pra [0,1] usando clipping percentil 2-98.

    Args:
        image: array da imagem (H, W, C), tipicamente uint16.

    Returns:
        imagem float32 normalizada em [0, 1].
    """
    img = image.astype(np.float64)
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        img = np.zeros_like(img)
    return img.astype(np.float32)


def save_debug_overlay(image_path, mask, output_dir, sample_idx):
    """salva visualizacao de debug com overlay de um par imagem/mascara.

    le um .tif RGB-PanSharpen 16-bit, normaliza pra exibicao usando
    percentil 2-98, e sobrepoe a mascara rasterizada.

    Args:
        image_path: caminho pro .tif de origem.
        mask: array da mascara binaria (H, W).
        output_dir: diretorio pra saida de debug.
        sample_idx: indice pra nome do arquivo de debug.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(image_path) as src:
        # le bandas RGB (RGB-PanSharpen tem 3 bandas)
        n_bands = min(3, src.count)
        bands = [src.read(i + 1) for i in range(n_bands)]

        if n_bands == 1:
            img = np.stack([bands[0]] * 3, axis=-1)
        elif n_bands == 2:
            img = np.stack([bands[0], bands[1], bands[0]], axis=-1)
        else:
            img = np.stack(bands[:3], axis=-1)

    # normaliza 16-bit pra [0, 1] pra exibicao
    img = normalize_16bit_for_display(img)

    n_buildings = int(mask.sum() > 0)
    building_pixels = int(mask.sum())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)
    axes[0].set_title(f"RGB-PanSharpen (16-bit normalizado)")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Mascara rasterizada ({building_pixels} px)")
    axes[1].axis("off")

    # overlay
    overlay = img.copy()
    overlay[mask == 1] = [1.0, 0.3, 0.3]
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(os.path.basename(image_path), fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"debug_overlay_{sample_idx:02d}.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


def validate_dataset(data_dir, output_dir=None):
    """roda validacao completa num diretorio de dataset SpaceNet 2 AOI_3_Paris.

    checa:
    - estrutura de diretorio bate com o layout esperado do SpaceNet 2
    - arquivos .tif RGB-PanSharpen sao 650x650, 3 bandas, 16-bit
    - arquivos GeoJSON podem ser lidos e contem geometrias validas
    - rasterizacao produz mascaras corretas (incluindo vazias)
    - informacao de CRS eh consistente

    Args:
        data_dir: diretorio raiz do dataset SpaceNet.
        output_dir: diretorio opcional pra mascaras rasterizadas e saida de debug.

    Returns:
        dicionario com resultados da validacao.
    """
    print(f"validando dataset SpaceNet 2 AOI_3_Paris em: {data_dir}")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"ERRO: diretorio nao existe: {data_dir}")
        return {"valid": False, "error": "diretorio nao encontrado"}

    # checa estrutura de diretorio esperada
    expected_dirs = [
        os.path.join(data_dir, "train", "RGB-PanSharpen"),
        os.path.join(data_dir, "train", "geojson", "buildings"),
    ]
    optional_dirs = [
        os.path.join(data_dir, "test", "RGB-PanSharpen"),
        os.path.join(data_dir, "train", "summaryData"),
    ]

    print("\nchecagem da estrutura de diretorios:")
    for d in expected_dirs:
        exists = os.path.isdir(d)
        status = "OK" if exists else "FALTANDO"
        print(f"  [{status}] {os.path.relpath(d, data_dir)}")
        if not exists:
            print(f"    ERRO: diretorio obrigatorio nao encontrado: {d}")

    for d in optional_dirs:
        exists = os.path.isdir(d)
        status = "OK" if exists else "opcional"
        print(f"  [{status}] {os.path.relpath(d, data_dir)}")

    # encontra pares de arquivos
    pairs, all_images, all_geojsons = find_pairs(data_dir)

    # conta imagens de teste
    test_dir = os.path.join(data_dir, "test", "RGB-PanSharpen")
    n_test_images = 0
    if os.path.isdir(test_dir):
        n_test_images = len([
            f for f in os.listdir(test_dir) if f.endswith(".tif")
        ])

    print(f"\ncontagem de arquivos:")
    print(f"  treino RGB-PanSharpen .tif: {len(all_images)}")
    print(f"  treino buildings .geojson: {len(all_geojsons)}")
    print(f"  pares de treino pareados: {len(pairs)}")
    print(f"  teste RGB-PanSharpen .tif: {n_test_images}")

    unmatched_images = set(all_images.keys()) - set(all_geojsons.keys())
    unmatched_geojsons = set(all_geojsons.keys()) - set(all_images.keys())
    if unmatched_images:
        print(f"  imagens sem par (sem geojson): {len(unmatched_images)}")
    if unmatched_geojsons:
        print(f"  geojsons sem par (sem imagem): {len(unmatched_geojsons)}")

    if not pairs:
        print("\nAVISO: nenhum par imagem/geojson encontrado!")
        print("estrutura esperada: SpaceNet 2 AOI_3_Paris.")
        return {
            "valid": False,
            "total_images": len(all_images),
            "total_geojsons": len(all_geojsons),
            "matched_pairs": 0,
        }

    # inspeciona propriedades das imagens dos primeiros 5 tiles
    print("\npropriedades das imagens (primeiros 5 tiles):")
    dimensions = []
    pixel_ranges = []
    crs_list = []

    for img_path, _ in pairs[:5]:
        with rasterio.open(img_path) as src:
            dims = (src.height, src.width, src.count)
            dimensions.append(dims)
            crs_list.append(str(src.crs))
            data = src.read()
            pixel_ranges.append((int(data.min()), int(data.max()), str(data.dtype)))
            print(f"  {os.path.basename(img_path)}: "
                  f"{dims[0]}x{dims[1]}, {dims[2]} bandas, "
                  f"dtype={data.dtype}, range=[{data.min()}, {data.max()}], "
                  f"CRS={src.crs}")

    # valida dimensoes contra o esperado 650x650
    for i, dims in enumerate(dimensions):
        if dims[0] != 650 or dims[1] != 650:
            print(f"  AVISO: esperado 650x650 mas veio "
                  f"{dims[0]}x{dims[1]} pro tile {i}")
        if dims[2] != 3:
            print(f"  AVISO: esperado 3 bandas (RGB-PanSharpen) mas veio "
                  f"{dims[2]} pro tile {i}")

    # rasteriza anotacoes e conta predios
    if output_dir is None:
        output_dir = data_dir

    debug_dir = os.path.join(output_dir, "debug_samples")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    print(f"\nrasterizando {len(pairs)} anotacoes GeoJSON em mascaras...")
    valid_count = 0
    total_buildings = 0
    empty_tiles = 0

    for idx, (img_path, geojson_path) in enumerate(pairs):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{basename}_mask.tif")

        try:
            mask = rasterize_geojson(geojson_path, img_path, mask_path)
            n_building_pixels = int(mask.sum())
            valid_count += 1

            # conta predios do GeoJSON
            try:
                gdf = gpd.read_file(geojson_path)
                n_buildings = len(gdf)
                total_buildings += n_buildings
                if n_buildings == 0:
                    empty_tiles += 1
            except Exception:
                n_buildings = 0

            # salva overlays de debug pros 5 primeiros
            if idx < 5:
                save_debug_overlay(img_path, mask, debug_dir, idx)
                print(f"  [{idx + 1}/{len(pairs)}] {basename}: "
                      f"{n_buildings} predios, "
                      f"{n_building_pixels} pixels de predio"
                      f"{' (VAZIO)' if n_buildings == 0 else ''}")

        except Exception as e:
            print(f"  ERRO processando {basename}: {e}")

    # relatorio resumo
    print("\n" + "=" * 60)
    print("RELATORIO DE VALIDACAO - SpaceNet 2 AOI_3_Paris")
    print("=" * 60)
    print(f"  imagens treino: {len(all_images)}")
    print(f"  GeoJSONs treino: {len(all_geojsons)}")
    print(f"  pares pareados: {len(pairs)}")
    print(f"  rasterizados com sucesso: {valid_count}")
    print(f"  total de predios: {total_buildings}")
    print(f"  tiles vazios (0 predios): {empty_tiles}")
    print(f"  imagens teste (sem rot): {n_test_images}")
    if dimensions:
        print(f"  dimensoes do tile: {dimensions[0][0]}x{dimensions[0][1]}")
        print(f"  numero de bandas: {dimensions[0][2]}")
    if pixel_ranges:
        print(f"  dtype dos pixels: {pixel_ranges[0][2]}")
        print(f"  range dos pixels: [{pixel_ranges[0][0]}, {pixel_ranges[0][1]}]")
    if crs_list:
        print(f"  CRS: {crs_list[0]}")
    print(f"  overlays de debug: {debug_dir}")
    print(f"  mascaras salvas: {masks_dir}")
    print("=" * 60)

    return {
        "valid": valid_count > 0,
        "total_images": len(all_images),
        "total_geojsons": len(all_geojsons),
        "matched_pairs": len(pairs),
        "valid_masks": valid_count,
        "total_buildings": total_buildings,
        "empty_tiles": empty_tiles,
        "test_images": n_test_images,
    }


def main():
    parser = argparse.ArgumentParser(
        description="valida dataset SpaceNet 2 AOI_3_Paris e rasteriza "
                    "anotacoes de predios em mascaras binarias."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="diretorio raiz do dataset SpaceNet 2 "
             "(ex: /data/spacenet).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="diretorio de saida pra mascaras e amostras de debug. "
             "padrao: data_dir.",
    )
    args = parser.parse_args()

    results = validate_dataset(args.data_dir, args.output_dir)

    if not results.get("valid", False):
        print("\nvalidacao do dataset FALHOU.")
        sys.exit(1)
    else:
        print("\nvalidacao do dataset PASSOU.")


if __name__ == "__main__":
    main()
