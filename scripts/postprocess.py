"""pos-processa predicoes do modelo em poligonos GeoJSON georreferenciados.

carrega um modelo treinado, roda inferencia nas imagens de teste usando
sliding-window tiling (patches de 256x256), aplica limpeza morfologica
(opening depois closing), extrai contornos, simplifica poligonos, filtra
por area, e converte coordenadas de pixel pra coordenadas geograficas via
transformada afim do rasterio.

saida: predictions.geojson com um Feature por predio (EPSG:4326).

Usage:
    python postprocess.py \
        --data_dir /path/to/spacenet/test/RGB-PanSharpen \
        --model best_model.keras \
        --output predictions.geojson
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Polygon, MultiPolygon

# segmentation_models precisa do Keras 2; TF moderno (>=2.16) usa Keras 3 por padrao,
# entao redireciona tf.keras pro backend legado Keras 2. precisa setar antes do import do tf.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
try:
    import tensorflow as tf
    import segmentation_models as sm
    HAS_TF = True
except ImportError:
    tf = None
    sm = None
    HAS_TF = False

from dataset import normalize_percentile, extract_image_number


def load_model(model_path, backbone="resnet34"):
    """carrega um modelo Keras treinado com objetos customizados.

    Args:
        model_path: caminho pro arquivo .keras salvo.
        backbone: backbone usado no treino (pros objetos customizados).

    Returns:
        modelo Keras carregado.
    """
    from train import bce_dice_loss, iou_metric

    custom_objects = {
        "bce_dice_loss": bce_dice_loss,
        "iou_metric": iou_metric,
    }

    model = tf.keras.models.load_model(
        model_path, custom_objects=custom_objects
    )
    print(f"modelo carregado de: {model_path}")
    return model


def predict_tile_tiling(model, image, patch_size=256, backbone="resnet34",
                        stride=None):
    """roda inferencia num tile inteiro usando sliding-window tiling.

    divide o tile em patches de patch_size x patch_size, prediz cada um,
    e remonta a mascara completa. usa media ponderada por overlap
    onde os patches se sobrepoem.

    isso garante que treino (crop 256) e inferencia operem na MESMA
    escala espacial - sem resize do tile inteiro 650x650.

    Args:
        model: modelo Keras treinado.
        image: array da imagem de entrada, shape (H, W, 3), valores em [0, 1].
        patch_size: tamanho de cada patch (deve bater com o crop size do treino).
        backbone: nome do backbone pro preprocessamento.

    Returns:
        mascara predita, shape (H, W), valores float em [0, 1].
    """
    preprocess_fn = sm.get_preprocessing(backbone)
    h, w = image.shape[:2]

    # cria arrays de saida
    pred_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # calcula posicoes dos patches. stride padrao = patch_size (sem overlap);
    # um stride menor sobrepoe patches, removendo as emendas duras onde
    # um predio na borda do patch seria cortado ao meio.
    stride = stride or patch_size

    y_positions = list(range(0, h - patch_size + 1, stride))
    if not y_positions or y_positions[-1] + patch_size < h:
        y_positions.append(max(0, h - patch_size))

    x_positions = list(range(0, w - patch_size + 1, stride))
    if not x_positions or x_positions[-1] + patch_size < w:
        x_positions.append(max(0, w - patch_size))

    # remove duplicatas e ordena
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    # junta todos os patches pra inferencia eficiente
    patches = []
    positions = []
    for y in y_positions:
        for x in x_positions:
            patch = image[y:y + patch_size, x:x + patch_size]
            # faz padding se necessario (caso de borda)
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros(
                    (patch_size, patch_size, 3), dtype=np.float32
                )
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            patches.append(patch)
            positions.append((y, x))

    # preprocessa e prediz em batches
    batch = np.array(patches, dtype=np.float32)
    # aplica preprocessamento do backbone: [0,1] -> [0,255] -> norm do backbone
    batch_preprocessed = preprocess_fn(batch * 255.0)
    predictions = model.predict(batch_preprocessed, verbose=0)

    # remonta
    for idx, (y, x) in enumerate(positions):
        pred_patch = predictions[idx, :, :, 0]
        ph = min(patch_size, h - y)
        pw = min(patch_size, w - x)
        pred_sum[y:y + ph, x:x + pw] += pred_patch[:ph, :pw]
        count_map[y:y + ph, x:x + pw] += 1.0

    # media das regioes sobrepostas
    count_map = np.maximum(count_map, 1.0)
    pred_mask = pred_sum / count_map

    return pred_mask


def predict_resize(model, image, target_size=256, backbone="resnet34"):
    """roda inferencia redimensionando o tile inteiro pro tamanho do modelo.

    alternativa ao tiling via 'resize': achata o tile inteiro pro
    target_size, prediz uma vez, depois faz upsample da mascara de
    probabilidade de volta pro tamanho original do tile. sem emendas,
    mas predios pequenos perdem detalhe.

    Args:
        model: modelo Keras treinado.
        image: array da imagem de entrada (H, W, 3), valores em [0, 1].
        target_size: tamanho de entrada do modelo (quadrado).
        backbone: nome do backbone pro preprocessamento.

    Returns:
        mascara predita, shape (H, W), valores float em [0, 1].
    """
    preprocess_fn = sm.get_preprocessing(backbone)
    h, w = image.shape[:2]

    resized = cv2.resize(
        image, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )
    inp = preprocess_fn(np.expand_dims(resized, 0) * 255.0)
    pred = model.predict(inp, verbose=0)[0, :, :, 0]

    # faz upsample da mascara de probabilidade pro tamanho original do tile
    pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    return pred_full


def predict_full_tile(model, image, mode="tiling", patch_size=256,
                      backbone="resnet34", stride=None):
    """prediz um tile inteiro usando o modo escolhido (tiling ou resize).

    mantem a inferencia consistente com como o modelo foi treinado:
      - "tiling": sliding-window com patches de 256 (opcionalmente com overlap).
      - "resize": tile inteiro redimensionado pro patch_size.

    Args:
        model: modelo Keras treinado.
        image: array da imagem de entrada (H, W, 3), valores em [0, 1].
        mode: "tiling" ou "resize".
        patch_size: tamanho do patch / resize (deve bater com o crop size do treino).
        backbone: nome do backbone pro preprocessamento.
        stride: stride do tiling (so pro modo tiling).

    Returns:
        mascara predita, shape (H, W), valores float em [0, 1].
    """
    if mode == "resize":
        return predict_resize(
            model, image, target_size=patch_size, backbone=backbone
        )
    return predict_tile_tiling(
        model, image, patch_size=patch_size, backbone=backbone, stride=stride
    )


def apply_morphology(binary_mask, kernel_size=5):
    """aplica opening morfologico e depois closing pra limpar mascara binaria.

    opening remove ruido pequeno (erosao depois dilatacao).
    closing preenche buracos pequenos (dilatacao depois erosao).

    Args:
        binary_mask: mascara binaria uint8 (0 ou 255).
        kernel_size: tamanho do elemento estruturante morfologico.

    Returns:
        mascara binaria limpa (uint8, 0 ou 255).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    # opening: remove ruido pequeno
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # closing: preenche buracos pequenos
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def mask_to_polygons(mask, threshold=0.5, min_area=100, epsilon_factor=0.002,
                     morph_kernel=5):
    """converte mascara predita em poligonos com morfologia + simplificacao.

    pipeline:
    1. binariza no limiar
    2. opening morfologico e depois closing
    3. cv2.findContours
    4. simplificacao approxPolyDP
    5. filtra por area minima

    Args:
        mask: array da mascara predita (H, W), valores em [0, 1].
        threshold: limiar de binarizacao.
        min_area: area minima do poligono em pixels pra manter.
        epsilon_factor: fator pro approxPolyDP (relativo ao perimetro).
        morph_kernel: tamanho do kernel morfologico.

    Returns:
        lista de objetos shapely Polygon em coordenadas de pixel.
    """
    # binariza
    binary = (mask > threshold).astype(np.uint8) * 255

    # limpeza morfologica
    binary = apply_morphology(binary, kernel_size=morph_kernel)

    # encontra contornos
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue

        # simplifica contorno com approxPolyDP
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # contorno pra coordenadas de poligono
        coords = approx.reshape(-1, 2).astype(float)

        # fecha o poligono
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        if len(coords) < 4:
            continue

        try:
            poly = Polygon(coords)
            if poly.is_valid and poly.area >= min_area:
                polygons.append(poly)
            elif not poly.is_valid:
                # tenta consertar poligono invalido
                poly = poly.buffer(0)
                if isinstance(poly, MultiPolygon):
                    for p in poly.geoms:
                        if p.area >= min_area:
                            polygons.append(p)
                elif poly.area >= min_area:
                    polygons.append(poly)
        except Exception:
            continue

    return polygons


def pixel_to_geo(polygon, transform):
    """converte poligono de coordenadas de pixel pra coordenadas geograficas.

    coordenadas de pixel (col, row) sao mapeadas pra (lon, lat) via
    transformada afim do rasterio.

    Args:
        polygon: poligono shapely em coordenadas de pixel (col, row).
        transform: transformada Affine do rasterio.

    Returns:
        poligono shapely em coordenadas geograficas (lon, lat).
    """
    def transform_coords(coords):
        geo_coords = []
        for col, row in coords:
            x, y = transform * (col, row)
            geo_coords.append((x, y))
        return geo_coords

    exterior = transform_coords(polygon.exterior.coords)
    interiors = [
        transform_coords(interior.coords)
        for interior in polygon.interiors
    ]

    return Polygon(exterior, interiors)


def process_image(model, image_path, backbone="resnet34", threshold=0.5,
                  min_area=100, patch_size=256, morph_kernel=5,
                  mode="tiling", stride=None):
    """processa uma unica imagem: carrega, prediz via tiling, pos-processa.

    Args:
        model: modelo Keras treinado.
        image_path: caminho pro GeoTIFF de entrada.
        backbone: nome do backbone pro preprocessamento.
        threshold: limiar de binarizacao.
        min_area: area minima do poligono em pixels.
        patch_size: tamanho do patch pra inferencia via tiling.
        morph_kernel: tamanho do kernel morfologico.

    Returns:
        lista de poligonos shapely em coordenadas geograficas,
        e o CRS da imagem de origem.
    """
    with rasterio.open(image_path) as src:
        n_bands = min(3, src.count)
        bands = [src.read(i + 1) for i in range(n_bands)]
        while len(bands) < 3:
            bands.append(bands[-1])
        image = np.stack(bands, axis=-1)
        transform = src.transform
        crs = src.crs

    # normaliza 16-bit pra [0, 1] usando percentil 2-98
    image = normalize_percentile(image, low=2, high=98)

    # prediz o tile inteiro usando o modo escolhido (tiling ou resize)
    pred_mask = predict_full_tile(
        model, image, mode=mode, patch_size=patch_size,
        backbone=backbone, stride=stride,
    )

    # extrai poligonos com morfologia + simplificacao
    pixel_polygons = mask_to_polygons(
        pred_mask, threshold=threshold, min_area=min_area,
        morph_kernel=morph_kernel,
    )

    # converte pra coordenadas geograficas
    geo_polygons = []
    for poly in pixel_polygons:
        geo_poly = pixel_to_geo(poly, transform)
        if geo_poly.is_valid and not geo_poly.is_empty:
            geo_polygons.append(geo_poly)

    return geo_polygons, crs


def main():
    parser = argparse.ArgumentParser(
        description="pos-processa predicoes do modelo em GeoJSON. "
                    "usa sliding-window tiling (256x256) pra inferencia."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="diretorio com imagens de teste (arquivos .tif). "
             "pode ser spacenet/test/RGB-PanSharpen/ ou um dir simples.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="caminho pro arquivo do modelo treinado (.keras).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.geojson",
        help="caminho do arquivo GeoJSON de saida.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        help="arquitetura do backbone usada no treino.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="limiar de binarizacao pras predicoes.",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=100,
        help="area minima do poligono em pixels.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="tamanho do patch pra inferencia via tiling (deve bater com o crop do treino).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tiling",
        choices=["tiling", "resize"],
        help="modo de inferencia (deve bater com como o modelo foi treinado).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="stride do tiling (modo tiling). padrao: patch_size // 2 "
             "(50%% overlap) pra evitar emendas cortando predios.",
    )
    parser.add_argument(
        "--morph_kernel",
        type=int,
        default=5,
        help="tamanho do kernel morfologico pra limpeza da mascara.",
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default=None,
        help="caminho pro splits.json. com --split, so tiles daquele split "
             "sao processados (ex: o set de teste rotulado interno).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="qual split processar (requer --splits_json).",
    )
    args = parser.parse_args()

    # encontra imagens de teste
    image_dir = args.data_dir
    # tambem checa subdiretorio RGB-PanSharpen
    rgb_subdir = os.path.join(args.data_dir, "RGB-PanSharpen")
    if os.path.isdir(rgb_subdir):
        image_dir = rgb_subdir

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.tif")))

    if not image_files:
        print(f"ERRO: nenhuma imagem .tif encontrada em {image_dir}")
        return

    # restringe a um split especifico (ex: o set de teste rotulado interno).
    # isso permite rodar inferencia exatamente nos tiles de teste holdout, que
    # ficam misturados em train/RGB-PanSharpen/, pra que evaluate.py possa avaliar.
    if args.splits_json and args.split:
        with open(args.splits_json) as f:
            splits = json.load(f)
        split_ids = set(str(i) for i in splits.get(args.split, []))
        image_files = [
            f for f in image_files
            if extract_image_number(os.path.basename(f)) in split_ids
        ]
        print(f"  restrito ao split '{args.split}': {len(image_files)} tiles")
        if not image_files:
            print(
                f"ERRO: nenhum tile do split '{args.split}' encontrado em {image_dir}. "
                f"verifique se --splits_json bate com esse dataset."
            )
            return
    elif args.splits_json or args.split:
        print("AVISO: --splits_json e --split devem ser usados juntos; "
              "processando TODOS os tiles do diretorio.")

    # stride do tiling: padrao 50% overlap pra que predios na borda dos
    # patches nao sejam cortados (so usado no modo tiling).
    stride = args.stride if args.stride else args.patch_size // 2

    print(f"SpaceNet 2 AOI_3_Paris - pos-processamento")
    print("=" * 50)
    print(f"  imagens: {len(image_files)} em {image_dir}")
    print(f"  modelo: {args.model}")
    print(f"  modo: {args.mode}")
    print(f"  patch size: {args.patch_size}x{args.patch_size}")
    if args.mode == "tiling":
        print(f"  stride: {stride}")
    print(f"  limiar: {args.threshold}")
    print(f"  area minima: {args.min_area} px")
    print(f"  kernel morf: {args.morph_kernel}")
    print(f"  saida: {args.output}")

    # carrega modelo
    model = load_model(args.model, args.backbone)

    # processa todas as imagens
    all_polygons = []
    all_image_ids = []
    crs = None

    for idx, img_path in enumerate(image_files):
        basename = os.path.basename(img_path)
        print(
            f"  [{idx + 1}/{len(image_files)}] {basename}...",
            end="", flush=True,
        )

        try:
            polygons, img_crs = process_image(
                model, img_path,
                backbone=args.backbone,
                threshold=args.threshold,
                min_area=args.min_area,
                patch_size=args.patch_size,
                morph_kernel=args.morph_kernel,
                mode=args.mode,
                stride=stride,
            )
            if crs is None:
                crs = img_crs

            image_id = os.path.splitext(basename)[0]
            for poly in polygons:
                all_polygons.append(poly)
                all_image_ids.append(image_id)

            print(f" {len(polygons)} predios")

        except Exception as e:
            print(f" ERRO: {e}")

    # cria GeoDataFrame e exporta
    if not all_polygons:
        print("\nAVISO: nenhum predio detectado em nenhuma imagem.")
        gdf = gpd.GeoDataFrame(
            {"image_id": [], "geometry": []},
            crs="EPSG:4326",
        )
    else:
        gdf = gpd.GeoDataFrame(
            {"image_id": all_image_ids},
            geometry=all_polygons,
            crs=crs if crs else "EPSG:4326",
        )

        # reprojetar pra EPSG:4326 se o crs nao bater
        if gdf.crs and str(gdf.crs) != "EPSG:4326":
            try:
                gdf = gdf.to_crs("EPSG:4326")
            except Exception:
                pass

    # salva GeoJSON
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    gdf.to_file(args.output, driver="GeoJSON")

    print(f"\nresultados:")
    print(f"  total de predios: {len(all_polygons)}")
    print(f"  imagens processadas: {len(image_files)}")
    print(f"  saida: {args.output}")
    print(f"  CRS: EPSG:4326")


if __name__ == "__main__":
    main()
