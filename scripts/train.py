"""treina u-net com backbone resnet34 pra segmentacao de predios.

usa a lib segmentation_models pra arquitetura do modelo com backbone
pre-treinado no imagenet. treinamento usa loss combinada BCE + Dice
com otimizador Adam e callbacks padrao (EarlyStopping,
ModelCheckpoint, ReduceLROnPlateau).

o dataset eh SpaceNet 2 AOI_3_Paris (RGB-PanSharpen, 650x650, 16-bit).
tiles sao divididos 70/15/15 e cortados pra 256x256 pro treino.

Usage:
    python train.py --data_dir /path/to/spacenet --epochs 50
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

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

from dataset import discover_tiles, create_splits, save_splits_json
from dataset import load_splits_json, create_dataset


def dice_loss(y_true, y_pred, smooth=1.0):
    """dice loss pra segmentacao binaria.

    Args:
        y_true: mascaras binarias ground truth.
        y_pred: mascaras preditas (saida sigmoid).
        smooth: fator de suavizacao pra evitar divisao por zero.

    Returns:
        valor do dice loss (1 - coeficiente dice).
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def bce_dice_loss(y_true, y_pred):
    """loss combinada BCE + Dice com pesos iguais (0.5 / 0.5).

    Args:
        y_true: mascaras binarias ground truth.
        y_pred: mascaras preditas (saida sigmoid).

    Returns:
        valor da loss combinada.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.keras.backend.mean(bce)
    dl = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dl


def iou_metric(y_true, y_pred, threshold=0.5):
    """metrica intersection over union pra segmentacao binaria.

    Args:
        y_true: mascaras binarias ground truth.
        y_pred: mascaras preditas.
        threshold: limiar de binarizacao pras predicoes.

    Returns:
        score IoU.
    """
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.keras.backend.sum(y_true * y_pred_bin)
    union = (
        tf.keras.backend.sum(y_true)
        + tf.keras.backend.sum(y_pred_bin)
        - intersection
    )
    return (intersection + 1e-7) / (union + 1e-7)


def build_model(input_shape=(256, 256, 3), backbone="resnet34",
                encoder_weights="imagenet"):
    """constroi modelo u-net com backbone (opcionalmente pre-treinado).

    Args:
        input_shape: shape do tensor de entrada (H, W, C).
        backbone: nome da arquitetura do backbone.
        encoder_weights: "imagenet" pra transfer learning, ou None pra treinar
            o encoder do zero (sem download de pesos - util pra smoke
            tests rapidos).

    Returns:
        modelo Keras compilado.
    """
    model = sm.Unet(
        backbone_name=backbone,
        input_shape=input_shape,
        classes=1,
        activation="sigmoid",
        encoder_weights=encoder_weights,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=bce_dice_loss,
        metrics=[iou_metric],
    )

    return model


def preprocess_batch(preprocess_fn):
    """cria funcao de preprocessamento pra tf.data.Dataset.

    aplica o preprocessamento especifico do backbone via segmentation_models
    na imagem, mantendo a mascara inalterada.

    o dataset fornece imagens ja normalizadas pra [0, 1] via
    percentil 2-98. a gente escala de volta pra [0, 255] pro
    preprocessador do backbone que espera esse range.

    Args:
        preprocess_fn: funcao de preprocessamento do segmentation_models.

    Returns:
        funcao que preprocessa pares (imagem, mascara).
    """
    def _preprocess(image, mask):
        # sm.get_preprocessing espera input no range [0, 255]
        image = image * 255.0
        image = preprocess_fn(image)
        return image, mask
    return _preprocess


def main():
    parser = argparse.ArgumentParser(
        description="treina u-net pra segmentacao de predios "
                    "no SpaceNet 2 AOI_3_Paris."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="diretorio raiz do dataset SpaceNet 2.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="numero maximo de epocas de treino.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="tamanho do batch de treino.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="tamanho do crop de entrada (quadrado).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tiling",
        choices=["tiling", "resize"],
        help="modo de entrada: 'tiling' (crops aleatorios de 256, cobertura "
             "total na validacao) ou 'resize' (tile inteiro redimensionado pra 256).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        help="arquitetura do backbone do encoder.",
    )
    parser.add_argument(
        "--encoder_weights",
        type=str,
        default="imagenet",
        choices=["imagenet", "none"],
        help="'imagenet' pra transfer learning, ou 'none' pra pular o download "
             "de pesos e treinar do zero (smoke tests rapidos).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate inicial.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="diretorio pra salvar modelo e logs de treino.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="paciencia do early stopping (epocas).",
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default="splits.json",
        help="caminho pra salvar/carregar splits.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed aleatoria pra reprodutibilidade.",
    )
    args = parser.parse_args()

    if not HAS_TF:
        print("ERRO: TensorFlow eh necessario pro treino.")
        print("instale com: pip install tensorflow segmentation-models")
        import sys
        sys.exit(1)

    print("SpaceNet 2 AOI_3_Paris - treino U-Net")
    print("=" * 50)
    print(f"  data dir: {args.data_dir}")
    print(f"  backbone: {args.backbone}")
    print(f"  modo: {args.mode}")
    print(f"  crop size: {args.crop_size}x{args.crop_size}")
    print(f"  batch size: {args.batch_size}")
    print(f"  epocas: {args.epochs}")
    print(f"  learning rate: {args.lr}")
    print(f"  paciencia: {args.patience}")
    print(f"  output dir: {args.output_dir}")
    print(f"  splits file: {args.splits_json}")

    # descobre tiles e cria/carrega splits
    print("\nbuscando tiles...")
    tiles = discover_tiles(args.data_dir)
    print(f"  encontrados {len(tiles)} tiles rotulados")

    if os.path.exists(args.splits_json):
        print(f"  carregando splits existentes de {args.splits_json}")
        splits = load_splits_json(args.splits_json, tiles)
    else:
        print(f"  criando splits 70/15/15 (seed={args.seed})...")
        splits = create_splits(tiles, seed=args.seed)
        os.makedirs(args.output_dir, exist_ok=True)
        save_splits_json(
            splits,
            os.path.join(args.output_dir, os.path.basename(args.splits_json)),
        )

    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    print(f"  treino: {n_train}, val: {n_val}, teste: {n_test}")

    # usa mascaras em cache se create_masks.py foi rodado (rapido); senao o
    # loader rasteriza on-the-fly a cada epoca (correto mas lento).
    mask_dir = os.path.join(args.data_dir, "train", "masks")
    if os.path.isdir(mask_dir):
        print(f"  usando mascaras em cache de {mask_dir}")
    else:
        mask_dir = None
        print("  cache de mascaras nao encontrado - rasterizando on-the-fly (lento). "
              "rode create_masks.py antes pra acelerar o treino.")

    # cria datasets tf.data.
    # treino: uma amostra por tile (crop aleatorio, ou tile inteiro redimensionado).
    # validacao: cobertura TOTAL de cada tile (todos os patches no modo tiling,
    # ou o tile inteiro redimensionado) pra que o melhor modelo seja escolhido
    # pela performance no tile inteiro, nao so no crop central.
    print("\ncriando datasets...")
    train_ds = create_dataset(
        splits["train"],
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        shuffle=True,
        augment=True,
        seed=args.seed,
        mask_dir=mask_dir,
        mode=args.mode,
        full_coverage=False,
    )
    val_ds = create_dataset(
        splits["val"],
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        shuffle=False,
        augment=False,
        seed=args.seed,
        mask_dir=mask_dir,
        mode=args.mode,
        full_coverage=True,
    )

    # aplica preprocessamento especifico do backbone
    preprocess_fn = sm.get_preprocessing(args.backbone)
    preprocess_map = preprocess_batch(preprocess_fn)

    train_ds = train_ds.map(preprocess_map, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_map, num_parallel_calls=tf.data.AUTOTUNE)

    # constroi modelo
    print("\nconstruindo modelo...")
    enc_weights = None if args.encoder_weights == "none" else args.encoder_weights
    model = build_model(
        input_shape=(args.crop_size, args.crop_size, 3),
        backbone=args.backbone,
        encoder_weights=enc_weights,
    )

    # atualiza learning rate do otimizador se nao for o padrao
    model.optimizer.learning_rate.assign(args.lr)
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "best_model.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_iou_metric",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_iou_metric",
            mode="max",
            patience=args.patience,
            verbose=1,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # treina. os datasets sao finitos (uma passada = uma epoca), entao deixamos
    # o Keras inferir a quantidade de steps. isso importa porque a validacao com
    # cobertura total emite multiplos patches por tile, entao um validation_steps
    # fixo so avaliaria uma fracao do val set.
    print("\niniciando treino...")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # salva historico de treino
    history_path = os.path.join(args.output_dir, "training_history.json")
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]

    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"\ntreino finalizado!")
    print(f"  melhor modelo salvo: {model_path}")
    print(f"  historico de treino: {history_path}")

    # mostra metricas finais
    if "val_iou_metric" in history.history:
        best_iou = max(history.history["val_iou_metric"])
        best_epoch = history.history["val_iou_metric"].index(best_iou) + 1
        print(f"  melhor val IoU: {best_iou:.4f} (epoca {best_epoch})")

    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    print(f"  loss treino final: {final_loss:.4f}")
    print(f"  loss val final: {final_val_loss:.4f}")


if __name__ == "__main__":
    main()
