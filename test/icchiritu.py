import cv2
import numpy as np

def calculate_image_similarity(image1_path, image2_path):
    # 画像を読み込む
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 画像が正しく読み込まれたかチェック
    if image1 is None:
        raise ValueError(f"画像ファイルが見つかりません: {image1_path}")
    if image2 is None:
        raise ValueError(f"画像ファイルが見つかりません: {image2_path}")

    # 画像のサイズを揃える
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # 画像をグレースケールに変換
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 画像のヒストグラムを計算
    hist_image1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
    hist_image2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

    # ヒストグラムを正規化
    cv2.normalize(hist_image1, hist_image1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_image2, hist_image2, 0, 1, cv2.NORM_MINMAX)

    # ヒストグラムの類似度を計算
    similarity = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_CORREL)

    return similarity

# 画像のパスを指定
image1_path = '2.jpg'
image2_path = '1.jpg'

# 一致率を計算
similarity = calculate_image_similarity(image1_path, image2_path)
print(f'画像の一致率: {similarity * 100:.2f}%')

