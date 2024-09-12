import cv2
import numpy as np

# OpenCVのバージョンを表示
print(cv2.__version__)

def Judge_Matching(num):
    return num > 0.99

# 画像を読み込み
ookinnkeigiku_image = cv2.imread('./picture/ookinkeigiku/03.jpg')
ookinnkeigiku_template = cv2.imread('./picture/ookinkeigiku/03.jpg')

# 画像をグレースケールに変換
ookinnkeigiku_image_gray = cv2.cvtColor(ookinnkeigiku_image, cv2.COLOR_BGR2GRAY)

# Cannyエッジ検出器を用いてエッジを検出
edges = cv2.Canny(ookinnkeigiku_image_gray, 100, 200)

# Harrisコーナー検出器を用いてコーナーを検出
corners = cv2.cornerHarris(ookinnkeigiku_image_gray, 2, 3, 0.04)

# コーナーの結果を正規化して視覚化
dst = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX)
dst = np.uint8(dst)

# コーナーを強調するために膨張処理を行う
dst_dilated = cv2.dilate(dst, None)

# 結果を3チャンネルに変換して表示用に準備
dst_color = cv2.cvtColor(dst_dilated, cv2.COLOR_GRAY2BGR)
ookinnkeigiku_image_with_corners = ookinnkeigiku_image.copy()

# コーナーの位置を強調
threshold = 0.01表示
cv2.imshow('Edges', edges)
cv2.imshow('Corners', dst_color)
cv2.imshow('Image with Corners', ookinnkeigiku_image_with_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OpenCVで画像部分一致を検索
result = cv2.matchTemplate(ookinnkeigiku_image, ookinnkeigiku_template, cv2.TM_CCORR_NORMED)

# 最も類似度が高い位置と低い位置を取得
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

# 類似度が閾値を超えているか判定（上で作った関数を使用）
Judg = Judge_Matching(maxVal)

# 結果を出力
print(Judg)
# （実行結果→）True or False
 * dst_dilated.max()
ookinnkeigiku_image_with_corners[dst_dilated > threshold] = [0, 0, 255]

# 特徴抽出結果を