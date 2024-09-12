import cv2
import numpy as np
from sklearn.cluster import KMeans

def main():
    # 画像を読み込む
    image_path = '1.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # BGRからHSVに変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 黄色の範囲を定義 (HSV空間)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # マスクを作成
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # 黄色の部分を抽出
    yellow_parts = cv2.bitwise_and(image, image, mask=mask)

    # 黄色のピクセルのカラーコードを抽出
    yellow_pixels = image[mask != 0]

    if yellow_pixels.size == 0:
        print("No yellow pixels detected.")
    else:
        # K-meansクラスタリングを使用して色をグループ化
        num_clusters = 5  # クラスタの数（代表色の数）
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(yellow_pixels)

        # クラスタの中心を取得
        cluster_centers = kmeans.cluster_centers_
        print("Detected yellow color codes (BGR):")
        for color in cluster_centers:
            print(np.round(color).astype(int))  # カラーコードを整数に丸めて表示

    # 結果を表示
    cv2.imshow('Original Image', image)
    cv2.imshow('Yellow Parts', yellow_parts)

    # キーが押されるのを待ってからウィンドウを閉じる
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
