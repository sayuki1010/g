import cv2
import os

# 動画ファイルのパス
video_path = 'IMG_7236.mp4'

# 出力する画像のディレクトリ
output_dir = 'output_images'

# ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 動画を読み込み
cap = cv2.VideoCapture(video_path)

# フレームカウントの初期化
frame_count = 0

# 動画が正常にオープンされたか確認
if not cap.isOpened():
    print(f"動画ファイル '{video_path}' を開くことができませんでした。")
else:
    # 動画のフレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"動画の総フレーム数: {total_frames}")

    while cap.isOpened():
        # フレームを1つずつ読み込む
        ret, frame = cap.read()

        # フレームが正常に読み込まれた場合
        if ret:
            # 出力ファイル名の作成 (例: frame000.jpg, frame001.jpg,...)
            output_filename = os.path.join(output_dir, f'frame{frame_count:03d}.jpg')

            # 画像を保存
            cv2.imwrite(output_filename, frame)

            # フレームカウントをインクリメント
            frame_count += 1

            # 進捗を表示
            if frame_count % 100 == 0:
                print(f"{frame_count}/{total_frames} フレームを処理中...")
        else:
            break

    # 動画のキャプチャを解放
    cap.release()
