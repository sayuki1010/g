import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# モデルの読み込み
model = tf.keras.models.load_model('flower_classifier_model.keras')

# 画像の前処理と予測
img_dir = 'C:\\Users\\USER\\OneDrive - 文教大学\\ドキュメント\\test\\picuture\\gohantei'
img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')]

img_arrays = []

for img_path in img_paths:
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    img_arrays.append(img_array)

img_arrays = np.vstack(img_arrays)

# 予測
predictions = model.predict(img_arrays)
predicted_classes = np.argmax(predictions, axis=1)

class_labels = {0: 'ブタナ', 1: 'キバナコスモス', 2: 'オオキンケイギク'}

# 件数を集計
prediction_counts = {label: 0 for label in class_labels.values()}

for pred_class in predicted_classes:
    prediction_counts[class_labels[pred_class]] += 1

# 結果を表示
for img_path, pred_class in zip(img_paths, predicted_classes):
    img_name = os.path.basename(img_path)
    print(f'{img_name}: これは{class_labels[pred_class]}です')

print("\n予測結果の件数:")
for label, count in prediction_counts.items():
    print(f'{label}: {count}件')
