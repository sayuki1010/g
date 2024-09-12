import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# モデルの読み込み
model = tf.keras.models.load_model('flower_classifier_model.keras')

# 画像の前処理と予測
img_path = 'C:\\Users\\USER\\OneDrive - 文教大学\\ドキュメント\\test\\test.jpg'

img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# 予測
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)

class_labels = {0: 'ブタナ', 1: 'キバナコスモス', 2: 'オオキンケイギク'}
print(f'これは{class_labels[predicted_class[0]]}です')