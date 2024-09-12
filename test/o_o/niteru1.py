import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# データのディレクトリパス
train_dir = 'C:\\Users\\USER\\OneDrive - 文教大学\\ドキュメント\\test\\picture2\\train'
validation_dir = 'C:\\Users\\USER\\OneDrive - 文教大学\\ドキュメント\\test\\picture2\\validation'

# データジェネレータの設定
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

# 訓練データの生成
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical'
)

# 検証データの生成
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical'
)

# モデルの構築
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデルの訓練
epochs = 50
steps_per_epoch = np.ceil(train_generator.samples / train_generator.batch_size)
validation_steps = np.ceil(validation_generator.samples / validation_generator.batch_size)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # 訓練データを使用してモデルを訓練
    train_generator.reset()  # データジェネレータをリセット
    history = model.fit(
        train_generator,
        steps_per_epoch=int(steps_per_epoch),
        epochs=1,
        validation_data=validation_generator,
        validation_steps=int(validation_steps),
        callbacks=[early_stopping]
    )

# モデルの保存
model.save('flower_classifier_model.keras')
