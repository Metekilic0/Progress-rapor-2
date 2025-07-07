import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATASET_PATH = "C:/Users/mkasl/Desktop/donem5/yap470/Ara_rapor_ 1/dataset1_clean"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
#%70 %15 %15 dağılım
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

tf.random.set_seed(42) #aynı sonucu vermesi için bunu gerçi dosyalara test train valid olarak ayırıyoruz
#tekrarlanabilirliği sağlıyor
full_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',           
    label_mode='categorical', #Etiketler one-hot encoding formatına çevrilir çünkü bird drone vs. saçma 
    image_size=IMAGE_SIZE,  #ortak boyut tutmak için
    batch_size=BATCH_SIZE,       
    shuffle=True,                
    seed=42
)
class_names = full_dataset.class_names #Görselleştirmede kullanıyorum 
#Toplam batch sayısını bulma
dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
#Oranlara göre batch sayısı hesaplama
val_size = int(dataset_size * VALIDATION_SPLIT_RATIO)
test_size = int(dataset_size * TEST_SPLIT_RATIO)
train_size = dataset_size - val_size - test_size
#Önce test setini ayır sonra valid sonra train 
test_dataset = full_dataset.take(test_size)
validation_dataset = full_dataset.skip(test_size).take(val_size)
train_dataset = full_dataset.skip(test_size + val_size)
#data augmentation yeni veriler üretip yapay olarak overfittingi önlemeye çalışıyorum
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")

#Burası cache gibi önbellekleme vs. kullanıp performansı arttırmaya çalıştığım kısım ama buna rağmen daha farklı şeylere
#ihtiyacım var prefecthi sonraki batchi hazırlar cache önbelleğe alır
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential(name="V12 model")
model.add(tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
#1. blok
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1"))
model.add(tf.keras.layers.BatchNormalization(name="bn1"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), name="pool1"))
#2. blok
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2"))
model.add(tf.keras.layers.BatchNormalization(name="bn2"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), name="pool2"))
#3. blok
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3"))
model.add(tf.keras.layers.BatchNormalization(name="bn3"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), name="pool3"))
#GlobalAveragePooling2D ile sınıflandırma eskiden burasında flatten ve dense katmanı vardı
model.add(tf.keras.layers.GlobalAveragePooling2D(name="global_pool"))
model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
model.add(tf.keras.layers.Dense(2, activation='softmax', name="output"))
model.summary()

#default olan adam optimizer kullanıyorum
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
#Bu modelimde değiştirdiğim kısım
#Veri setindeki dosya sayılarını buluyorum ardından sınıfları sayıya göre ağırlıklandırıyorum
path_bird = os.path.join(DATASET_PATH, 'bird') #sayma
path_drone = os.path.join(DATASET_PATH, 'drone') #sayma
num_birds = len(os.listdir(path_bird))
num_drones = len(os.listdir(path_drone))
total_samples = num_birds + num_drones
#Ağırlıkları hesaplama
weight_for_bird = (1 / num_birds) * (total_samples / 2.0)
weight_for_drone = (1 / num_drones) * (total_samples / 2.0)

#Ağrılıklar
class_weights = {
    class_names.index('bird'): weight_for_bird,
    class_names.index('drone'): weight_for_drone
}

# patience=5 olarak güncelledim öğrenme oranına ve sınıflandırmaya adapte olması için
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
#En iyi modeli kaydeder earlystopping için gerekli.
model_checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
#çok iyi başarım gösteren v10 modelimden dinamik öğrenme
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=0.00001) #dinamik öğrenme

EPOCHS = 30 #EarlyStopping'in 
history = model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=EPOCHS,
  callbacks=[early_stopping, model_checkpoint, reduce_lr],
  class_weight=class_weights #Bu modelde yeni eklediğim parametre
)
#history'den eğitim ve validasyon metriklerini alıyorum.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
#Gerçekte kaç epoch çalıştığını history listesinin uzunluğundan anlıyorum.
actual_epochs = len(acc) 
epochs_range = range(actual_epochs) #X eksenini dinamik olarak ayarlıyorum.
test_loss, test_accuracy = model.evaluate(test_dataset) #accuracy hesaplama 
print(f"\nTest Seti Doğruluğu: {test_accuracy:.4f}")
print(f"Test Seti Kaybı: {test_loss:.4f}")
#Test setindeki tüm resimler için tahminleri al
y_pred_probs = model.predict(test_dataset) #karmaşıklık matrisi
y_pred = np.argmax(y_pred_probs, axis=1)
#gerçek etiketleri al
y_true = []
for images, labels in test_dataset:
  y_true.extend(np.argmax(labels.numpy(), axis=1))
y_true = np.array(y_true)
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Karmaşıklık Matrisi')
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.show()