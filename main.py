import numpy as np
import cv2
import os


# Функция отрисовки цифр - далее понадобится, чтобы отрисовать циферки для демонстрации работы нейронки
def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    plt.figure(figsize=(2 * n, 2 * len(args)))

    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i * n + j + 1)
            plt.imshow(args[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig("pictures.png")
    plt.show()



train_path = "dataset/train"

# инициализируем данные
print("[INFO] loading images...")
data = []
train_labels = os.listdir(train_path)
train_labels.sort()

# просто считываем изображения из каждой папки, Ресайзим их и запоминаем
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name
    path = train_path + "/" + training_name
    x = os.listdir(dir)
    for y in x:
        file = path + "/" + str(y)
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        # масштабируем интенсивности пикселей в диапазон [0, 1]
        image = image.astype('float32') / 255
        data.append(image)
    print("[INFO] processed folder: {}".format(current_label))

# Преобразование в матрицу Numpy
data = np.reshape(data, (len(data), 28, 28, 1))

print("[INFO] completed loading images...")
print("Размер данных: ", len(data))

# разбиваем данные на обучающую и тестовую выборки, используя 80% данных для обучения и оставшиеся 20% для тестирования
from sklearn.model_selection import train_test_split
(trainX, testX) = train_test_split(data, test_size=0.2, random_state=73)
print("Размер Train и Test выборки: ", len(trainX), len(testX))



# Создадим функцию для создания архитектур моделей Кодера, Декодера, Автоэнкодера
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
def create_denses():
    encoding_dim = 49   # Размер слоя кодировки - свернем до такого размера, а затем развернем

    # Энкодер: 1) Входное изображение 2) Flatten 3) Полносвязный для кодирования
    input_img = Input(shape=(28, 28, 1))    # Входной слой для получения картинки на входе
    flat_img = Flatten()(input_img)  # Flatten слой для вытягивания картинки в строку для подачи на вход нейронки
    encoded = Dense(encoding_dim, activation='relu')(flat_img)  # Полносвязный слой меньшей размерности

    # Декодер: 1) Входной код изображения 2) Декодирование изображения 3)* Информация по картинке
    input_encoded = Input(shape=(encoding_dim,))    # Входной слой размерности закодированной картинки
    flat_decoded = Dense(28 * 28, activation='sigmoid')(input_encoded)  # Расширенный слой до изначального размера
                                                                        # картинки - декодирование
    decoded = Reshape((28, 28, 1))(flat_decoded)    # Формат картинки

    # определим архитектуру для кодера
    encoder = Model(input_img, encoded, name="encoder")
    # определим архитектуру для декодера
    decoder = Model(input_encoded, decoded, name="decoder")
    # определим архитектуру для автоэнкодера
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

# Создаем и компилируем модель автоэнкодера
encoder, decoder, autoencoder = create_denses()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()



# Сделаем на основе обычного автоэнкодера модель автоэнкодера с возможностью шумоподавления
import keras.backend as K
from keras.layers import Lambda
batch_size = 16
# Создадим функцию для создания архитектур моделей для Зашумления и Декодирования зашумленных картинок
def create_denoising_model(autoencoder):
    def add_noise(x):
        noise_factor = 0.3
        # создаем матрицу с шумами и склеиваем её с исходным изображением
        x = x + K.random_normal(x.get_shape(), 0.2, noise_factor)
        x = K.clip(x, 0., 1.)
        return x

    # на входной слой подаем по 16 картинок
    input_img  = Input(batch_shape=(batch_size, 28, 28, 1))
    # слоем который принимает функцию добавляем шумы на картинки
    noised_img = Lambda(add_noise)(input_img)

    noiser = Model(input_img, noised_img, name="noiser")
    # с помощью уже созданного автокодера создаем новую модель, способную убирать шумы
    denoiser_model = Model(input_img, autoencoder(noiser(input_img)), name="denoiser")
    return noiser, denoiser_model

noiser, denoiser_model = create_denoising_model(autoencoder)
denoiser_model.compile(optimizer='adam', loss='binary_crossentropy')
denoiser_model.summary()



# обучаем нейросеть-автоэнкодер для зашумленных картинок - 50 эпох
epochs = 50
H = denoiser_model.fit(trainX, trainX, validation_data=(testX, testX), epochs=epochs, batch_size=batch_size)



# строим графики потерь
import matplotlib.pyplot as plt
plt.style.use("ggplot")
N = np.arange(1, epochs+1)
plt.figure()
plt.plot(N, H.history["loss"], label="training loss")
plt.plot(N, H.history["val_loss"], label="validation loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("LOSS_func.png")
plt.show()



# Попробуем прогнать по автоэнкодеру некоторые картинки из тестов
n = 15
# отрисуем первые 15 картинок чтобы посмотреть эффективность автокодера
imgs = testX[:batch_size]
noised_imgs = noiser.predict(imgs, batch_size=batch_size)
encoded_imgs = encoder.predict(noised_imgs[:n],  batch_size=n)
decoded_imgs = decoder.predict(encoded_imgs[:n], batch_size=n)

plot_digits(imgs[:n], noised_imgs, decoded_imgs)
