import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

class TrashClassifier:
    CHUNK_SIZE = 40960
    DATA_SOURCE_MAPPING = 'trash-type-image-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F3863975%2F6704311%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240717%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240717T022422Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D03c2be96f92b5455c01a8d257115d17d0ab3934810cb150acb570d6dd2667db7249e4de689f552523c9ef89ac9c0be1675ce090bf9045154cb928f39126c48d5fa43f03a00f665c995810652b112bc4dd75687faa8552fdb2aa44b7c6a2d0fc7d36246d757674cea32bbec7ee05499f68cc2259f2e1cc494cfff2e78536bc698a6789de6dc76e618275393a092a619597995b20c1f52caa82056c433e579f6bbec03033f7999c96e3689b9e07fcf8d596442ed4a87d29be436f0a757cfdebad8880d3425c0e03bea91f94c5497d53a2053269c443dcc2772d459ad55680dcceb9b9b95a4daca9edba19adaec0f2b9f6105c9881299d04f038fa512f809b03fab'
    KAGGLE_INPUT_PATH = '/kaggle/input'
    KAGGLE_WORKING_PATH = '/kaggle/working'
    MODEL_DIR = 'model'
    MODEL_PATH = os.path.join(MODEL_DIR, 'trash_classifier.h5')

    def __init__(self):
        if not os.path.exists(self.MODEL_PATH):
            self._setup_directories()
            self.download_data()
            self.model = self.train_model()
        else:
            self.model = self.load_model()

    def _setup_directories(self):
        os.makedirs(self.KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
        os.makedirs(self.KAGGLE_WORKING_PATH, 0o777, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        try:
            os.symlink(self.KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
        except FileExistsError:
            pass
        try:
            os.symlink(self.KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
        except FileExistsError:
            pass

    def download_data(self):
        for data_source_mapping in self.DATA_SOURCE_MAPPING.split(','):
            directory, download_url_encoded = data_source_mapping.split(':')
            download_url = unquote(download_url_encoded)
            filename = urlparse(download_url).path
            destination_path = os.path.join(self.KAGGLE_INPUT_PATH, directory)
            
            if os.path.exists(destination_path):
                print(f'{directory} already exists at {destination_path}. Skipping download.')
                continue
            
            try:
                with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
                    total_length = fileres.headers['content-length']
                    print(f'Downloading {directory}, {total_length} bytes compressed')
                    dl = 0
                    data = fileres.read(self.CHUNK_SIZE)
                    while len(data) > 0:
                        dl += len(data)
                        tfile.write(data)
                        done = int(50 * dl / int(total_length))
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                        sys.stdout.flush()
                        data = fileres.read(self.CHUNK_SIZE)
                    if filename.endswith('.zip'):
                        with ZipFile(tfile) as zfile:
                            zfile.extractall(destination_path)
                    else:
                        with tarfile.open(tfile.name) as tarfile:
                            tarfile.extractall(destination_path)
                    print(f'\nDownloaded and uncompressed: {directory}')
            except HTTPError as e:
                print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
            except OSError as e:
                print(f'Failed to load {download_url} to path {destination_path}')
                
        print('Data source import complete.')

    def load_model(self):
        print(f'Loading existing model from {self.MODEL_PATH}')
        return keras.models.load_model(self.MODEL_PATH)

    def train_model(self):
        image_train_dataset, image_val_dataset = keras.utils.image_dataset_from_directory(
            '/kaggle/input/trash-type-image-dataset/TrashType_Image_Dataset',
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb",
            batch_size=64,
            image_size=(224, 224),
            seed=42,
            shuffle=True,
            validation_split=0.2,
            subset='both'
        )
        
        model = keras.Sequential([
            layers.InputLayer((224, 224, 3)),
            layers.Rescaling(1./255),
            layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.MaxPooling2D(pool_size=(7, 7)),
            layers.Flatten(),
            layers.Dense(6),
            layers.Activation("softmax")
        ])
        
        model.summary()
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy']
        )
        
        model.fit(
            image_train_dataset,
            validation_data=image_val_dataset,
            epochs=100
        )
        model.save(self.MODEL_PATH)
        return model

    def preprocess_image(self, img_path, target_size):
        img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    
    def predict(self, img_array):
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        return predicted_class
    
    def display_samples(self, dataset, n=9):
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(n):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(np.argmax(labels[i].numpy()))
                plt.axis("off")

if __name__ == "__main__":
    tc = TrashClassifier()
    img_array = tc.preprocess_image('static/img/sample.jpg', (224, 224))
    prediction = tc.predict(img_array)
    print(f'Predicted class: {prediction}')
