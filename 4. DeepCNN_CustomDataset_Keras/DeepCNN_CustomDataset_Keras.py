import keras
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.optimizers import SGD
from keras.models import  Model
import numpy as np

# create the base pre-trained  model
base_model = InceptionV3(weights='imagenet',
                         input_shape=(150,150,3),
                         include_top=True)
base_model.summary()
x = base_model.get_layer(name = 'avg_pool').output
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# Create a Model to Train
model = Model(inputs = base_model.input, outputs = predictions)

# Freeze base_model Layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=SGD(lr=0.01),
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

### Loading Data:
train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
        'hymenoptera_data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle = True)

validation_generator = test_datagen.flow_from_directory(
        'hymenoptera_data/val',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle = True)

model.fit_generator(
        train_generator,
        steps_per_epoch=int(250/32),
        epochs=3,
        validation_data=validation_generator,
        validation_steps=int(153/32))

model.evaluate_generator(
        validation_generator,
        steps=int(153/32)
)