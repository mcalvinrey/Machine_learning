
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def ClassifyC_M30():
    data_dir = 'C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/Fotos_M30'  # Carpeta con subcarpetas
    tt = 128   # Tamaño de las fotos (size = (tt*rat, tt))(ancho, alto)
    rat = 1.5  # Ratio ancho/alto para fotos
    batch_size = 32    # Batch_size 
    epochs = 50    # Epochs
    datAug = True  # Activar Data Augmentation (adding layer)

    
    start_time = time.time()
    data_dir = pathlib.Path(data_dir)
    
    img_height = tt
    img_width = int(tt * rat)
    
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split = 0.2, subset = "both",
    seed = 2123, image_size=(img_height, img_width), batch_size = batch_size)
    
    class_names = train_ds.class_names
    train_ds, test_ds=tf.keras.utils.split_dataset(train_ds, left_size=None, 
            right_size=.20,)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print('\nTenemos', image_count, 'fotos')
    print('\nCategorías', class_names)
    num_classes = len(class_names)
        
    fluido = list(data_dir.glob('fluido/*'))
    congestionado = list(data_dir.glob('congestionado/*'))

    fig = plt.figure(figsize=(15, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    fig.savefig('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/Imagenes_M30/Ejemplos.jpg', 
                            dpi = 320)
    #plt.show()
        
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = batch_size)
    val_ds = val_ds.cache().prefetch(buffer_size = batch_size)
    
    # Model Definition
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)])
        
    model.compile(optimizer = 'adamax', loss = tf.keras.losses.
            SparseCategoricalCrossentropy(from_logits = True),
            metrics = ['accuracy'])

    print(model.summary())
        
    history = model.fit(train_ds, validation_data=val_ds, epochs = epochs)
        
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
        
    loss = history.history['loss']
    val_loss = history.history['val_loss']
        
    epochs_range = range(epochs)
    
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    fig.savefig('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/Imagenes_M30/Training.jpg', 
                dpi = 320)
    #plt.show()

    print('\n(LOSS, ACCURACY) EN CONJUNTO DE TEST:\n', model.evaluate(test_ds),
        '\n')
        
    predicciones = np.argmax(model.predict(test_ds), axis= 1)

    predicciones = tf.convert_to_tensor(predicciones, dtype = tf.int32)
    print('\nSHAPE DE PREDICCIONES:', predicciones.shape)
    #print('\nPREDICCIONES:', predicciones)

    y_true = pd.DataFrame()
    for image, label in test_ds:  
        label = pd.DataFrame(label)
        y_true = pd.concat([y_true,label], ignore_index = True) 
        
    y_true = tf.convert_to_tensor(y_true, dtype = tf.int32)
    print('\nSHAPE DE Y_TRUE:', y_true.shape)
    #print('\nY_TRUE:', y_true)
        
    mc = tf.math.confusion_matrix(y_true, predicciones)
    #print('\nCONFUSION MATRIX TYPE:', type(mc))
    print('\nMATRIZ DE CONFUSIÓN:\n', mc)
        
    img = tf.keras.utils.load_img('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/Imagenes_M30/pruTrafM30.jpg',
        target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
        
    print("\nImage seems to belong to", class_names[np.argmax(score)],
        ' with a', round(100 * np.max(score)), " percent confidence\n")

    if datAug:
        # Vemos como funciona Data Augmentation:
        data_augmentation = keras.Sequential([layers.RandomFlip("horizontal",
        input_shape=(img_height, img_width, 3)),
            layers.RandomRotation(0.10), layers.RandomZoom(0.15),])
            
        fig2 = plt.figure(figsize=(15, 10))
        for images, _ in train_ds.take(1):
            for i in range(9):
                augmented_images = data_augmentation(images)
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_images[0].numpy().astype("uint8"))
                plt.axis("off")
        fig2.savefig('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/Imagenes_M30/Training2.jpg', 
                        dpi = 320)
        #plt.show()
                    
        # Añadimos al principio layers de  Data Augmentation, para training:
        modelAug = Sequential([
            layers.RandomFlip("horizontal", input_shape=(img_height,img_width,3)),
            layers.RandomRotation(0.1), 
            layers.RandomZoom(0.15),
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)])
            
        modelAug.compile(optimizer = 'adamax', loss = tf.keras.losses.
                SparseCategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy'])
        
        print(modelAug.summary())
            
        historyAug = modelAug.fit(train_ds, validation_data=val_ds, epochs = epochs)
            
        acc = historyAug.history['accuracy']
        val_acc = historyAug.history['val_accuracy']
            
        loss = historyAug.history['loss']
        val_loss = historyAug.history['val_loss']
            
        fig = plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Accuracy using Data Augmentation')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Loss using Data Augmentation')
        #plt.show()
        fig.savefig('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/Imagenes_M30/AugmentedTraining.jpg', 
            dpi = 320)
            
        print('\n(LOSS, ACCURACY) EN CONJUNTO DE TEST:\n', modelAug.evaluate(test_ds),
        '\n')
        
        prediccionesAug = np.argmax(modelAug.predict(test_ds), axis= 1)

        prediccionesAug = tf.convert_to_tensor(prediccionesAug, dtype = tf.int32)
        print('\nSHAPE DE PREDICCIONES:', prediccionesAug.shape)
        #print('\nPREDICCIONES:', predicciones)

        y_trueAug = pd.DataFrame()
        for image, label in test_ds:  
            label = pd.DataFrame(label)
            y_trueAug = pd.concat([y_trueAug,label], ignore_index = True) 
        
        y_trueAug = tf.convert_to_tensor(y_trueAug, dtype = tf.int32)
        print('\nSHAPE DE Y_TRUE:', y_trueAug.shape)
        #print('\nY_TRUE:', y_true)
        
        mcAug = tf.math.confusion_matrix(y_trueAug, prediccionesAug)
        #print('\nCONFUSION MATRIX TYPE:', type(mc))
        print('\nMATRIZ DE CONFUSIÓN:\n', mcAug)
        
        img = tf.keras.utils.load_img('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/Imagenes_M30/pruTrafM30.jpg',
            target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictionsAug = modelAug.predict(img_array)
        scoreAug = tf.nn.softmax(predictionsAug[0])
        
        print("\nAUG. Image seems to belong to", class_names[np.argmax(scoreAug)],
           ' with a', round(100 * np.max(scoreAug)), " percent confidence\n")
            
        
        
    # Guardar el modelo en folder \model:
    model.save("C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/modelos/M-30/model")
    modelAug.save("C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/modelos/M-30/modelAug")
        
    # To use later:
    #loaded_model = tf.keras.models.load_model(
    #    "C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/modelos/M-30/model")
    #loaded_modelAug = tf.keras.models.load_model(
    #    "C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/modelos/M-30/modelAug")
    # Comprobar si se ha cargado bien:
    #assert np.allclose(model.predict(img_array), 
    #   loaded_model.predict(img_array)), "Modelo cargado != modelo"
    #assert np.allclose(modelAug.predict(img_array), 
    #   loaded_modelAug.predict(img_array)), "Modelo cargado != modelo"
        
    elapsed_time = time.time() - start_time
    print('\n',round(elapsed_time/60,2), 'minutes passed\n')
