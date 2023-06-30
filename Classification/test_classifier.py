from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, regularizers, Input, activations, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, SeparableConv2D, Dense, Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
from optuna.visualization import matplotlib as optuna_plots
import pandas as pd
import os
import sys




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
def get_optimizer(name, trial):
    if name == 'SGD':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == 'Adam':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif name == 'RMSprop':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif name == 'Adagrad':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif name == 'Adadelta':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif name == 'NAG':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        momentum = trial.suggest_float('momentum', 0.0, 1.0)
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    elif name == 'Adamax':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif name == 'Nadam':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer name: {name}")


def createModel(configs):
    
    activations = {"relu": relu}
    regularizer = regularizers.l2(configs["reg_coef"])
    
    # Create model
    img_inputs = Input(shape=(configs["image_height"], configs["image_width"], 3))
    conv1 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(img_inputs)
    act1 = Activation(activations[configs["activation"]])(conv1)
    batch1 = BatchNormalization()(act1)
    mPool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch1)

    conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(mPool1)
    act2 = Activation(activations[configs["activation"]])(conv2)
    batch2 = BatchNormalization()(act2)
    mPool2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch2)


    conv3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(mPool2)
    act3 = Activation(activations[configs["activation"]])(conv3)
    batch3 = BatchNormalization()(act3)
    mPool3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch3)

    conv4 = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(mPool3)
    act4 = Activation(activations[configs["activation"]])(conv4)
    batch4 = BatchNormalization()(act4)
    mPool4 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch4)

    conv5 = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(mPool4)
    act5 = Activation(activations[configs["activation"]])(conv5)
    batch5 = BatchNormalization()(act5)
    
    global_pool = GlobalAveragePooling2D()(batch5)

    fc = Dense(configs["n_class"], activation='softmax', name='predictions', kernel_regularizer = regularizer)(global_pool)

    model = Model(inputs=img_inputs, outputs=fc)

    return model


def objective(trial):
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'Adagrad', 'Adadelta', 'Nadam'])
    batchSize = 32
    widthShift= trial.suggest_float('width_shift_range',0.0,0.5, step=0.1)
    heightShift = trial.suggest_float('heightShift_shift_range',0.0,0.5, step=0.1)
    horizontalFlip =  trial.suggest_categorical('horizontal_flip',[True,False])
    verticalFlip =  trial.suggest_categorical('vertical_flip',[True,False])
    rotation = trial.suggest_int('Rotation range', 0, 45, step=5)
    zoom = trial.suggest_float('zoom',0,1, step=0.1)
    brightnes = trial.suggest_categorical('brightness',[[0,0],[0,1]]),
    shearRange = 0
    
    configs = dict(
        dataset = 'subset cloudsen12',
        n_class = 4,
        image_width = 256,
        image_height = 256,
        batch_size = batchSize,
        epochs = 50,
        loss_fn = 'categorical_crossentropy',
        activation = "relu",
        regularizer = "l2",
        reg_coef = 0.01,
        # Data augmentation
        width_shift_range = widthShift,
        height_shift_range = heightShift,
        horizontal_flip = horizontalFlip,
        vertical_flip = verticalFlip,
        rotation_range = rotation,
        brightness_range = [0, 0],
        zoom_range = zoom,
        shear_range = 0
    
    )

    train_data_dir= '/home/msiau/workspace/cloudsen12/split/train'
    test_data_dir = '/home/msiau/workspace/cloudsen12/split/test'
    validation_samples = 1144
    
    folder = "./results/"
    MODEL_FNAME = folder + "best_model.h5"
    
    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet')
   # plot_model(base_model, to_file='modelResNet50a.png', show_shapes=True, show_layer_names=True)
    
    x = base_model.layers[-2].output
    x = Dense(4, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = True
    
    optimizer = get_optimizer(optimizer_name, trial)

    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    for layer in model.layers:
        print(layer.name, layer.trainable)
    
    model.save(MODEL_FNAME)
    #preprocessing_function=preprocess_input,
    datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
    	rescale=1./255.,
        rotation_range=configs["rotation_range"],
        width_shift_range=configs["width_shift_range"],
        height_shift_range=configs["height_shift_range"],
        shear_range=configs["shear_range"],
        brightness_range = configs["brightness_range"],
        zoom_range=configs["zoom_range"],
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=configs["horizontal_flip"],
        vertical_flip=configs["vertical_flip"])
    
    datagenTest = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
    	rescale=1./255.,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False)
    
    train_generator = datagen.flow_from_directory(train_data_dir,
            target_size=(configs["image_height"], configs["image_width"]),
            batch_size=configs["batch_size"],
            class_mode='categorical')
    
    test_generator = datagenTest.flow_from_directory(test_data_dir,
            target_size=(configs["image_height"], configs["image_width"]),
            batch_size=configs["batch_size"],
            class_mode='categorical')
    
    
    
    history=model.fit(train_generator,
            steps_per_epoch=(int(300//configs["batch_size"])+1),
            epochs=configs["epochs"],
            validation_data=test_generator,
            validation_steps= (int(validation_samples//configs["batch_size"])+1), callbacks=[])

    
    # To save the best model
    checkpointer = ModelCheckpoint(filepath=MODEL_FNAME, verbose=1, save_best_only=True, 
                                   monitor='val_accuracy')
    
    # Finish
    
    
    model = load_model(MODEL_FNAME)
    
    result = model.evaluate(test_generator)
    loss = model.evaluate(test_generator)[0]
    #accuracy  = model.evaluate(test_generator)[1]
    print( result)
    
    
    # list all data in history
    num_p = model.count_params()
    
    file  = open(folder + "results.txt", "w")
    file.write(f"Score: {result}\n")
    file.write(f"Num param: {num_p}\n")
    ratio = result[1]/(num_p/100000)
    file.write(f"Ratio: {ratio}\n")
    file.close()
    
    # summarize history for accuracy
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='upper left')
    fig1.savefig(folder + 'accuracy.jpg')
    plt.close(fig1)
      # summarize history for loss
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='upper left')
    fig1.savefig(folder + 'loss.jpg')
    return loss



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
best_params = study.best_params
print("Best Hyperparameters:", best_params)
# Get the trials DataFrame
trials_df = study.trials_dataframe()

# Save the DataFrame to a .csv file
trials_df.to_csv('optuna_results.csv', index=False)

# Save the DataFrame to an .xlsx file
trials_df.to_excel('optuna_results.xlsx', index=False)


# Plot the optimization history
plot = plot_optimization_history(study)
plot.write_image("optimization_history.png")
plot1 = plot_param_importances(study)
plot1.write_image("param.png")
plot2 = plot_slice(study)
plot2.write_image("slice.png")
plot3 = plot_contour(study)
plot3.write_image("contour.png")

# Generate a plot using Optuna's visualization functions
#fig = optuna_plots.plot_optimization_history(study)
#fig1= optuna_plots.plot_contour(study)
#fig2= optuna_plots.plot_slice(study)
#fig3= optuna_plots.plot_param_importances(study)

# Save the figure using savefig method of the figure object
#fig.figure.savefig('plot_optimization_history.png')
#fig1.figure.savefig('plot_contour.png')
#fig2.figure.savefig('plot_slice.png')
#fig3.figure.savefig('plot_param_importances.png')