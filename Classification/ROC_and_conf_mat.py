from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from itertools import cycle


#train_data_dir='/ghome/mcv/datasets/MIT_split/train'
train_data_dir = "C:/Users/MICHE/Documents/Datasets/Cloudsen12/split2/train"
#val_data_dir='/ghome/mcv/datasets/MIT_split/test'
test_data_dir = "C:/Users/MICHE/Documents/Datasets/Cloudsen12/split2/test"
#test_data_dir='/ghome/mcv/datasets/MIT_split/test'

img_width = 224
img_height= 224
batch_size=32

initial_model = "C:/Users/MICHE/Desktop/Master/MTP/tensorFlow/results/best_model.h5"



# create the base pre-trained model
model = load_model(initial_model)



datagenTest = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	preprocessing_function=preprocess_input,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)


test_generator = datagenTest.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=300,
        class_mode='categorical')

labels = ["0 to 25 ", "25 to 50", "50 to 75", "75 to 100"]
X, y_test = test_generator.next()
y_pred = model.predict(X)

y_test_1 = np.argmax(y_test, axis=1)
y_pred_1 = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test_1, y_pred_1)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()


# Compute ROC curve and ROC area for each class
n_classes = len(labels)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(4,4))
lw = 2
plt.plot(fpr["macro"], tpr["macro"],
        label='average ROC curve (auc = {0:0.3f})'
              ''.format(roc_auc["macro"]),
        color='navy', linestyle=':', linewidth=4)

palette = sns.color_palette("hls", 4)
colors = cycle(palette)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
            label='Class {0} (auc = {1:0.3f})'
            ''.format(labels[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve ')
plt.legend(loc="lower right")
