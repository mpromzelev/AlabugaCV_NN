import cv2
import numpy as np
import tensorflow as tf 
from keras import layers
import keras

#all_num = 1109
all_num=1109
num=1
Train=[]
Labels=[]
while num<all_num:
    image = cv2.imread(f"./train/images/1 ({num}).jpg")
    with open(f"./train/labels/1 ({num}).txt", "r") as f:
        lines = f.readlines()

    Labels.append(int(lines[0][0]))
    kolvo=0
    srx1=0
    srx2=0
    sry1=0
    sry2=0

    for line1 in lines:
        line = line1.split()

        for i in range(1, len(line), 4):
            x_center = float(line[i])
            y_center= float(line[i+1])
            width = float(line[i+2])
            height = float(line[i+3])
            x = int(x_center*image.shape[1])
            y = int(y_center*image.shape[0])
            w = int(width*image.shape[1])
            h = int(height*image.shape[0])
            kolvo+=1
            srx1 += x-w//2
            sry1 += y-h//2
            srx2 += x+w//2
            sry2 += y+h//2
            if i+6>=len(line):
                break
    image2 = image[sry1//kolvo:sry2//kolvo, srx1//kolvo:srx2//kolvo]
    image2 = cv2.resize(image2, dsize = (128, 128), interpolation = cv2.INTER_AREA)
    Train.append(image2)
    num +=1
cv2.destroyAllWindows()
#print(Train)
#print(Labels)
Train = np.array(Train, dtype='float32')
Labels = np.array(Labels, dtype='uint8')
#print("------------------1")
#print(Train)
#print(Labels)
FINAL = []
Train = Train / 255

def cozdat_modelky():
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(64, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation = 'relu'),
        layers.Dense(6, activation = 'softmax')
        ])
    return model
            
model = cozdat_modelky()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(Train, Labels, epochs=10, batch_size=8, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(Train, Labels)

model.save_weights('modelN5.weights.h5')
print(f"Точность на тестовых данных: {test_accuracy:.2f}")
