import telebot
import keras
from keras import layers
import cv2
import numpy as np
from detect import detect
TOKEN = '7446407032:AAFkf9EUAWOoRsfXbC3uDTHu9EigcOI-WDI'
bob= telebot.TeleBot(TOKEN)

@bob.message_handler(commands=['start'])
def handle_start(message):
    bob.send_message(message.chat.id, 'Здравствуйте, скидывайте фото, а я определю - брак это или нет')

@bob.message_handler(content_types=['photo', 'document'])
def handle_file(message):
    if message.content_type == 'photo':
        photo = message.photo[-1]
    else:
        photo = message.document
    file_info = bob.get_file(photo.file_id)
    downloaded_file = bob.download_file(file_info.file_path)
    save_path = 'photo.jpg' #if message.content_type == 'photo' else photo.file_name
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    bob.reply_to(message, 'Фотография получена, ожидайте.')
    image = cv2.imread(f"photo.jpg")
    image = cv2.resize(image, dsize = (128, 128), interpolation = cv2.INTER_AREA)
    image = np.array(image, dtype='float32')
    image = image / 255.0
    predictions = model.predict(np.expand_dims(image, axis=0))
    predicted_class = np.argmax(predictions)
    
    if predicted_class==0:
        bob.send_message(message.chat.id, 'Я думаю, что это изображение относится к классу Bad Welding - плохой сварочный шов')
    elif predicted_class==1:
        bob.send_message(message.chat.id, 'Я думаю, что это изображение относится к классу Crack - трещина')
    elif predicted_class==2:
        bob.send_message(message.chat.id, 'Я думаю, что это изображение относится к классу Excess Reinforcement - избыточное армирование в шве')
    elif predicted_class==3:
        bob.send_message(message.chat.id, 'Я думаю, что это изображение относится к классу Good Welding - хороший шов')
    elif predicted_class==4:
        bob.send_message(message.chat.id, 'Я думаю, что это изображение относится к классу Porosity - пористость шва')
    elif predicted_class==5:
        bob.send_message(message.chat.id, 'Я думаю, что это изображение относится к классу Spatters - разбрызгивание')
    '''
    if predicted_class==3:
        bob.send_message(message.chat.id,"True")
    '''
    if predicted_class!=3:
        print(detect(save_path))
        bob.send_message(message.chat.id,(detect(save_path)))
    
    

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
model.load_weights('modelN5.weights.h5')
bob.polling()