# 새로운 이미지의 캡션 생성
from tensorflow import keras
import tensorflow as tf
from pickle import load
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.utils import img_to_array
from keras.utils import load_img
from keras.utils import pad_sequences
from numpy import argmax
from keras.preprocessing.text import Tokenizer
import numpy as np
from pyparsing import Word
from gtts import gTTS
from playsound import playsound
import time
import cv2
from googletrans import Translator
now = time.strftime('%Y%m%d%H%M%S'+'.jpg')
features = {}

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(100) < 0:
    ret, frame = capture.read()
    cv2.imshow("Video", frame)

cv2.imwrite('./kaggle/input/newimg/{}'.format(now),frame)


capture.release()
cv2.destroyAllWindows()
# 특징 추출
def extract_features(filename):
    # 이미지 분석 모델 불러오기
    model = VGG16()
    # 모델 재구성
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # 이미지 불러오기
    image = load_img(filename, target_size=(224, 224))
    # 이미지 픽셀을 numpy 배열로 변경
    image = img_to_array(image)
    # 배열 크기 변경
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # 이미지 처리
    image = preprocess_input(image)
    # 특징 추출
    feature = model.predict(image, verbose=0)
    return feature


# 벡터화된 단어를 실제 단어로 변환
# def word_for_id(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# # # 이미지 캡션 생성
# # def generate_desc(model, tokenizer, photo, max_length):
# #     # 시작 시퀀스
# #     in_text = 'startseq'
# #     # 시퀀스의 전체 길이를 반복
# #     for i in range(max_length):
# #         # sequence 벡터화
# #         sequence = tokenizer.texts_to_sequences([in_text])[0]
# #         # input의 길이맞추기(padding)
# #         sequence = pad_sequences([sequence], maxlen=max_length)
# #         # 다음 단어 예측
# #         yhat = model.predict([photo, sequence], verbose=0)
# #         # 예측한 결과값의 가장 큰 값을 통해
# #         # 이미지에 나와있을 가장 큰 가능성을 가진 단어를 찾음
# #         yhat = argmax(yhat)
# #         word = idx_to_word(yhat, tokenizer)
# #         # 매핑되는 단어가 없을때
# #         if word is None:
# #             break
# #         # 문장에 다음 단어 추가
# #         in_text += ' ' + word
# #         # 종료 시퀀스
# #         if word == 'endseq':
# #             break
# #     return in_text

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence],maxlen=max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

# tokenizer 불러오기
tokenizer = load(open('./kaggle/working/tokenizer.pkl', 'rb'))

# # 모델이 만들 수 있는 최대 문장 길이
max_length = 74
# # 모델 불러오기
# # 100 Dataset Model
# # model = load_model('model-ep020-loss4.394-val_loss5.023.h5')
# # 8000 Dataset Model
model = load_model('./kaggle/working/best_model.h5')



# converter= tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_spec.supported_ops=[
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     tf.lite.OpsSet.SELECT_TF_OPS
# ]
# tflite_model= converter.convert()
# with open('./keras_model.tflite','wb') as f:
#     f.write(tflite_model)
# model.summary()
photo1 = extract_features('./kaggle/input/newimg/{}'.format(now))
print(photo1)

description1 = predict_caption(model,photo1 , tokenizer, max_length)
tran = Translator()
new_str = description1.replace('startseq', '')
new_str1 = str(new_str.replace('endseq', ''))

result = tran.translate(new_str1, src='en', dest='ko')
print('1.jpg caption: ' + description1)

voice = gTTS(result.text, lang='ko')
voice.save('./voice/kor.mp3')
audio = './voice/kor.mp3'
print(new_str1)
print(result.text)
playsound(audio)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with open('./keras_model.tflite','wb') as f:
#     f.write(tflite_model)