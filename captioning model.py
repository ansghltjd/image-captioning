#캡셔닝 모델 생성
import os
import pickle
import numpy as np
from requests import head
from tqdm.notebook import tqdm

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
BASE_DIR = './kaggle/input/flickr8k/'
WORKING_DIR = './kaggle/working'

# vgg16 모델 로드
model = VGG16()
# 모델을 재구성
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# 요약
print(model.summary())

# 이미지에서 특징 추출
features = {}
directory = os.path.join(BASE_DIR, 'Images')

#tqdm for문의 진행상황 확인, os.listdir 디렉토리에 있는 모든 파일을 가져온다.
print(directory)
a=0
for img_name in tqdm(os.listdir(directory)):
    
    # 파일에서 이미지 불러오기
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    # 이미지 픽셀을 numpy 배열로 변환
    image = img_to_array(image)

    # 모델에 대한 데이터 재구성
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
     # vgg를 위한 전처리 이미지
    image = preprocess_input(image)
    
     # 특징 추출
    feature = model.predict(image, verbose=0)
     # 이미지 ID 가져오기
    image_id = img_name.split('.')[0]
#     # store feature
    features[image_id] = feature
    a+=1
    print(a)
  
    
#print(image.shape[0]) 224
    #print(image.shape[1]) 224
    #print(image.shape[2]) 3




# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))  #pickle.dump(data, file)
# 피클에서 특징 로드
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:  #rb : 이진파일
    features = pickle.load(f)   #load : 한줄 저장
print(features)
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r',encoding="UTF-8") as f:
    next(f)   # 다음 데이터 읽어옴
    captions_doc = f.read()  

# 캡션에 대한 이미지 매핑 생성
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # 쉼표(,)로 줄을 나눕니다.
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    
    # 이미지 ID에서 확장자 제거
    image_id = image_id.split('.')[0]
    
    # 캡션 목록을 문자열로 변환
    caption = " ".join(caption)
    
    # 필요한 경우 목록 생성
    
    if image_id not in mapping:
        mapping[image_id] = []
    # 캡션 저장
    mapping[image_id].append(caption)
 
   

# print(len(mapping))


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # 한 번에 하나의 캡션을 가짐
            caption = captions[i]
            # 전처리 단계
            # 소문자로 변환
            caption = caption.lower()
            # 숫자, 특수 문자 등 삭제,
            caption = caption.replace('[^A-Za-z]', '')
            # 추가 공백 삭제
            caption = caption.replace('\s+', ' ')
            # 캡션에 시작 및 종료 태그 추가
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
            
# before preprocess of text

print(mapping['1000268201_693b08cb0e'])

# preprocess the text
clean(mapping)

# after preprocess of text
print(mapping['1000268201_693b08cb0e'])
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
print(len(all_captions))
print(all_captions[:10])

# 텍스트를 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

# tokenizer 저장
pickle.dump(tokenizer, open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'wb'))

#사용 가능한 캡션의 최대 길이 가져오기
max_length = max(len(caption.split()) for caption in all_captions)
print(max_length)

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

# startseq girl going into wooden building endseq
#        X                   y
# startseq                   girl
# startseq girl              going
# startseq girl going        into
# ...........
# startseq girl going into wooden building      endseq
# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # 각 캡션 처리
            for caption in captions:
                # 시퀀스를 인코딩
                seq = tokenizer.texts_to_sequences([caption])[0]
                # 시퀀스를 X, y 쌍으로 분할
                for i in range(1, len(seq)):
                    # 입력 및 출력 쌍으로 분할
                    in_seq, out_seq = seq[:i], seq[i]
                    # 패드 입력 시퀀스
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # 인코딩 출력 시퀀스
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # 시퀀스 저장
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0
# encoder model
# image feature layers

inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# # decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
plot_model(model, show_shapes=True)

# train the model
epochs = 15
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
# save the model
model.save(WORKING_DIR+'/best_model.h5')


# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None
# # generate caption for an image
# def predict_caption(model, image, tokenizer, max_length):
#     # add start tag for generation process
#     in_text = 'startseq'
#     # iterate over the max length of sequence
#     for i in range(max_length):
#         # encode input sequence
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         # pad the sequence
#         sequence = pad_sequences([sequence], max_length)
#         # predict next word
#         yhat = model.predict([image, sequence], verbose=0)
#         # get index with high probability
#         yhat = np.argmax(yhat)
#         # convert index to word
#         word = idx_to_word(yhat, tokenizer)
#         # stop if word not found
#         if word is None:
#             break
#         # append word as input for generating next word
#         in_text += " " + word
#         # stop if we reach end tag
#         if word == 'endseq':
#             break
      
#     return in_text
# from nltk.translate.bleu_score import corpus_bleu
# # validate with test data
# actual, predicted = list(), list()

# for key in tqdm(test):
#     # get actual caption
#     captions = mapping[key]
#     # predict the caption for image
#     y_pred = predict_caption(model, features[key], tokenizer, max_length) 
#     # split into words
#     actual_captions = [caption.split() for caption in captions]
#     y_pred = y_pred.split()
#     # append to the list
#     actual.append(actual_captions)
#     predicted.append(y_pred)
    
# # calcuate BLEU score
# print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
# print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

# from PIL import Image
# import matplotlib.pyplot as plt
# def generate_caption(image_name):
#     # load the image
#     # image_name = "1001773457_577c3a7d70.jpg"
#     image_id = image_name.split('.')[0]
#     img_path = os.path.join(BASE_DIR, "Images", image_name)
#     image = Image.open(img_path)
#     captions = mapping[image_id]
#     print('---------------------Actual---------------------')
#     for caption in captions:
#         print(caption)
#     # predict the caption
#     y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
#     print('--------------------Predicted--------------------')
#     print(y_pred)
#     plt.imshow(image)
#     plt.show()
# generate_caption("1001773457_577c3a7d70.jpg")






