from keras.models import load_model
import numpy as np 
from keras.preprocessing.image import img_to_array,load_img
from keras.models import model_from_json
import cv2
import matplotlib.pyplot as plt 

model_weight = "cat_CNN_result.h5"
model_name = "cat_CNN_model.json"

#モデルの読み込み
model = model_from_json(open(model_name).read())

#重みの読み込み
model.load_weights(model_weight)

# テスト画像の読み込み
img = cv2.imread("test.jpg")
# 画像サイズの変更
img = cv2.resize(img,(100,100))
# 画像の色の順番を変更
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = img.astype(float)
# 正規化
img /= 255.0

# ラベルを設定
keys = ["persian","americanshorthair","munchkin","bengal","russianblue"]

#予測を行う
pred = model.predict(img.reshape(1,100,100,3)).argmax()

#予測結果の出力
print("predict type:",keys[pred])