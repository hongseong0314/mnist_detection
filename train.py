import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

from model import load_model

def train():
    """
    keras 내장 데이터셋 mnist를 통하여 학습
    """
    mnist = tf.keras.datasets.mnist

    # 데이터 정규화
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # load model
    model = load_model()
    
    create_directory('model')
    filename = 'model/model_wights.h5'

    checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                                monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                                verbose=1,            # 로그를 출력합니다
                                save_best_only=True,  # 가장 best 값만 저장합니다
                                mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                                )

    earlystopping = EarlyStopping(monitor='val_loss',  # 모니터 기준 설정 (val loss) 
                                patience=3,         # 10회 Epoch동안 개선되지 않는다면 종료
                                )

    # adam외의 optimizer로 변경
    # sparse_categorical_crossentropy외의 loss로 변경
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                )
    
    # epochs 값 변경
    model.fit(x_train, y_train, epochs=10, 
                validation_data=(x_test,  y_test),
                callbacks=[checkpoint, earlystopping])

    # 임의의 5가지 test data의 이미지와 레이블값을 출력하고 예측된 레이블값 출력
    predictions = model.predict(x_test)
    idx_n = np.random.randint(1,len(x_test), 5)

    for i in idx_n:
        img = x_test[i].reshape(28,28)
        plt.imshow(img,cmap="gray")
        plt.show()
        # plt.savefig("Number_test{}.png".format(i))

        print("Label: ", y_test[i])
        print("Prediction: ", np.argmax(predictions[i]))
        

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    train()