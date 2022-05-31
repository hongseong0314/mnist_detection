import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def mnist_detection(img, verbose=False, margin_pixel = 50):
    """
    img를 받아서 img 안에서 숫자를 탐지후 각각으로 나눠서 반환.
    img : 숫자가 적힌 이미지
    mergin_pixel : 숫자 탐지 후 자를 마진 범위
    """
    if verbose:
        print("원본 이미지를 출력합니다.")
        plt.figure(figsize=(15,12))
        plt.imshow(img)
        plt.show()
    
    # 이미지 이진화
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_th = cv2.threshold(img_blur, 155, 250, cv2.THRESH_BINARY_INV)[1]
    
    # 경계선 탐지
    contours, hierachy= cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rects = [cv2.boundingRect(each) for each in contours]
    
    #탐지 사각형 넓이 계산
    tmp = [w*h for (x,y,w,h) in rects]
    
    rects = [(x,y,w,h) for (x,y,w,h) in rects if ((w*h>1000)and(w*h<500000))]
    
    print("\n이미지를 분할할 영역을 표시합니다.")
    for rect in rects:
    # 원래 이미지에 탐지한 부분 사각형으로 표시
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5) 
    plt.clf()
    plt.figure(figsize=(15,12))
    plt.imshow(img)
    plt.show()
    
    seg_img = []

    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 

        # 탐지한 사각형이 지정 범위안에 있으면
        if ((w*h>1000)and(w*h<500000)): 

            # 탐지 사각형을 마진을 더하여 잘라준다.
            cropped = img.copy()[y - margin_pixel:y + h + margin_pixel, x - margin_pixel:x + w + margin_pixel] 
            seg_img.append(cropped)

            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 5)

    print("\n분할 된 이미지를 출력합니다.")
    for i in range(len(seg_img)):
        plt.imshow(seg_img[i])
        plt.show()
        # plt.savefig("result4-{}.png".format(i+1))
    
    re_seg_img = []

    for i in range(len(seg_img)):
        re_seg_img.append(cv2.resize(seg_img[i], (28,28), interpolation=cv2.INTER_AREA))
    
    # gray = cv2.cvtColor(re_seg_img[0], cv2.COLOR_BGR2GRAY)
    return re_seg_img


def data_predit(re_seg_img):
    """
    숫자이미지가 적힌 img 배열을 받아서 detection 후 분류를 해준다.
    """
    checkpoint_dir = os.path.join(os.getcwd(), 'model/model_wights.h5')
    
    #이전 모델 가중치 넣어주기
    from model import load_model
    model = load_model()
    model.load_weights(checkpoint_dir)
    
    for i in range(len(re_seg_img)):
    
        gray = cv2.cvtColor(re_seg_img[i], cv2.COLOR_BGR2GRAY)
        img_binary = cv2.threshold(gray, 150, 250, cv2.THRESH_BINARY_INV)[1]
        test = img_binary.reshape(1,28,28) / 255.0

        #print(len(test))

        predictions = model.predict(test)

        img_test = test.reshape(28,28)
        plt.clf()
        
        plt.subplot(121)
        plt.imshow(re_seg_img[i])
        plt.title('Origin')

        plt.subplot(122)
        plt.imshow(img_test,cmap="gray")
        plt.title('Coverted')
        plt.show()
        
        # plt.savefig("result4.png")

        #print("Label: ", y_test[i])
        print("Prediction: ", np.argmax(predictions))