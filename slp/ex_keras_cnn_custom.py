# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as nr
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku
import matplotlib.pyplot as plt



def drawFeatMap(dstFn, featMap):
    fig, ax = plt.subplots()
    ax.imshow(featMap, cmap='gray')
    plt.savefig(dstFn)
    
    # flush
    plt.cla()
    plt.clf()
    plt.close()
    
    

nr.seed(12345)  # random seed 설정

# 데이터 정의
data1 = np.array([1.0, 1.0, 1.0, 0.9,
                  1.0, 0.0, 0.0, 1.0,
                  1.0, 0.0, 0.0, 1.0,
                  0.9, 1.0, 1.0, 0.9]).reshape(1, 4, 4, 1) # "0"
data2 = np.array([0.0, 1.0, 1.0, 0.0,
                  0.0, 0.9, 1.0, 0.0,
                  0.0, 1.0, 0.9, 0.0,
                  0.0, 1.0, 1.0, 0.0]).reshape(1, 4, 4, 1) # "1"
data3 = np.array([0.0, 1.0, 1.0, 0.0,
                  0.0, 0.9, 0.9, 0.0,
                  0.0, 0.9, 0.9, 0.0,
                  0.0, 0.9, 0.9, 0.0]).reshape(1, 4, 4, 1) # "1"
data4 = np.array([1.0, 0.9, 1.0, 1.0,
                  1.0, 0.0, 0.0, 1.0,
                  1.0, 0.0, 0.0, 1.0,
                  1.0, 0.9, 0.9, 1.0]).reshape(1, 4, 4, 1) # "0"
trFeatArr = np.vstack((data1, data2, data3, data4))
print 'trFeatArr', trFeatArr.shape
print trFeatArr

# 레이블 정의(one-hot representation)
trLabelArr = np.array([1.0, 0.0,
                       0.0, 1.0,
                       0.0, 1.0,
                       1.0, 0.0]).reshape(-1, 2)
print 'trLabelArr', trLabelArr.shape
print trLabelArr

# 모델 구성(2(input) -> CONV(ReLU) -> CONV(ReLU) -> FC(sigmoid))
inputFeat = kl.Input(shape=(4, 4, 1))

conv1 = kl.Conv2D(filters=5,
                  kernel_size=(3, 3), strides=1,
                  padding='same')(inputFeat)    # zero-padding
relu1 = kl.Activation('relu')(conv1)
conv2 = kl.Conv2D(filters=3,
                  kernel_size=(3, 3), strides=1)(relu1)
relu2 = kl.Activation('relu')(conv2)
flatten = kl.Flatten()(relu2)
dense = kl.Dense(units=2)(flatten)
output = kl.Activation('sigmoid')(dense)

modelFull = km.Model(inputs=[inputFeat], outputs=[output])
modelRelu1 = km.Model(inputs=[inputFeat], outputs=[relu1])


# 학습 설정(MSE / SGD / learning rate decay / momentum)
modelFull.compile(loss='mean_squared_error',
                  optimizer=ko.SGD(lr=0.1, decay=0.01, momentum=0.9))

# 모델 구조 그리기
ku.plot_model(modelFull, 'model.png')

# 학습(500회 반복, 4개 샘플씩 배치 학습)
modelFull.fit(trFeatArr, trLabelArr, epochs=500, batch_size=4)

# 테스트
res = modelFull.predict(trFeatArr, batch_size=4)
print 'res', res.shape
print res
print np.argmax(res, axis=1)

# 테스트2
res2 = modelRelu1.predict(trFeatArr, batch_size=4)
print 'res2', res2.shape
print res2

for sampleIndex in range(res2.shape[0]):
    for filterIndex in range(res2.shape[3]):
        dstFn = '%d_%d.png' % (sampleIndex, filterIndex)
        featMap = res2[sampleIndex, :, :, filterIndex]
        drawFeatMap(dstFn, featMap)