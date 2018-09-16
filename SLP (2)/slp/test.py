# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk#피클 모듈


# MNIST 데이터 경로
_SRC_PATH = u'C:\\Users\\IWG\\Desktop\\slp-20171209T093608Z-001\\slp\\mnist'#현재 작업중인 컴퓨터의 mnist 파일 경로
_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels.idx1-ubyte'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL

# 출력 이미지 경로
_DST_PATH = u'img_gray'


#####activation function g(z)
def activation(x):
    return 1/(1+np.exp(-x))
    #activation function의 미분
def gPrime(a):
    
    return a*(1-a)



def drawImage(dataArr, fn):
    fig, ax = plt.subplots()
    ax.imshow(dataArr, cmap='gray')
    #plt.show()
    plt.savefig(fn)
    
    
    
def loadData(fn):
    print 'loadData', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    nRow = st.unpack('>I', fd.read(4))[0]
    nCol = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    print 'nRow', nRow
    print 'nCol', nCol
    
    # data: unsigned byte
    dataList = []
    for i in range(nData):
        dataRawList = fd.read(_N_PIXEL)
        dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
        dataArr = np.array(dataNumList).reshape(_N_ROW, _N_COL)
        dataList.append(dataArr.astype('int32'))
        
    fd.close()
    
    print 'done.'
    print
    
    return dataList
    


def loadLabel(fn):
    print 'loadLabel', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    
    # data: unsigned byte
    labelList = []
    for i in range(nData):
        dataLabel = st.unpack('B', fd.read(1))[0]
        labelList.append(dataLabel)
        
    fd.close()
    
    print 'done.'
    print
    
    return labelList



def loadMNIST():
    # 학습 데이터 / 레이블 로드
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    # 테스트 데이터 / 레이블 로드
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return trDataList, trLabelList, tsDataList, tsLabelList

    
    
if __name__ == '__main__':
    trDataList, trLabelList, inputs, labels = loadMNIST()
    errC = 0.0
    output = open('test_output.txt', 'w')
    best = open('best_param.pkl','r')
    param = pk.load(best)#best_param 불러오기
    ip0 = np.array(inputs).reshape(-1,784)#input값을 10000*784 matrix로 변환
    ip1 = np.ones((10000,1))
    ip  = np.hstack((ip1,ip0))
    label = np.array(labels).reshape(-1,1)#label = [5,0,4,....] type = ndarray
    onehotLabel = activation(np.dot(ip,param))
    
    print("test start\n")
    output.write("test start\n")
    
    
    for i in range(10000) : 
        if np.argmax(onehotLabel[i]) != label[i][0]: #onehot- representation 으로 나타낸 layers를 수치화해서 에러율을 구한다. 
            errC+=1.0
        output.write("%dth result = %d , label = %d\n"%(i,np.argmax(onehotLabel[i]),label[i][0]))
            
    print("test sample`s error rate = %f"%(errC/10000.0))
    
    
    if op.exists(_DST_PATH) == False:
        os.mkdir(_DST_PATH)
        
    
        
