# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk#피클 모듈



# MNIST 데이터 경로
_SRC_PATH = u'C:\\Users\\IWG\\Desktop\\slp-20171209T093608Z-001\\slp\\mnist'#현재 데이터 경로로 바꿈
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


###########load_mnist.py
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
    

    
    # data: unsigned byte
    dataList = []
    for i in range(nData):
        dataRawList = fd.read(_N_PIXEL)
        dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
        dataArr = np.array(dataNumList).reshape(_N_ROW, _N_COL)
        dataList.append(dataArr.astype('int32'))
        
    fd.close()
    

    
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

 ###################SLP 클래스

class slp :
    def __init__(self,inputs):
        np.random.seed(12)
        self.layer = np.random.randn(785,10)# type = ndarray
    

        
    def training(self, inputs, labels, count, log, best):
        errC = 0.0 #에러 카운트
        lR = 0.0004#러닝레이트
        
        label = np.array(labels).reshape(-1,1)#label = [5,0,4,....] type = ndarray

        
        onehotLabel = np.zeros((60000,10))

        
        for i in range(60000) : #레이블을 one hot representation으로 표현
            a = label[i][0]
            onehotLabel[i][a] = 1
            

        ip0 = np.array(inputs).reshape(-1,784)#input값을 60000*784 matrix로 변환
        ip1 = np.ones((60000,1))
        ip  = np.hstack((ip1,ip0))
        series = 0
        err_p = 0
        while count>0 :

            layers = activation(np.dot(ip,self.layer))
            l_err = onehotLabel - layers

            l_delta = np.multiply(l_err, gPrime(layers))
            
            self.layer += lR*np.dot(ip.T, l_delta)
            #lR *= 0.99#학습이 되면 고칠 것
 
            for i in range(5000) : #에러율을 구한다. 처음 5000번 까지만 표본으로 뽑아 계산한다.
                if np.argmax(layers[i]) != label[i][0]: #onehot- representation 으로 나타낸 layers를 수치화해서 에러율을 구한다. 
                    errC+=1.0
                    
            if err_p <= errC :
                series += 1
                if series >= 50 : 
                    lR *= 0.9
                    series = 0
            
                    
            if errC/5000.0 <= 0.5:
                print("error Rate = %f\n"%(errC/5000.0))
                log.write("error Rate = %f\n"%(errC/5000.0))
                pk.dump(self.layer, best) #에러율이 일정 수치를 만족하면 피클화 시켜서 덤프한다. 
                break
                
            else :
                print("error Rate = %f\n"%(errC/5000.0))
                log.write("error Rate = %f\n"%(errC/5000.0))
                err_p = errC
                errC = 0
                
                
            
            
            
            
            
            
        
        
    
if __name__ == '__main__':
    trDataList, trLabelList, tsDataList, tsLabelList = loadMNIST()
    train_log = open('train_log.txt','w')
    best = open('best_param.pkl','w')#best_param생성
    
    
    
    if op.exists(_DST_PATH) == False:
        os.mkdir(_DST_PATH)
    print("training start\n")
    train_log.write("training start\n")
    per = slp(trDataList)
    per.training(trDataList, trLabelList, 1, train_log, best)
    
  
  