# -*- coding: utf-8 -*-

import numpy as np
import random as ra

#####activation function g(z)
def activation(x):
    return 1/(1+np.exp(-x))
    #activation function의 미분
def gPrime(a):
    
    return a*(1-a)
    
    #multi layer perceptron class
class mlp:
    def __init__(self, inputs):
        self.inputs = inputs
        #인풋값 4개
        self.l = len(self.inputs)
        #인풋의 길이 2
        self.li = len(self.inputs[0])
        
        #파라미터 벡터값 행렬, 2*4 matrix
        self.hiddenLr = np.random.random((self.li,self.l))
        
        #벡터값 4*1 matrix
        self.opLr = np.random.random((self.l,1))
        
        
    def trans (self, inp):
        #g(wh)
        sig1 = activation(np.dot(inp, self.hiddenLr))
        sig2 = activation(np.dot(sig1,self.opLr))
    
        return sig2
            
        # 훈련시키는 메소드
    def trainI(self, inputs, labels, count, log):
        i = 1
        while count>0:
            ip = inputs
            l1 = activation(np.dot(ip, self.hiddenLr))
            l2 = activation(np.dot(l1,self.opLr))
            
            #back propagation
            l2_err = labels - l2#d-o
            
            l2_delta = np.multiply(l2_err, gPrime(l2))#(d-o)o(1-o)
            
            l1_err = np.dot(l2_delta, self.opLr.T)#(d-o)o(1-o).opLr
            l1_delta = np.multiply(l1_err, gPrime(l1))#((d-o)o(1-o).opLr)h(1-h)
            
            
            
            self.opLr+=np.dot(l1.T,l2_delta)
            self.hiddenLr+=np.dot(ip.T, l1_delta)            
            err_array = np.array([[0.5],[0.5],[0.5],[0.5]])
            arr_zero = np.zeros((4,1))
            err_count =0.0
            for j in range(4) :
                if self.trans(ip)[j][0] < 0.1 :
                    err_array[j][0] = 0
                elif self.trans(ip)[j][0] >0.9 :
                    err_array[j][0] = 1
                else :
                    err_count += 1.0
            
            
                
            i += 1
            print("%dth traing error rate : %f\n"%(i,(err_count/4.0)))
            log.write("%dth traing error rate : %f\n"%(i,(err_count/4.0)))
            
            if (err_array - labels).any() == arr_zero.any() :
                fd = open("test_output.txt",'w')
                print("hidden parameters >>>> ")
                fd.write("hidden parameters >>>> ")
                print(self.hiddenLr)
                fd.write(self.hiddenLr)
                print("\noutput param >>>>")
                fd.write("\noutput param >>>>")
                print(self.opLr)
                fd.write(self.opLr)
                print("\ntest error rate : %f\n"%((err_count/4.0)))
                fd.write("\ntest error rate : %f\n"%((err_count/4.0)))
                break
                
        
            


if __name__ == '__main__':
    
        
    train_xor = open('train_xor.txt','r')
    train_log = open('train_log.txt','w')
        
    trnL = train_xor.readlines()
        
    #train_xor.txt에서 읽은 값 리스트에 저장 array 변환 각각 [[1,x1,y1],[1,x2,y2]...]형태로
    #1은 계산을 위한 상수
    inputs = np.array([[1, float(trnL[0].split()[0]),float(trnL[0].split()[1])],[1, float(trnL[1].split()[0]),float(trnL[1].split()[1])], [1,float(trnL[2].split()[0]),float(trnL[2].split()[1])], [1,float(trnL[3].split()[0]),float(trnL[3].split()[1])]])
    #train_xor.txt에서 읽은 값 중 레이블 값 저장 리스트로 array변환
    #labels = [0,1,1,0] column
    labels = np.array([[float(trnL[0].split()[2])],[float(trnL[1].split()[2])],[float(trnL[2].split()[2])],[float(trnL[3].split()[2])]])

    per = mlp(inputs)
    print("Training Start : \n")
    train_log.write("Training Start : \n")
    print(per.trans(inputs))
    train_log.write(per.trans(inputs))
    per.trainI(inputs, labels, 1, train_log)
    
    
    
    ##np.hstack([a1,a2])