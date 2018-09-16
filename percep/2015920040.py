#-*- coding: utf-8 -*-
import numpy as np
import random as ra
import sys
import os.path as op

'''
2015920040 컴퓨터과학부 임원경 5주차 perceptron 과제
'''

def gx(wt,feaVec):
   
    if wt[0]*feaVec[0]+wt[1]*feaVec[1]+feaVec[2] >=0:
        return 1
    else :
        return 0
        
def test(wt, fSal, fSea,testOut) :
    #train file open
    salF = open(fSal,'r')
    seaF = open(fSea,'r')
    

    #train file 의 데이터 읽어오기
    salL = salF.readlines()
    seaL = seaF.readlines()

   
    #연어와 농어의 몸길이, 꼬리길이 데이터 B= 몸길이 T = 꼬리
    salB = []
    seaB = []

    salT = []
    seaT = []
    
    #학습 데이터 세트 리스트 [[특징벡터, 정답 레이블]]
    salX=[]
    seaX=[]
    #파일에서 읽은 값을 각각 꼬리와 몸통 길이의 리스트로 나누어 저장
    for i in range(0,len(salL)) :
        salB.append(float(salL[i].split()[0]))
        salT.append(float(salL[i].split()[1]))

    for i in range(0,len(seaL)) :
        seaB.append(float(seaL[i].split()[0]))
        seaT.append(float(seaL[i].split()[1]))
        
    for i in range(0,len(salB)):
        salX.append([salB[i],salT[i],1,1])
        seaX.append([seaB[i],seaT[i],1,0])    
        
    for i in range(0,50):
        if gx(wt,salX[i])==1 :
            print('body : %f, tail : %f (salmon) ==> salmon\n'%(salB[i],salT[i]))
            testOut.write('body : %f, tail : %f (salmon) ==> salmon\n'%(salB[i],salT[i]))
        else :
            print('body : %f, tail : %f (salmon) ==> seabass (wrong)\n'%(salB[i],salT[i]))
            testOut.write('body : %f, tail : %f (salmon) ==> seabass(wrong)\n'%(salB[i],salT[i]))
        if gx(wt,seaX[i])==0 :
            print('body : %f, tail : %f (seabass) ==> seabass\n'%(seaB[i],seaT[i]))
            testOut.write('body : %f, tail : %f (seabss) ==> seabss\n'%(seaB[i],seaT[i]))
        else :
            print('body : %f, tail : %f (seabass) ==> salmon (wrong)\n'%(seaB[i],seaT[i]))
            testOut.write('body : %f, tail : %f (seabass) ==> salmon (wrong)\n'%(seaB[i],seaT[i]))
            
def train(param,fSal,fSea,lR):
    #train file open
    salF = open(fSal,'r')
    seaF = open(fSea,'r')
    

    #train file 의 데이터 읽어오기
    salL = salF.readlines()
    seaL = seaF.readlines()

    wt = param
    t = 0
    #연어와 농어의 몸길이, 꼬리길이 데이터 B= 몸길이 T = 꼬리
    salB = []
    seaB = []

    salT = []
    seaT = []
    
    #학습 데이터 세트 리스트 [[특징벡터, 정답 레이블]]
    salX=[]
    seaX=[]
    #파일에서 읽은 값을 각각 꼬리와 몸통 길이의 리스트로 나누어 저장
    for i in range(0,len(salL)) :
        salB.append(float(salL[i].split()[0]))
        salT.append(float(salL[i].split()[1]))

    for i in range(0,len(seaL)) :
        seaB.append(float(seaL[i].split()[0]))
        seaT.append(float(seaL[i].split()[1]))
        #특징벡터 x
    for i in range(0,len(salB)):
        salX.append([salB[i],salT[i],1,1])
        seaX.append([seaB[i],seaT[i],1,0])
     
     #w(t+1) = w(t) + lR(dk - ok)xki
    for i in range(0,50) :
        k = float(gx(wt,salX[i]))
        wt[0] = wt[0] + lR*(salX[i][3]-k)*salX[i][0]
        wt[1] = wt[1] + lR*(salX[i][3]-k)*salX[i][1]
        wt[2] = wt[2] + lR*(salX[i][3]-k)*salX[i][2]
        wt[0] = wt[0] + lR*(seaX[i][3]-k)*seaX[i][0]
        wt[1] = wt[1] + lR*(seaX[i][3]-k)*seaX[i][1]
        wt[2] = wt[2] + lR*(seaX[i][3]-k)*seaX[i][2]
        
    return wt
    
def errorCounter (wt,fSal,fSea):
     #train file open
    salF = open(fSal,'r')
    seaF = open(fSea,'r')
    

    #train file 의 데이터 읽어오기
    salL = salF.readlines()
    seaL = seaF.readlines()

    
    #연어와 농어의 몸길이, 꼬리길이 데이터 B= 몸길이 T = 꼬리
    salB = []
    seaB = []

    salT = []
    seaT = []
    
    salX=[]
    seaX=[]
    
    errC=0

    #파일에서 읽은 값을 각각 꼬리와 몸통 길이의 리스트로 나누어 저장
    for i in range(0,len(salL)) :
        salB.append(float(salL[i].split()[0]))
        salT.append(float(salL[i].split()[1]))

    for i in range(0,len(seaL)) :
        seaB.append(float(seaL[i].split()[0]))
        seaT.append(float(seaL[i].split()[1]))
        #특징벡터 x
    for i in range(0,len(salB)):
        salX.append([salB[i],salT[i],1,1])
        seaX.append([seaB[i],seaT[i],1,0])
        
    for i in range(0,50) :
        if gx(wt,salX[i])==0 : 
            errC +=1
        if gx(wt, seaX[i])!=0 :
            errC +=1
    return float(errC/100.0)
    
    
    

###################################################main     
if __name__ == '__main__':
    argmentNum = len(sys.argv)
    if argmentNum == 2:
        learningRate = float(sys.argv[1])#배치파일 learning rate 를 comand line argument 로 받는다.
        
    trainLogString = 'train_log_%f.txt'%(learningRate)#train_long와 test_output 을 쓰기 파일로 연다
    trainLog = open(trainLogString,'w')
    testOutString = 'test_output_%f.txt'%(learningRate)
    testOut = open(testOutString,'w')
    
    t = 0 #학습반복 횟수 0
    #초기 파라미터값 랜덤 생성
    wt = [ra.uniform(-10,10),ra.uniform(-10,10),ra.uniform(-200,200)]
    
    fSal = 'salmon_train.txt'
    fSea = 'seabass_train.txt'
    
    ######################################################loof 시작
    while True :
        wt = train(wt,fSal,fSea,learningRate)
        err = errorCounter(wt,fSal,fSea)
        print('%dth train : wt ==> %f, %f, %f error rate : %f\n'%(t,wt[0],wt[1],wt[2],err))
        trainLog.write('%dth train : wt ==> %f, %f, %f error rate : %f\n'%(t,wt[0],wt[1],wt[2],err))
        t+=1
        if err <= 0.15 :
            break
    
    print('last parameter : wt ==> %f, %f, %f error rate : %f\n'%(wt[0],wt[1],wt[2],err))
    trainLog.write('last parameter : wt ==> %f, %f, %f error rate : %f\n'%(wt[0],wt[1],wt[2],err))
    test(wt,'salmon_test.txt','seabass_test.txt',testOut)
    errR = errorCounter(wt,'salmon_test.txt','seabass_test.txt')
    print('final error rate : %f\n'%errR)
    testOut.write('final error rate : %f\n'%errR)
    
    
    
    
    
    
    
    
    
    
    
    
    
    