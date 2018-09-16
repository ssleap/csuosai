#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random as ra
import sys
import os.path as op

'''
2015920040 임원경 4주차 과제
Genetic Algorithm
'''


######메인코드에서 command line argument 받기
#batch 파일에서 입력값을 받아서 출력
'''
def runExp(popSize, eliteNum, mutProb):
    print 'training...'
    trResFn = 'train_log_%d_%d_%.2f'%(popSize,eliteNum,mutProb)
    print('result file :',trResFn)
    print('testing...')
    tsResFn = 'test_output_%d_%d_%.2f'(popSize,eliteNum,mutProb)
    print('result file:',tsResFn)
'''  
#부포 파라미터 값으로부터 자식 파라미터값 추출  
#def nextPram(param,):

#파라미터 값에 파일을 읽어들여 오류여부 확인
def checkP(param,fSal,fSea):
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
    #파일에서 읽은 값을 각각 꼬리와 몸통 길이의 리스트로 나누어 저장
    for i in range(0,len(salL)) :
        salB.append(float(salL[i].split()[0]))
        salT.append(float(salL[i].split()[1]))

    for i in range(0,len(seaL)) :
        seaB.append(float(seaL[i].split()[0]))
        seaT.append(float(seaL[i].split()[1]))
    
    hme = []#파라미터 값마다 오류가 얼마인지 저장하는 리스트
    
    for i in range(0,len(param)):
        ecount = 0
        for j in range(0,50):
            if (param[i][0]*salB[j]+param[i][1]*salT[j]+param[i][2])<0:
                ecount +=1
              
            elif (param[i][0]*seaB[j]+param[i][1]*seaT[j]+param[i][2])>0:
                ecount+=1
        hme.append(ecount)
    return hme
    
    ###오류가 적은 개체를 찾는 함수. 인덱스를 담은 리스트를 돌려준다.
def findBest(elNum, errL):
    elidxs = []
    while elNum >= 0 :#elNum 은 eliteNum으로 시작해서 가장 작은 오류를 찾을 때마다 줄어가는 변수이다. 
        errC = 0# 오류율이 같은 에러를 카운트하는 변수
        for i in range(0, len(errL)):
            if errL[i] == min(errL) :
                elidxs.append(i)
                errC += 1
                errL[i] = 1000
        elNum = elNum - errC
    return elidxs

    
    
if __name__ == '__main__':# 4주차 pdf 70p 참조
    argmentNum = len(sys.argv)
    if argmentNum == 4:
        popSize = int(sys.argv[1])
        eliteNum = int(sys.argv[2])
        mutProb = float(sys.argv[3])
        
        #runExp(popSize,eliteNum, mutProb)
        ##train_log 파일과 test_output 파일 오픈
    trainLogString = 'train_log_%d_%d_%f.txt'%(popSize,eliteNum,mutProb)
    trainLog = open(trainLogString,'w')
    testOutString = 'test_output_%d_%d_%f.txt'%(popSize,eliteNum,mutProb)
    testOut = open(testOutString,'w')
    T = 0    
    param = []
    #초기 랜덤 파라미터 popsize 만큼 생성     
    for i in range(0,popSize) :
        temp = [ra.uniform(-20,20),ra.uniform(-20,20),ra.uniform(-200,200)]
        param.append(temp)
    #파라미터 별 오류 갯수를 hme 리스트에 저장한다. (how many error)  
###############################################################################loof 시작
    while T !=1000 :
        hme = checkP(param,'salmon_train.txt','seabass_train.txt')
        for id in range(0, len(hme)):
            print('%dth train Error Rate : %f\n'%(T,hme[id]/100.0))
            trainLog.write('%dth train Error Rate : %f\n'%(T,hme[id]/100.0))
        
        #우수 개체들을 저장한 리스트
        elidx = findBest(eliteNum,hme)
        #print(elidx)
        elParamx = []#x,y 의 계수와 상수를 저장하는 리스트
        elParamy = []
        elParamc = []
        elParam = []
        for i in elidx :
            elParam.append(param[i])#부모 개체중 우수한 개체를 뽑는다.
            elParamx.append(param[i][0])
            elParamy.append(param[i][1])
            elParamc.append(param[i][2])
        newParam = elParam#부모 개체중 우수한 개체는 다음세대에 유지한다.
        while len(newParam) == popSize:#다음 개체를 newParam에 만든다.
            if ra.random >= mutProb : #mutation 은 mutProb 보다 랜덤 인자가 크거나 같으면 돌연변이가 생기지 않는다.
                newParam.append([elParamx[ra.randrange(0,len(elParamx))],elParamy[ra.randrange(0,len(elParamy))],elParamc[ra.randrange(0,len(elParamc))]])
                #새로운 세대는 부모 세대 우수한 개체 파라미터 중에서 랜덤 추출 하여 이식한다. 예를들어 개체 A의 파라미터가 a1 a2 a3 이고 B가 b1 b2 b3 이면 A와 B의 파라미터들이 1/2확률로 다음 개체에 배정된다. 
                #ex) a1 b2 b3 or b1 b2 b3
            else :#만약 mutProb가 랜덤인자보다 크면 돌연변이가 생긴다. 돌연변이는 모든 파라미터를 최초 파라미터를 지정하는 방식으로 정한다. 
                newParam.append([ra.uniform(-20,20),ra.uniform(-20,20),ra.uniform(-200,200)])
        #newParam을 완성하면 새로운 세대가 기존 세대가 된다.
        param = newParam
        T+=1
    ######################################################loof 끝
    hme = checkP(param,'salmon_train.txt','seabass_train.txt')
    errorR = 0
    elidx = findBest(eliteNum,hme)
    #print(elidx)
    bestParam = [param[elidx[0]]]
    bestparam = bestParam[0]
    bestHme = checkP(bestParam,'salmon_test.txt','seabass_test.txt')
    fd1 = open('salmon_test.txt','r')
    fd2 = open('seabass_test.txt','r')
    
    print(bestparam)
    
    salL = fd1.readlines()
    seaL = fd2.readlines()

    #연어와 농어의 몸길이, 꼬리길이 데이터 B= 몸길이 T = 꼬리
    salB = []
    seaB = []

    salT = []
    seaT = []
    #파일에서 읽은 값을 각각 꼬리와 몸통 길이의 리스트로 나누어 저장 checkP 함수와 구조적으로 동일하다.
    for i in range(0,len(salL)) :
        salB.append(float(salL[i].split()[0]))
        salT.append(float(salL[i].split()[1]))

    for i in range(0,len(seaL)) :
        seaB.append(float(seaL[i].split()[0]))
        seaT.append(float(seaL[i].split()[1]))
        
   
    
    for i in range(0,50):
        print('parameter : %f %f %f ===>'%(bestparam[0],bestparam[1],bestparam[2]))
        testOut.write('parameter : %f %f %f ===>'%(bestparam[0],bestparam[1],bestparam[2]))
        if bestparam[0]*salB[i]+bestparam[1]*salT[i]+bestparam[2] >= 0:
            print('salmon correct\n')
            testOut.write('salmon correct\n')
        else :
            print('wrong\n')
            testOut.write('wrong\n')
            errorR +=1
            
    for i in range(0,50):
        print('parameter : %f %f %f ===>'%(bestparam[0],bestparam[1],bestparam[2]))
        testOut.write('parameter : %f %f %f ===>'%(bestparam[0],bestparam[1],bestparam[2]))
        if bestparam[0]*seaB[i]+bestparam[1]*seaT[i]+bestparam[2] < 0:
            print('seabass correct\n')
            testOut.write('seabass correct\n')
        else :
            print('wrong\n')
            testOut.write('wrong\n')
            errorR +=1
    
    print('error rate : %f'%float(errorR/100.0))
    testOut.write('error rate : %f'%float(errorR/100.0))

        


      
        






























