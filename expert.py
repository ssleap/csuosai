#-*- coding: utf-8 -*-
import random as ra
import numpy as np
import matplotlib.pyplot as plt

'''
2015920040 임원경 3주차 과제
smulated anealing
'''
#데이터 초기화
param = [2.0,-1.0,-180.0]
T = 100.0 # 초기 온도 T
lowE=0.0
lowparam=0.0
#train file open
salF = open('salmon_train.txt','r')
seaF = open('seabass_train.txt','r')

#train file 의 데이터 읽어오기
salL = salF.readlines()
seaL = seaF.readlines()

#연어와 농어의 몸길이, 꼬리길이 데이터 B= 몸길이 T = 꼬리
salB = []
seaB = []

salT = []
seaT = []

for i in range(0,len(salL)) :
    salB.append(float(salL[i].split()[0]))
    salT.append(float(salL[i].split()[1]))

for i in range(0,len(seaL)) :
    seaB.append(float(seaL[i].split()[0]))
    seaT.append(float(seaL[i].split()[1]))

#초기오류 E 계산
#train 파일에서 에러가 난 연어, 농어의 개수
eSal=0.0
eSea=0.0
p=0
for i in range(0,len(salL)) :
    if (param[0]*salB[i]+param[1]*salT[i]+param[2])<=0 :
        eSal+=1
    elif (param[0]*seaB[i]+param[1]*seaT[i]+param[2])>0 :
        eSea+=1

#초기에러 E        
E = float( (eSal + eSea)/100.0)
print('%d-th training, T = %f, E = %f\n'%(0,T,E))
fd = open('train_log.txt','w')
fd.write('%d-th training, T = %f, E = %f\n'%(0,T,E))
   ####train 파일을 가지고 학습 
while T >0.000001 :
    newParam = [param[0]+ ra.uniform(-0.01,0.01),param[1] + ra.uniform(-0.01,0.01),param[2] + ra.uniform(-10,10)]
    
    #오류 E 계산
    #train 파일에서 에러가 난 연어, 농어의 개수
    eSalmon=0.0
    eSeabass=0.0
    p+=1
    #train 하나하나 확인
    for i in range(0,len(salL)) :
        if (newParam[0]*salB[i]+newParam[1]*salT[i]+newParam[2])<=0 :
            eSalmon+=1
        elif (newParam[0]*seaB[i]+newParam[1]*seaT[i]+newParam[2])>0 :
            eSeabass+=1

    #새로운 에러 E        
    newE = float( (eSalmon + eSeabass)/100.0)
    print('%d-th training, T = %f, E = %f\n'%(p,T,newE))
    fd.write('%d-th training, T = %f, E = %f\n'%(p,T,newE))
    if newE - E <=0 :
        lowparam = newParam
        lowE = newE
        param = newParam
        E = newE
        
    else :
        r = ra.random()
        if r < np.exp(-(newE-E)/T):
            param = newParam
            E = newE
            
    T = 0.99*T
E = lowE
param = lowparam
    #최선의 파라미터와 오류율 출력
print('best parameter : %fx + %fy + %f'%(param[0],param[1],param[2]))
fd.write('best parameter : %fx + %fy + %f'%(param[0],param[1],param[2]))
print('\n best error rate : %f'%E)
fd.write('\n best error rate : %f'%E)


#test 파일을 읽고 선형분류기로 분류
output = open('test_output.txt','w')#분류 결과 출력할 파일    
st = open('salmon_test.txt','r')#test 파일 
seat = open('seabass_test.txt','r')

salmon = st.readlines()
seabass = seat.readlines()
#test파일의 연어의 몸길이 :B 꼬리 : T 
salmonB = []
salmonT = []
seabassB = []
seabassT = []
sale = 0.0 # error
seae = 0.0

#plot 
plt.figure()

#test파일을 분류(연어)
for i in range(0,len(salmon)):
    salmonB.append(float(salmon[i].split()[0]))
    salmonT.append(float(salmon[i].split()[1]))
    
for i in range(0,len(seabass)):
    seabassB.append(float(seabass[i].split()[0]))
    seabassT.append(float(seabass[i].split()[1]))
    
for i in range(0,len(salmon)):

    
    print('body : %f    tail : %f    (salmon) ==>'%(salmonB[i],salmonT[i]))
    output.write('body : %f    tail : %f    (salmon) ==>'%(salmonB[i],salmonT[i]))
    if (param[0]*salmonB[i] + param[1]*salmonT[i] + param[2])>=0 :
        print('salmon  (correct)\n')
        output.write('salmon  (correct)\n')
        #연어 마커 ㅇ 맞으면 초록색 틀리면 빨간색
        plt.plot(salmonB[i],salmonT[i], marker = 'o',markerfacecolor = 'green')
    else :
        print('seabass   (error)\n')
        output.write('seabass    error\n')
        sale +=1
        plt.plot(salmonB[i],salmonT[i], marker = 'o',markerfacecolor = 'red')


    #test 파일을 분류 (농어)
for i in range(0,len(seabass)):

    
    print('body : %f    tail : %f    (seabass) ==>'%(seabassB[i],seabassT[i]))
    output.write('body : %f    tail : %f    (seabass) ==>'%(seabassB[i],seabassT[i]))
    if (param[0]*seabassB[i] + param[1]*seabassT[i] + param[2])<0 :
        print('seabass  (correct)\n')
        output.write('seabass  (correct)\n')
        plt.plot(seabassB[i],seabassT[i], marker = 's',markerfacecolor = 'green')
    else :
        print('salmon   (error)\n')
        output.write('salmon    error\n')
        plt.plot(seabassB[i],seabassT[i], marker = 's',markerfacecolor = 'red')
        seae +=1    
        
print('==============test result============')
print('error rate = %f%%'%(seae+sale))
output.write('==============test result============')
output.write('error rate = %f%%'%(seae+sale))
    

   ### plot 출력 
plt.show()
    
    
    
    
    
