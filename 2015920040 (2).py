#-*- coding: utf-8 -*-
#############################
"""
2015920040 컴퓨터과학부 임원경 2주차 과제
"""
#############################
#전반적으로 관련 부분 dfs_gui.py 파일 참조
from Tkinter import *

###그래프를 딕셔너리로 표현
dict = {}
dict[1] = [2,6]
dict[2] = [3,1]
dict[3] = [8,2]
dict[5] = [10]
dict[6] = [11,1]
dict[8] = [13,3]
dict[10] = [5,15]
dict[11] = [16,6]
dict[13] = [18,8]
dict[15] = [10,20]
dict[16] = [17,11,21]
dict[17] = [18,22,16]
dict[18] = [19,13,23,17]
dict[19] = [20,18]
dict[20] = [15,25]
dict[21] = [22,16]
dict[22] = [17,21,23]
dict[23] = [22,18]
dict[25] = [20]



#H(n)을 얻어내는 함수
#H(n)은 현재위치와 15칸의 상하좌우 칸수 차이로 결정한다. 공백은 고려하지 않는다.
def getHn(location) :
    if location <= 5 :
        return 5-location+2
    elif location >5 and location<=10 :
        return 10-location+1
    elif location >10 and location<=15 :
        return 15-location
    elif location >15 and location <=20 : 
        return 20-location+1
    else :
        return 25-location+2

class App:
    def __init__(self, master):
    
    
        self.canvas = Canvas(master, width = 800, height = 600)
        self.canvas.pack()
        
        # 버튼 생성
        button = Button(master, text = 'run', command = self.run)
        button.pack()
        
        
        
      
        for row in range(0,5) :
            for col in range(0,5) :
                if row*5+col+1 == 4 or row*5+col+1 ==7 or row*5+col+1 == 9 or row*5+col+1 == 12 or row*5+col+1 == 14 or row*5+col+1 ==24 :
                    fillColor = 'black'
                else :
                    fillColor = 'white'
                
                self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = fillColor, outline = 'blue')
                
                
                self.canvas.create_text(col * 100 + 50, row * 100 + 50, text = int(col+1 + row*5))
                
                
        ###### A* 초기화
        
        self.loc =int(11)
        self.canvas.create_rectangle(0 * 100, 2 * 100, 0 * 100 + 100, 2 * 100 + 100, fill = 'red', outline = 'blue')
        self.gn = 0
        self.visited = []
        print('%d : \nf(n) = %d\ng(n) = %d\nh(n) = %d\n'%(11,4,0,4))
        
        
       
        
        

    def run(self) :
        
        while self.loc != 15 :
            minL = {}#위치 i를 받으면 fn을 주는 딕셔너리
            fnL = []#fn만 저장하는 리스트
            iL = []#경로를 저장하는 리스트
            
            #각 노드의 f(n)들을 구하는 for문
            for i in dict[self.loc] :
                #locL = [i,getHn(i)+self.gn]
                #minL.append(locL)
                iL.append(i)
                minL[i] = getHn(i)+self.gn
                fnL.append(getHn(i)+self.gn)
            minF = min(fnL)#minL 의 fn중에서 최솟값을 표기하는 변수
            idx = 0 #아래의 for 문에서 iL과 fnL의 인덱스
            for i in dict[self.loc] :
                if minL[i] == minF :
                    print('%d : \nf(n) = %d\ng(n) = %d\nh(n) = %d\n'%(i,fnL[idx]+1,self.gn+1,getHn(i)))
                    rows = int(i/5)
                    cols = 0
                    if i%5 == 0 :
                        cols = 4
                        rows-=1
                    else :
                        cols = i%5-1
                    self.canvas.create_rectangle(cols * 100, rows * 100, cols * 100 + 100, rows * 100 + 100, fill = 'red', outline = 'blue')
                    self.loc = i
                    break
                idx+=1  
            self.gn = self.gn +1
            
        
        


        
        # for i in self.visited :
            # int(i/5) = int(i/5)
            # (i%5)-1 = i%5-1
            # self.canvas.create_rectangle((i%5)-1 * 100, int(i/5) * 100, (i%5)-1 * 100 + 100, int(i/5) * 100 + 100, fill = 'red', outline = 'blue')
            
            
            
            
            
#main
root = Tk()
app = App(root)
root.mainloop()




