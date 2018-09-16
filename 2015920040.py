######################
#file input and output
#distinguish fish
######################

#open intput_data.txt
inputF = open('input_data.txt','r')

#open output_data.txt
outputF = open('output_result.txt','w')

#list for input lines 
#read lines at input_data.txt
body = []
tail= []
lines= [] #input lines from input_data.txt
#result=[] #ouput lines from output_result.txt
for i in range(0,10):
    lines.append(inputF.readline())#read lines one by one
    body.append(int(lines[i].split()[0]))#change to integer
    tail.append(int(lines[i].split()[1]))
    print('body : %d   tail : %d'%(body[i],tail[i]))
    if body[i]/tail[i]>7 :
        print('bass\n')
        outputF.write('body : %d , tail : %d => bass\n'%(body[i],tail[i]))
    else :
        print('salmon\n')
        outputF.write('body : %d , tail : %d => salmon\n'%(body[i],tail[i]))
    
outputF.close()