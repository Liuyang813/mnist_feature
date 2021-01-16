import pickle
import numpy as np

with open("feature2class2cnt","rb")as f:
    t=pickle.load(f)
print(len(t))

def get_std(content):
    lst=[]
    lst_=[]
    for i in range(0,10):
        name=str(i)
        num=content.get(name,0)
        lst.append(num)
        if num!=0:
            lst_.append(num)
        #print(lst_)
        #exit()
   
    try:
        std=np.std(normalization(lst_))
    except:
        std=0
    return lst_,std
    
    
def normalization(list):
    normlist =[]
    for k in range(0,len(list)):
            if max(list) - min(list) != 0:
                normlist.append((list[k] - min(list)) / (max(list) - min(list)))
    return normlist
    
   
j=0
good_feature={}
for key,content in sorted(t.items(),key=lambda x:len(x[1]),reverse=True):
     
    lst_,std=get_std(content)
    #lst_ = normalization(lst_)
    if std>=0.3:
        good_feature[key]={"content":content,"std":std}
    '''
    if j<100:
        print("======================")
        print(key,lst_,std)
    j+=1
    '''
print(len(good_feature))


with open("good_feature","wb")as f:
    pickle.dump(good_feature,f)