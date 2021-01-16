import numpy as np
import os
import sys
from PIL import Image as I
import pickle

def do_one(img,origin_size):
    w,h = origin_size
    sample_point = []
    for m in range(w-1):
        for n in range(h-1):
            sample_point.append((m+1,n+1))    
    #截取3*3区域
    crop_img = []
    for data in sample_point:
        x,y = data
        box = (x-1,y-1,x+2,y+2)
        img_ = img.crop(box)
        i,j = (np.array(img_)).shape
        img_arr = list(np.array(img_))

        for index,img_f in enumerate(img_arr):
            for i,pix in enumerate(img_f):
                img_arr[index][i]=int(pix/20)
                #if pix <= 10 and pix >= 0:
                #    img_arr[index][i] = 0
                
                #elif pix > 10 and pix <= 100:
                #    img_arr[index][i] = 1

                #elif pix > 100 and pix <= 200:
                #    img_arr[index][i] = 2

                #else:
                #    img_arr[index][i] = 3
        #print(img_arr)
        pix_label = "_".join([str(x) for x in img_arr])
        crop_img.append(pix_label)
    return crop_img

def main(file_path):
    feature2class2cnt={}
    class_ret={}
    for file_name in os.listdir(file_path):
        print(file_name)
        ret={}
        for img_name in os.listdir(file_path+file_name):
            img_path = file_path +file_name +'\\'+img_name
            new_img_name = file_name+"_"+img_name
            img = I.open(img_path) 
            img_size = np.array(img).shape
            pix_label = do_one(img,img_size)
            for p in pix_label:
                if p not in feature2class2cnt:
                    feature2class2cnt[p]={}
                
                feature2class2cnt[p][file_name]=feature2class2cnt[p].get(file_name,0)+1
                ret[p]=ret.get(p,0) +1
        class_ret[file_name]=ret
    return class_ret,feature2class2cnt

if __name__=="__main__":
    file_path = sys.argv[1] + '\\'
    all_feature,feature2class2cnt=main(file_path)
    for key,value in feature2class2cnt.items():
        print(key,value)
    with open("all_feature","wb") as f:
        pickle.dump(all_feature,f)
    with open("feature2class2cnt","wb") as f:
        pickle.dump(feature2class2cnt,f)
        