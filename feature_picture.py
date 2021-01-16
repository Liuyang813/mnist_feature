import numpy as np
import os
import sys
from PIL import Image as I
import pickle

def load_good_feature(file_name="good_feature"):
    t_ret={}
    with open(file_name,"rb")as f:
        q=pickle.load(f)
    for ll,key_class_cnt in enumerate(q.items()):
        key_class,cnt=key_class_cnt
        t_ret[key_class]=ll
    return t_ret

def do_one(img,origin_size):
    w,h = origin_size
    sample_point = []
    for m in range(w-1):
        for n in range(h-1):
            sample_point.append((m+1,n+1))    
    #截取3*3区域
    feature_list = []
    for data in sample_point:
        x,y = data
        box = (x-1,y-1,x+2,y+2)
        img_ = img.crop(box)
        i,j = (np.array(img_)).shape
        img_arr = list(np.array(img_))

        for index,img_f in enumerate(img_arr):
            for i,pix in enumerate(img_f):
                img_arr[index][i]=int(pix/15)
        pix_label = "_".join([str(x) for x in img_arr])
        feature_list.append(pix_label)
    #fea_id=[]
    fea_id = []
    
    for good_fea,idd in t_feature2id.items():
        fea_id.append(feature_list.count(good_fea))
        #print(good_fea)
        #print(feature_list)
        #print(feature_list.count(good_fea))
        #exit()
    return fea_id


def main(file_path):
    with open("all_data_good_feature_1","w",encoding="utf-8")as f:
        for file_name in os.listdir(file_path):
            print(file_name)
            ret={}
            for img_name in os.listdir(file_path+file_name):
                img_path = file_path +file_name +'\\'+img_name
                img = I.open(img_path) 
                img_size = np.array(img).shape
                feature_id = do_one(img,img_size)
                to_write="_".join([str(x)for x in feature_id])
                f.write("%s\t%s\n"%(img_path,to_write))

if __name__=="__main__":
    t_feature2id=load_good_feature(file_name="good_feature")
    file_path = sys.argv[1] + '\\'
    main(file_path)