# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:44:31 2019

@author: hasee
"""


from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import FrankWolfeSSVM, OneSlackSSVM
from sklearn.metrics import precision_score, recall_score, f1_score
#from sklearn.externals import joblib
from threading import Thread
from queue import Queue
import pandas as pd
import numpy as np
import random
import time
import pickle
import os 
import gc


'''
#学习器learner选择问题：
#    当数据集 n_samples >= 5000 时, :FrankWolfeSSVM效果更加好
#    当数据集 n_samples < 5000 时, :OneSlackSSVM效果更加好
'''

K_FOLD = 10

DELTA = 0.5
K = 100
ALPHA = 2.0

CAMPLEN = [2]#range(10, K+1) #选择不同长度训练分类器及预测对应长度 range(10, K+1)表示MARS的长度 >= 10 放在一起训练
FAKE_CAMP = 2#CAMPLEN[0]*0.2
NODE_FEATURES = [0,1,2,4,5,6,7]  #0:RD, 1:RANK, 2:EXT, 3:CAMPAIGN_SIZE, 4:WORDSCOUNT, 5:PP1, 6:DAYS, 当前评论时间与第一条评论的时间间隔差, 7:BURST
EDGE_FEATURES = [0,1,2,3]     #0:RATE_DIFF, 1:DATE_DIFF, 2:TEXT_JACCARD_SIMILARITY, 3:FRT,first_review_time
CLASS_WEIGHT = [0.1, 0.9]   #类别权重 DELTA = 0.0 =>  #NYC [0.2, 0.8] if len >= 8 else [0.1, 0.9]     #Zip [0.2, 0.8] if len >= 8 else [0.1, 0.9] 

LEARNER = "OneSlackSSVM"    #FrankWolfeSSVM OneSlackSSVM

DATASET = "NYC"     #NYC, Zip

if DATASET == "NYC":
    PID_LIST = [p for p in range(0, 923)]   #NYC
else:
    PID_LIST = [p for p in range(0, 5044)]   #Zip

               
GSFG_HOME = "SourceData/Yelp{}/".format(DATASET) #元数据集

RESULT_FILE = "ResultData/{}_d{}_K{}_A{}/".format(DATASET, DELTA, K, ALPHA) #保存结果数据集

FILE_META = "metadata"
FILE_REVIEWCONTENT = "reviewContent"

def loadReview():
    # 读取评论数据
    file_meta = GSFG_HOME + FILE_META
    fp1 = open(file_meta)
    for lines in fp1:
        lines = lines.replace("\n", "")
        lines = lines.split('\t')

        globalLabel.append([0, int(lines[1]), 1 if lines[3] == "-1" else 0, 0]) #pred pid label istrain
    fp1.close()

def loadPickle(pid, campaignLength):
    '''
    载入当前pid长度为campaignLength用于CRF训练所需数据包括((features, edges, edge_features), originalIndexs)
    originalIndexs原始数据索引list和globalReviews的索引一一对应
    input:
        pid: 为所需要数据的产品编号
        campaignLength: 当前pid下的群组数据,其长度为campaignLength的所有群组数据
    output:
        campaignsList -> [((features, edges, edge_features), originalIndexs), ...]
    '''
    
    pidFileDirectly = RESULT_FILE+str(pid)#+"_1"  #读取当前pid子目录文件夹
    campDataFile = pidFileDirectly + '/PrepareData_length{}.pkl'.format(campaignLength) #读取当前pid文件夹下的campaignLength长度的所有群组文件
    if not os.path.exists(pidFileDirectly):
#        print("当前pid:{}文件不存在".format(pid), end=",")
        return None
    elif not os.path.exists(campDataFile):
#        print("当前pid:{}不存在长度为{}的群组".format(pid, campaignLength))
        return None
    else:
        #存在该文件, 调用pickle, 读取该文件, 返回当前pid下的群组数据,其长度为campaignLength的所有群组数据 ->data
        with open(campDataFile, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            return data

def loadFeature(campdata):
    """
    读取特征
    """
    featureData = list()
    label = list()
    campvid = list()
    
    for camp in campdata:
        feature, oidList = camp
        label.append(np.array([globalLabel[i][2] for i in oidList]))
        node_feature = feature[0][:, NODE_FEATURES]
        edge = feature[1]
        feature[2][:, 2] = np.nan_to_num(feature[2][:, 2])  #解决cosine文本相似度出现nan值
        edge_feature = feature[2][:, EDGE_FEATURES]
        featureData.append((node_feature, edge, edge_feature))

        campvid.append(oidList)
        
    return featureData, label, campvid

def loadPIDFeature():
    """
    载入所有产品的特征
    """
    for p in PID_LIST:
        for camplen in CAMPLEN:
            campdata = loadPickle(p, camplen)
            if campdata:
                feature, label, vidList = loadFeature(campdata)
                X_data.extend(feature)
                Y_data.extend(label)
                Y_camp.extend(vidList)

def statistic_result(y_pred, y_test, camptest):
    """
    统计评论结果
    """
    df_data = pd.DataFrame([], columns = ["oid", "pred", "label"])
    camp_arr = np.hstack(camptest)
    pred_arr = np.hstack(y_pred)
    label_arr = np.hstack(y_test)
    df_data["oid"] = camp_arr
    df_data["pred"] = pred_arr
    df_data["label"] = label_arr
    
    df_result = df_data.groupby(by="oid").max().reset_index()
    
    del pred_arr, camp_arr, label_arr, df_data
    
    return df_result

def statistic_campaign_result(y_pred, y_test):
    """
    统计群组结果
    """
    camps_pred = list()
    camps_lbl = list()
    for ypds, lbls in zip(y_pred, y_test):
        camps_pred.append(1 if ypds.sum() >= FAKE_CAMP else 0)
        camps_lbl.append(1 if lbls.sum() >= FAKE_CAMP else 0)
        
    return camps_pred, camps_lbl     

def model_test(k, head, tail):
    """
    CRF训练和预测
    """
    each_fold_time = time.time() #开始计时
	
    #divide train set and test set
    train_id = dataId[head : tail]
    test_id = dataId[:head] + dataId[tail:]
    
    X_train = X_arr[train_id, :]
    Y_train = Y_arr[train_id]
    X_test = X_arr[test_id, :]
    Y_test = Y_arr[test_id]
    campTest = Camp_arr[test_id]
    #ends divide train set and test set
    if len(X_train) > 0:
        #实例化CRF 
        EFGCRF = EdgeFeatureGraphCRF(inference_method='qpbo', class_weight=CLASS_WEIGHT)
        if LEARNER == "OneSlackSSVM":
            #利用OneSlackSSVM训练模型参数
            ssvm = OneSlackSSVM(EFGCRF, C=.1, tol=.1, max_iter=100, switch_to='ad3')
        elif LEARNER == "FrankWolfeSSVM":
			#利用FrankWolfeSSVM训练模型参数
            ssvm = FrankWolfeSSVM(EFGCRF, C=.1, tol=.1, max_iter=100)
        else:
            #没有选择分类器退出
            pass

        ssvm.fit(X_train, Y_train)
        Y_pred = ssvm.predict(X_test)
        
        df_result = statistic_result(Y_pred, Y_test, campTest)
        V_precision = precision_score(df_result["label"], df_result["pred"])
        V_recall = recall_score(df_result["label"], df_result["pred"])
        V_f1 = f1_score(df_result["label"], df_result["pred"])
        
        camps_pred, camps_lbl = statistic_campaign_result(Y_pred, Y_test)
        C_precision = precision_score(camps_lbl, camps_pred)
        C_recall = recall_score(camps_lbl, camps_pred)
        C_f1 = f1_score(camps_lbl, camps_pred)
        
        result_Queue.put([V_precision, V_recall, V_f1, C_precision, C_recall, C_f1])
        
    else:
        print("TRAIN SET is NULL")
        
    print("the {}th fold using time: {:.4f} min".format(k+1, (time.time() - each_fold_time) / 60) )
    del X_train, Y_train, X_test, Y_test, Y_pred, campTest
    
    
if __name__ == "__main__":  # 主函数
    
    print("DELTA:{}, K:{}, ALPHA:{}".format(DELTA, K, ALPHA))
    print("DATASET:{}  LEARNER:{}".format(DATASET, LEARNER))
    print("CALSS_WEIGHT:", CLASS_WEIGHT)
    print("FEATURE", NODE_FEATURES, EDGE_FEATURES)
    globalLabel = list()  #pred pid label istrain 预测值, PID, 标签值, 是否为训练集样本0:不是, 1:是
    readReviewTime = time.time()
    loadReview()        #载入评论
    print("load reviews time: {:.4f} min".format((time.time()-readReviewTime)/60))
    
    X_data = list() #特征集合
    Y_data = list() #标签集合
    Y_camp = list() #id集合
    loadPIDFeature()    #载入特征
    print("load feature time: {:.4f} min".format((time.time()-readReviewTime)/60))
    
    X_arr, Y_arr, Camp_arr = np.array(X_data), np.array(Y_data), np.array(Y_camp)

    campaigns_num = len(X_data)     #群组数量
    Vset = set(np.hstack(Y_camp))   #评论id集合
    Vs_num = len(Vset)    #评论数量
    
    fake_Vs_num = 0     #作弊评论数量
    for vid in Vset:
        fake_Vs_num += globalLabel[vid][2]
    
    fake_campaigns_num = 0  #作弊群组个数
    for d in Y_data:    
        if sum(d) >= FAKE_CAMP: fake_campaigns_num += 1
              
    del X_data, Y_data, Y_camp, globalLabel
    
    lenData = len(X_arr)
    dataId = [i for i in range(lenData)]    #数据集编号
    #混乱数据编号
    random.shuffle(dataId)
    
    #K_FOLD_CROSS_VALIDATION
    each_fold_size = (lenData + 1) / K_FOLD
    
    #用于存放每个FOLD的结果
    result_Queue = Queue()  #[V_precision, V_recall, V_f1, C_precision, C_recall, f1]
    
    threadList = list() #多线程操作, 存放所有线程的线程池
    for k in range(K_FOLD):     #each fold
#        print("the {}th fold =====".format(k+1))
        head, tail = int(k * each_fold_size), int((k + 1) * each_fold_size) 
        thr = Thread(target=model_test, args=(k, head, tail))   #将每FOLD加入线程
        threadList.append(thr)  #将该线程加入到线程池
        
        thr.start()     #开启当前线程
        
    for t in threadList:    #阻塞主线程, 当所有子线程全部运行完毕才继续运行
        t.join()  
        
    all_result = np.zeros((K_FOLD,6), dtype="float64")
    i = 0
    while not result_Queue.empty():     #读取结果
        all_result[i,:] = result_Queue.get()
        i += 1
    
    avg_V_p, avg_V_r, avg_V_f = np.mean(all_result[:, 0]), np.mean(all_result[:, 1]), np.mean(all_result[:, 2])
    avg_C_p, avg_C_r, avg_C_f = np.mean(all_result[:, 3]), np.mean(all_result[:, 4]), np.mean(all_result[:, 5])
    print("campaign_length: ", CAMPLEN[0])
    print("{}_fold_cross_validation result: ".format(K_FOLD))
    print("reviews: ")
    print("avg_precision||avg_recall||avg_f1_score||Vs_num||fake_Vs_num")
    print("%.4f" % avg_V_p, "%.4f" % avg_V_r, "%.4f" % avg_V_f, Vs_num, fake_Vs_num, sep="\t||")
    print("std_precision||std_recall||std_f1_score")
    print("%.4f" % np.std(all_result[:, 0]), "%.4f" % np.std(all_result[:, 1]), "%.4f" % np.std(all_result[:, 2]), sep="\t||")
    print("campaigns: ")
    print("avg_precision||avg_recall||avg_f1_score||campaigns_num||fake_campaigns_num")
    print("%.4f" % avg_C_p, "%.4f" % avg_C_r, "%.4f" % avg_C_f, campaigns_num, fake_campaigns_num, sep="\t||")
    print("std_precision||std_recall||std_f1_score")
    print("%.4f" % np.std(all_result[:, 3]), "%.4f" % np.std(all_result[:, 4]), "%.4f" % np.std(all_result[:, 5]), sep="\t||")
    print("   " + str(CAMPLEN[0]) if CAMPLEN[0] != 10 else ">=10", "%.4f" % avg_V_p, "%.4f" % avg_V_r, "%.4f" % avg_V_f, "%.4f" % np.std(all_result[:, 0]), "%.4f" % np.std(all_result[:, 1]), "%.4f" % np.std(all_result[:, 2]), sep=",")
    print("   " + str(CAMPLEN[0]) if CAMPLEN[0] != 10 else ">=10", "%.4f" % avg_C_p, "%.4f" % avg_C_r, "%.4f" % avg_C_f, "%.4f" % np.std(all_result[:, 3]), "%.4f" % np.std(all_result[:, 4]), "%.4f" % np.std(all_result[:, 5]), sep=",")
    
	#保存模型使用joblib函数即可
	
    del all_result
    gc.collect()
    print("total running time: {:.4f} min".format((time.time()-readReviewTime)/60))
