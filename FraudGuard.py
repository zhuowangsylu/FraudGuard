# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:47:18 2019

@author: hasee
"""

from sklearn.externals import joblib
import pickle
import time
import numpy as np
import reviewContent_process
import operator
import datetime
import math
import os
import gc

MAXSIMSUM = 0.5 #δ
K = 100
ALPHA = 2.0

OUT_CAMPAIGN_FEATURE = True
OUT_CAMPAIGN = True
LOAD_EFGCRF_PREDICT = False

#默认参数, 用于确定crf中图结构的边生成所需的条件, 根据不同数据集可能会有所更改, 两个Yelp数据集统一使用
sim_first_review_date_diff = 10
sim_content_length_diff = 0.6
sim_day_diff = 5
edge_exist_threshold = 0

NODE_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7]  #0:RD, 1:RANK, 2:EXT, 3:CAMPAIGN_SIZE, 4:WORDSCOUNT, 5:PP1, 6:DAYS, 7: BURST
EDGE_FEATURES = [0, 1, 2, 3]     #0:RATE_DIFF, 1:DATE_DIFF, 2:TEXT_JACCARD_SIMILARITY, 3:FRT,first_review_time
EXT = {1.0:0.8, 2.0:0.4, 3.0:0.1, 4.0:0.5, 5.0:1.0} #打分1-5分的极性值,根据key:打分获取对应特征值

DATASET = "NYC"  #NYC, Zip

if DATASET == "NYC":
    PID_LIST = range(0, 923)    #NYC
else:
    PID_LIST = range(0, 5044)   #Zip

# 作弊度高的pid
#PID_LIST = [800, 805, 552, 73, 150, 571, 572, 303, 174, 416, 267, 559, 36, 555]      

GSFG_HOME = "SourceData/Yelp{}/".format(DATASET) #数据文件路径

RESULT_FILE = "ResultData/{}_d{}_K{}_A{}/".format(DATASET, MAXSIMSUM, K, ALPHA)	#特征数据和实验数据文件路径

FILE_META = "metadata"
FILE_REVIEWCONTENT = "reviewContent"

def loadReview():
    # 读取评论数据
    file_meta = GSFG_HOME + FILE_META
    fp1 = open(file_meta)
    for lines in fp1:
        lines = lines.replace("\n", "")
        lines = lines.split('\t')
        # CUST_ID, PROD_ID, RATING, Label, fulldate, DAYDIFF
        v = [int(lines[0]), int(lines[1]), float(lines[2]), 1 if lines[3] == "-1" else 0,lines[4]] 
        v.append((datetime.datetime.strptime(v[4], "%Y-%m-%d") - datetime.datetime(2004, 1, 1, 0, 0, 0)).days)
        gvList.append(v)  # daydiff
    fp1.close()
    
    #APPEND text, index
    file_content = GSFG_HOME + FILE_REVIEWCONTENT
    fp2 = open(file_content, encoding = 'utf-8')
    for i, lines in enumerate(fp2):
        tmp = lines.split("\t")
        gvList[i].extend([tmp[3].replace("\n",""), i])
        
    fp2.close()
    
    gvList.sort(key=operator.itemgetter(5))  # sort review by daydiff           

def Similarity(date, date2, begin, k, alpha = 1):
    '''
    date:tj
    date2: tq
    begin:tq-k
    k:与tq前k个评论, 不足则取t0
    alpha:为sigma的权重
    sim(tj, tq) = 1 - (tq - tj)/(tq-tq-k)*k if tq - tq-k > 0 else 1
    '''
    k = alpha*k
    return 1 - (date2 - date) / (date2 - begin) * k if date2 - begin > 0 else 1

def LogisticNormalize(x, alpha=1):   #将x归一化0-1之间, alpha为系数
    return 2/(1 + math.exp(-x*alpha)) - 1

def GenerateEdge(vid, vid2, c, c2):
    
    rid, _, rate, label, _, date, text, _ = gvList[vid]
    rid2, _, rate2, label2, _, date2, text2, _ = gvList[vid2]
    
    create_edge = False #确定有边{True:有; False: 没有}
    if c2 - c == 1:   #是否相邻
        create_edge = True
    elif (rate >= 4 and rate2 >= 4) or (rate < 4 and rate2 < 4):  #是否共同高分或共同低分
        create_edge = True
    elif abs(FRT[rid] - FRT[rid2]) < sim_first_review_date_diff:    #首评日期是否相近
        create_edge = True
    #文本相似度
    #reviewContent_process.cosine_Distance2 余弦相似度
    #reviewContent_process.JaccardSim   Jaccard相似度
    elif reviewContent_process.cosine_Distance2(text, text2, True) >= sim_content_length_diff:  # 文本长度相差在多少以下，认为相似
        create_edge = True
    else:
        pass
    
    return create_edge

def mkdir(paths): #创建文件函数
 
    # 去除首位空格
    paths=paths.strip()
    # 去除尾部 \ 符号
    paths=paths.rstrip("/")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(paths)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(paths) 
#        print( paths+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
#        print( paths+' 目录已存在')
        return False

def FraudGuard(pid, k = 200):
    '''
    首先寻找最大相似序列，然后交给CRF处理，最终对每一条评论进行预测，进行(虚假/真实)分类
    '''
    maxlen = k + 1     #最大长度maxlen需申请k+1个位置, 因为还需要多申请一个位置保存当前评论tq
    V = dictProduct[pid]    #获取pid评论序号列表,序号为在gvList中排序后的序号
    
    def PopCampaignThreshold(begin, end, t0, k, alpha = 1):
        """
        #弹出该群组的时间阈值
        t0: t(q'-k)
        threshold = avg(Σ(tj))*k/(k-1) - t(q'-k)/(k-1)
        """
        k = alpha*k
        thres = 0
        for vi in V[begin: end+1]:
            thres += gvList[vi][5]
                    
        return (thres / (end - begin + 1) * k - t0) / (k - 1)
    #ends def PopCampaignThreshold
    
    #NONE NORMALIZE
    def ComputingFeature(camp):
        """
        计算特征，以及建立图结构
        input:
            camp: v0, v1, length, maxSimSum, rateSum:起始位置到前一条评论的累积打分和, 弹出临界值天数, sigma
        output:
            (CRF对该群组预测所需的数据: (feature, edges, edge_feature), 在原始评论中的索引: oid)
        """
        campSize = camp[2]  #当前群组大小
        campLen = campSize   #是否计算群组长度特征
        edge = list()   #稀疏矩阵存储
        edgeFeature = list()    #边特征函数, 每条边的边势为[rateDiff, dateDiff, textJaccSim, similarity]
        nodeFeature = list() #结点特征
        originalIndex = list()  #在此campaign中的评论在原始的数据集中的index
        beginVno = camp[0]
        endVno = camp[1]
       
        #newRD
#        sumRateK = 0
#        ki = beginVno - K if beginVno - K > 0 else 0
#        for i in range(ki, beginVno): sumRateK += gvList[V[i]][2]
#        avgSumRateK = sumRateK / (beginVno - ki)
        cNo = 0     #vi结点在campaign中的位置
        for ci in range(beginVno, endVno + 1):
            vid = V[ci]
            rid, _, rate, label, _, date, text, oid = gvList[vid]
            rd = 0.0
            if beginVno != 0:
                rd = abs(camp[4]/beginVno - rate)  #RD	camp[4]当前群组起始位置之前的该产品评论的打分和
		
            rank = ci      #RANK
            ext = EXT[rate] #EXT
            #文本单词数量, 第一人称代词占总人称代词的比率
            wordscount, pp1, _, _ = reviewContent_process.WordCount(text, True)    
            wc = wordscount
            #fri = math.exp(FRT[rid] - date) #youth score
            days = math.exp(gvList[V[0]][5] - date)
            qk = ci - K if ci - K > 0 else 0   #q-k的位置
            if ci == 0:
                burst = 0.0
            else:
                burst = (date - gvList[V[qk]][5]) / (ci - qk)
            nodeFeature.append([rd, rank, ext, campLen, wc, pp1, days, burst])
            #建立边以及计算边特征
            cjNo = cNo + 1    #vj结点在campaign中的位置
            for cj in range(ci + 1, endVno + 1):
                vid2 = V[cj]
                rid2, _, rate2, label2, _, date2, text2, _ = gvList[vid2]
                
                if GenerateEdge(vid, vid2, cNo, cjNo):    #大于阈值后建立一条边
                    
                    edge.append([cNo, cjNo])
                    rateDiff = math.exp(-abs(rate - rate2))     #计算评论打分相似度
                    dateDiff = math.exp(date - date2)   #计算评论时间相似度
                    cosineSim = reviewContent_process.cosine_Distance2(text, text2, True) #计算文本cosine相似度
                    frt = math.exp(-abs(FRT[rid] - FRT[rid2]))  #FRT
                    edgeFeature.append([rateDiff, dateDiff, cosineSim, frt])
                
                cjNo += 1
            #ends cj
            
            originalIndex.append(oid)
            cNo += 1
        #ends ci
        if len(edge) > 0:
            #保存特征
            nodeFeature, edge, edgeFeature = np.array(nodeFeature), np.array(edge), np.array(edgeFeature)
            return ((nodeFeature[:, NODE_FEATURES], edge, edgeFeature[:, EDGE_FEATURES]), originalIndex)
        else:
            return None
    #ends def ComputingFeature
    
    #crf 预测及结果保存
    def Predict(camp, feature):
        predict_result = EFGCRF_DICT[camp[2] if camp[2] <= 10 else 10].predict([feature])
        for pred, pvn in zip(predict_result[0], range(camp[0], camp[1] + 1)):
            if predictResult[pvn][3] == 0: #评论可能会重复预测, 只要该评论被其中一次预测为作弊, 无论曾经或以后是否还会被预测为非作弊, 那么该评论就预测为作弊.
                predictResult[pvn][3] = pred
    #ends def Predict
    
    cur_campaigns = list() #当前还没有达到临界值, 保存未被弹出的campaign集合
    if OUT_CAMPAIGN:
        campaigns = list()  #存储已经发现的campaign集合, 用于观察结果
    if OUT_CAMPAIGN_FEATURE:
        campaignsDict = dict()  #存储已经发现的campaign的字典, key:群组长度, value:[preparedata, preparedata1, ...]
    if LOAD_EFGCRF_PREDICT:
        #用于储存当前pid下的所有campaign的crf预测结果的集合 [date, rate, label, predict]
        predictResult = [[gvList[vi][5], gvList[vi][2], gvList[vi][3], 0] for vi in dictProduct[pid]]  

    #    FraudGuard(pid, K)
    i = 0    #当前进入的评论序号, 从头开始扫描每一条评论
    rate = gvList[V[i]][2]  #获取该评论的打分信息
    newV = [0, 0.0, 0.0, 0.0]   #初始化新评论 [vno, sim, simSum, sumRate]
    sumRate = rate #累积打分和
    VBuffer = []  #评论缓冲区，最大长度maxlen, 循环队列
    VBuffer.append(newV)    #首条评论进入队列
    Qlen = 1   #当前VBuffer长度
    
    while i < (len(V) - 1):
        i += 1  #进入下一条评论
        rate, date = gvList[V[i]][2], gvList[V[i]][5]   #当前评论的打分, 和时间
        
        # 计算新的date应的sigma, 缓冲区收尾两条评论的时间差, 其作用与计算评论之间的标准差类似, 但计算速度更快
#        sigma = T#(date - gvList[V[i-Qlen]][5])/Qlen
                   
        #[在V中的序号, 与新进来评论的相似度, 累积相似度和(新进来评论与VBuffer中的评论)]
        newV = [i, 0.0, 0.0, sumRate]    #[vno, sim, simSum, sumRate] sumRate是累积到上一条的打分和
        sumRate += rate
        
        pointer = i % maxlen    #campaign的结束位置(在VBuffer中的位置) ,队尾指针
        if Qlen < maxlen:                  # buffer not full
            VBuffer.append(newV)   # [vno, sim, simSum, Feature...]
            Qlen += 1
        else:
            VBuffer[pointer] = newV   #覆盖式入队
        
        begin = gvList[V[0]][5] if Qlen < maxlen else gvList[V[i - k]][5] #载入tq-k
        #计算新评论与之前评论之间的相似度, 倒序回溯
        simSumLine = 0.0    # 当前累积相似度
        maxSimSum = MAXSIMSUM
        maxpos = pointer    #campaign的起始位置(在VBuffer中的位置)
        
        p = (pointer-1) % Qlen    #指针向前移一位
        while (p != pointer):
            
            vno = VBuffer[p][0]     #取出p位置的评论序号，进而找到原始评论
            #计算sim 和 simSum
            VBuffer[p][1] = Similarity(gvList[V[vno]][5], date, begin, Qlen-1, ALPHA)
            simSumLine += VBuffer[p][1]     #从i->p累积simSum
            VBuffer[p][2] += simSumLine
            #替换保留当前最大的maxSimSum和maxpos
            if VBuffer[p][2] >= maxSimSum:
                maxpos = p
                maxSimSum = VBuffer[p][2]

            p = (p-1) % Qlen
            # ends while p
        
        maxvi = VBuffer[maxpos][0]  #maxvi表示在V中maxpos的index
        lenc = i - maxvi + 1   # length of campaign
        if lenc > 1:
            flag = True     #该campaign是否进入
            ci = 0    #群组编号, 遍历去掉重复序列和弹出已经到达阈值的序列
            while ci < len(cur_campaigns):
                cur_camp = cur_campaigns[ci]
                #去掉完全重复的群组
                if maxvi <= cur_camp[0] and i >= cur_camp[1]:
                    #当maxsimSum大于时去掉旧群组, 否则不加入
                    if maxSimSum < cur_camp[3]:
                        flag = False
                    else: #去重
                        cur_campaigns.pop(ci)
                        ci -= 1 #由于删掉群组, 编号向前进1
                        
                elif cur_camp[5] < date:
                    cur_len = cur_camp[2]
                    if OUT_CAMPAIGN:
                        campaigns.append(cur_camp)  #添加进入观察数据
                    
                    #数据处理, 为crf的输入数据做准备
                    if OUT_CAMPAIGN_FEATURE or LOAD_EFGCRF_PREDICT:
                        preData = ComputingFeature(cur_camp)
                    #进入CRF处理
                    if LOAD_EFGCRF_PREDICT and preData:
                        Predict(cur_camp, preData[0])   #预测
                    #存储特征
                    if OUT_CAMPAIGN_FEATURE and preData:
                        
                        if cur_len in campaignsDict:
                            campaignsDict[cur_len].append(preData)
                        else:
                            campaignsDict[cur_len] = [preData]
                            
                    cur_campaigns.pop(ci)
                    ci -= 1 #由于弹出群组, 编号向前进1
                #ends if 
                
                ci += 1 #编号进1
            #ends while ci
                       
            if flag:
                #VBuffer = [vno, sim, simSum, sumRate]
                #计算deadline
                vno = VBuffer[maxpos][0]
                newK = Qlen if Qlen < maxlen else k     #如果距离tq'前不足k个评论, 则取tq'的k = |q|, 如果满足则为k
                t0 = gvList[V[i + 1 - newK]][5]   #评论tq'-k的时间
                thres = PopCampaignThreshold(vno, i, t0, newK, ALPHA) #弹出群组所要达到的阈值天数
                #将此campaign信息保留: [v0, v1, length, maxSimSum, rateSum:起始位置到前一条评论的累积打分和, 弹出临界值天数, tq-k]
                if thres > date:
                    cur_campaigns.append([vno, i, lenc, maxSimSum, VBuffer[maxpos][3], thres, begin])
                else:
                    negativeThres[0] += 1   #统计阈值为负+1
        # ends if lenc > 1
    # ends while i
    
    #由于评论截止了, 将最后没有弹出的群组输出，进行预测
    for cur_camp in cur_campaigns:
        cur_len = cur_camp[2]
        if OUT_CAMPAIGN:
            campaigns.append(cur_camp)
        #数据处理, 为crf的输入数据做准备
        if OUT_CAMPAIGN_FEATURE or LOAD_EFGCRF_PREDICT:
            preData = ComputingFeature(cur_camp)
        #进入CRF处理
        if LOAD_EFGCRF_PREDICT and preData:
            Predict(cur_camp, preData[0])   #预测
            
        #以字典的方式存储方便, 同时按群组大小划分保存文件较为方便
        if OUT_CAMPAIGN_FEATURE and preData:
            
            if cur_len in campaignsDict:
                campaignsDict[cur_len].append(preData)
            else:
                campaignsDict[cur_len] = [preData]
        
    del cur_campaigns   #清空掉剩余的campaigns
    del VBuffer     #队列结束使用清空
    #建立以该产品pid为文件夹名称的子目录, 将该产品所输出的文件保存在此子目录下
    if OUT_CAMPAIGN or OUT_CAMPAIGN_FEATURE or LOAD_EFGCRF_PREDICT:
        pidFileDirectly = RESULT_FILE + str(pid) + "/"    #该pid文件夹目录位置
        mkdir(pidFileDirectly)  #该pid目录是否存在, 如果不存在, 创建以该pid命名的文件夹目录
    
    if OUT_CAMPAIGN:
        #保存用于浏览的该pid所生成的所有群组序列的信息的文件
        campaignFile = open(pidFileDirectly + "campaigns_pNo{}.txt".format(pid), "w")
        if len(campaigns) > 0:
            campsOidList = list()
            for c in campaigns:
                spamNum, camp_oid = 0, list() #作弊数量初始化, camp中oid索引
                for vi in V[c[0]:c[1]+1]: 
                    spamNum += gvList[vi][3]
                    camp_oid.append(gvList[vi][7])
                campsOidList.append(camp_oid)
                beginvid, lastvid =  V[c[0]], V[c[1]]   #beginTime, EndTime
                #[begin, end, length, simSum, spamRate, spamNum, delay, beginvid, lastVD, delayDate, sigma:T]
                #c[i] = [v0, v1, length, maxSimSum, rateSum:起始位置到前一条评论的累积打分和, 弹出临界值天数, tq-k]
                print(c[0], c[1], c[2], "%.4f" % c[3], "%.4f" % (spamNum/c[2]), spamNum, "%.2f" % (c[5]-gvList[lastvid][5]), gvList[beginvid][5], gvList[lastvid][5], "%.2f" % c[5], "%.4f" % c[6], sep="\t", file=campaignFile)
            #campsOidList =[camp1:[oid1, oid2, ..], camp2:[oid3, oid4, ...], ...]    
            with open(pidFileDirectly + "oidInCampaigns_pNo{}.pkl".format(pid), "wb") as file:
                pickle.dump(campsOidList, file)
                
            del campsOidList #保存完结果, 删除
        campaignFile.close()
        
        del campaigns
        
        
    if OUT_CAMPAIGN_FEATURE:    #输出特征文件
        
       #遍历字典, 按不同长度划分群组保存该pid文件夹下对应长度文件里
        for length, camps in campaignsDict.items():  
            #调用pickle, 可以将变量保存成文件, camps是所有长度为length的campaign, 每个campaign的内容为 -> PrepareData结果
            with open(pidFileDirectly + "PrepareData_length{}.pkl".format(length), "wb") as file:
                pickle.dump(camps, file)
        #该产品数据保存结束, 删除该字典, 回收内存
        del campaignsDict   #删除该字典
        
    if LOAD_EFGCRF_PREDICT: #输出可视化结果文件
        with open(pidFileDirectly + "MARS_plot_result_pNo{}.txt".format(pid), "w") as mfile:
            for p in predictResult:
                print(p[0], p[1], p[2], p[3], sep="\t", file=mfile)
        del predictResult
    del V   #删除评论索引
    gc.collect()    #删除字典后, 将垃圾内存回收


if __name__ == "__main__":  # 主函数
    
    #如果进入特征选择默认不输出特征文件
    if len(NODE_FEATURES) + len(EDGE_FEATURES) != 12:
        OUT_CAMPAIGN_FEATURE = False
    
    mkdir(RESULT_FILE)  #创建该文件
    
    pid = 555
    start = time.time()
    
    gvList = list()     #全局评论列表
    FRT = dict()    #全局评论人首次评论时间first_review_time
    loadReview()        #载入评论
    print("load reviews time: {} min".format((time.time()-start)/60))
    start = time.time()
    dictProduct = dict()    #key:pid, value:[vid]
    for i, v in enumerate(gvList):
        if v[1] in dictProduct:
            dictProduct[v[1]].append(i)
        else:
            dictProduct[v[1]] = [i]
        if v[0] not in FRT: FRT[v[0]] = v[5]    #加载FRT
    
    if LOAD_EFGCRF_PREDICT: #加载crf预测模型
        EFGCRF_DICT = dict()    #加载不同长度的训练好的模型字典
        for l in range(2, 11):
            paths = "model/FRAUDGUARD_{}_L{}.model".format(DATASET, l)
            if os.path.exists(paths):
                EFGCRF_DICT[l] = joblib.load(paths)
        
    negativeThres = [0] #判断阈值为负

    pno = 0
    for p in PID_LIST:
        FraudGuard(p, K)
        if pno%100 == 0:
            print("current pid: ", p)
            print(" current running time: {} min".format((time.time()-start)/60))
        pno += 1
    print("阈值为负的数量: ", negativeThres[0])

    del FRT, gvList, dictProduct
    gc.collect()
    print("running time: {} min".format((time.time()-start)/60))