import  numpy as np
import matplotlib.pyplot as plt

'''
AGNES层次聚类，采用自底向上聚合策略的算法。先将数据集的每个样本看做一个初始的聚类簇，然后算法运行的每一步中找出距离最近的两个
类簇进行合并，该过程不断重复，直至达到预设的聚类簇的个数。
'''
#计算两个向量之间的欧式距离
def calDist(X1 , X2 ):
    sum = 0
    for x1 , x2 in zip(X1 , X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5
def updateClusterDis(dataset,distance,sets,cluster_i):
    i=0
    while i<len(sets):
        dis = []
        for e in sets[i]:
            for ele in sets[cluster_i]:
                dis.append(calDist(dataset[e],dataset[ele]))
        distance[i,cluster_i]=max(dis)
        distance[cluster_i,i]=max(dis)
        i+=1
    #将每个簇和自身距离设为无穷大
    distance[np.diag_indices_from(distance)] = float('inf')
    return distance
def agens(dataset,k):
#初始化聚类簇:让每一个点都代表，一个类簇
    sets=[]
    for i in range(0,len(dataset)):
        sets.append({i})
#初始化类簇间距离的矩阵
    delta = np.array(dataset[0] - dataset)
    for e in dataset[1:, :]:
        delta = np.vstack((delta, (e - dataset)))
    distance = np.sqrt(np.sum(np.square(delta), axis=1))
    distance = np.reshape(distance, (len(dataset), len(dataset)))
    distance[np.diag_indices_from(distance)]=float('inf')
####################################################
    while len(sets)>k:
        locations=np.argwhere(distance==np.min(distance))
        #将集合合并，删除被合并的集合
        locations=locations[locations[:,0]<locations[:,1]]
        cluster_i=locations[0,0]
        cluster_j=locations[0,1]
        for e in sets[cluster_j]:
            sets[cluster_i].add(e)
        del sets[cluster_j]
        #删除被合并的簇，distance矩阵对应的行和列，并更新距离矩阵
        distance=np.delete(distance,cluster_j,axis=0)#删除对应列
        distance=np.delete(distance,cluster_j,axis=1)#删除对应行
        distance=updateClusterDis(dataset,distance,sets,cluster_i)
    # print(sets)
    return sets

from datetime import datetime
# 获取当前时间
current_time = datetime.now()
# 格式化输出时间
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("当前时间:", formatted_time)

#测试代码,使用西瓜书数据集4.0
dataset=np.loadtxt('normalized_data.txt')
results=agens(dataset,5)

for r  in  results:
    drawpoints = []
    for points in r:
        drawpoints.append(points)
    drawdata=dataset[drawpoints]
    plt.scatter(drawdata[:, 0], drawdata[:, 1], marker='o')

plt.savefig('ag.png', dpi=720, bbox_inches='tight')
# 获取当前时间
end_time = datetime.now()
# 格式化输出时间
formatted_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
print("结束时间:", formatted_end_time)
