import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from functools import reduce


def a_hash(image):
    # image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_LINEAR)
    # image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sum = 0
    for i in range(8):
        for j in range(8):
            sum = sum + image[i, j]
    avg = sum / 64
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if image[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def p_hash(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = image  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    # 拿到左上角的8 * 8
    vis1 = vis1[0:8, 0:8]

    # 把二维list变成一维list
    img_list = np.ndarray.flatten(vis1)

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    hash_str = ''
    avg_list = ['0' if i < avg else '1' for i in img_list]
    for i in avg_list:
        hash_str = hash_str + i

    # 得到哈希值
    return hash_str
    # return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 8 * 8, 4)])


def color_moments(img):
    """compute feature with color moment"""

    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    return color_feature


def image_cluster(data, k):
    """
    :param data:64
    :param k: 我们将图片分成3类
    :return: 类别索引
    """
    estimator = KMeans(n_clusters=k, init='k-means++')
    estimator.fit(data)
    label_pred = estimator.labels_
    index1 = []
    index2 = []
    index3 = []
    for l in range(len(label_pred)):
        if label_pred[l] == 0:
            index1.append(l)
        elif label_pred[l] == 1:
            index2.append(l)
        else:
            index3.append(l)

    centroids = estimator.cluster_centers_
    # print("标签：\n", label_pred)
    # print("聚类中心：\n",centroids)
    return index1, index2, index3


def hamming_dist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


def cal_dist(idx, feature):
    lenth = len(idx)
    distance = -np.ones((lenth, lenth))
    for i in range(lenth):
        for j in range(lenth):
            distance[i, j] = hamming_dist(feature[idx[i], -1], feature[idx[j], -1])
    return distance


def cal_dist_test(idx, avg, feature, feature_test):
    test_num = len(feature_test)
    train_num = len(idx)
    count = 0
    distance_test = np.ones((test_num, train_num))
    for i in range(test_num):
        for j in range(train_num):
            distance_test[i, j] = hamming_dist(feature_test[i, -1], feature[idx[j], -1])
    dis_sum = np.sum(distance_test, axis=1)/train_num
    # distance_test_norm = distance_test[0]
    # avg_test = reduce(lambda x, y: x+y, distance_test_norm)/(len(distance_test_norm))
    # [cow, row] = dis_sum.shape
    # print(cow, row)
    flag = np.zeros(test_num)
    for i in range(test_num):
        if dis_sum[i] < avg:  # 说明这张图片符合要求，是同一张图
            flag[i] = 1
    return dis_sum, flag


if __name__ == "__main__":

    filedir_norm = "./point1/train"
    file_dir_test = "./point1/test_p"

    files_norm = []
    for root, dirs, files_name in os.walk(filedir_norm):
        for file in files_name:
            files_norm.append(root + '\\' + file)
    feature = []
    feature_hsv = []
    file_num = 0
    for f in files_norm:
        img = cv2.imread(f)
        ft = a_hash(img)
        ft_phash = p_hash(img)
        ft_hsv = color_moments(img)
        # feature.append(ft)
        # print(ft)
        # print(ft_phash)
        feature.append(ft_phash)
        feature_hsv.append(ft_hsv)
        file_num = file_num + 1
    feature = np.array(feature).reshape(-1, 1)

    # 对64维hash向量进行聚类
    # idx1, idx2, idx3 = image_cluster(feature, 3)
    # 对hsv色彩空间图像聚类
    idx1, idx2, idx3 = image_cluster(feature_hsv, 3)

    # print(feature)
    idx = [idx1, idx2, idx3]
    array = np.arange(0, file_num)

    avgs = []
    for i in range(3):
        distance = cal_dist(idx[i], feature)
        dis_reshape = distance.reshape(1, -1)[0]
        if len(idx[i]) > 1:
            avg = reduce(lambda x, y: x+y, dis_reshape)/(len(dis_reshape)-len(idx[i]))
        else:
            avg = 0
        print("avg {} ".format(i+1) + str(avg))
        print("threshold {}".format(i+1), str(avg*1.75))
        avgs.append(avg)

    files_norm_test = []
    feature_test = []
    file_num_test = 0
    for root, dirs, files_name in os.walk(file_dir_test):
        for file in files_name:
            files_norm_test.append(root + '\\' + file)
    for f in files_norm_test:
        img = cv2.imread(f)
        ft = p_hash(img)
        feature_test.append(ft)
        file_num_test = file_num_test + 1
    test_num = len(feature_test)
    feature_test = np.array(feature_test).reshape(-1, 1)
    flags = []
    dis_sums = []
    for i in range(3):
        dis_sum, flag = cal_dist_test(idx[i], avgs[i]*1.75, feature, feature_test)
        flags.append(flag)
        dis_sums.append(dis_sum)

    count = 0
    for i in range(test_num):
        for j in range(3):
            print("the distance {} :".format(j) + str(dis_sums[j][i]))
        if flags[0][i] or flags[1][i] or flags[2][i]:
            print(files_norm_test[i] + "测试正常")
            count = count + 1
        else:
            print(files_norm_test[i] + "测试出错")
    print("accuracy = " + str(count/test_num))
