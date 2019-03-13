# -*- coding:utf-8 -*-

# from PIL import Image
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Compute low order moments(1,2,3)
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


def color_mean(img):
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

    return color_feature


def cut(file,vx,vy):
    """

    :param file:
    :param vx: 块大小
    :param vy: 块大小
    :return:分块集
    像素大小：4032 x 3024 分为：4 x 3个块
    1920*1080 分成4*3个块    每一块 480*360
    x=1080 y=1920
    """
    im = Image.open(file)

    #偏移量,滑动量
    dx = vx
    dy = vy
    n = 1

    #左上角切割
    x1 = 0
    y1 = 0
    x2 = vx
    y2 = vy

    imgs = []
    #纵向
    while x2 <= 1080:
        #横向切
        while y2 <= 1920:
            name3 = file + str(n) + ".jpg"
            #print n,x1,y1,x2,y2
            im2 = im.crop((y1, x1, y2, x2))
            im2_mat = np.asanyarray(im2)
            imgs.append(im2_mat)
            # im2.save(name3)

            y1 = y1 + dy
            y2 = y1 + vy
            n = n + 1
        x1 = x1 + dx
        x2 = x1 + vx
        y1 = 0
        y2 = vy

    return imgs


def abnorm_discriminant(filename, test, model, threshold):
    """
    :param test: 测试图片的块特征
    :param model: 合成的块特征
    :return: 异常 or 正常
    """
    flag = 0
    sum = 0
    for i in range(len(test)):
        sum = sum + np.linalg.norm(test[i]-model[i])

    error = sum / len(test)
    print('error is '+ str(error))

    if error < threshold:
        flag = 1
        return flag
    else:
        return flag


def get_mid(imgs_cf, model):
    """
    :param imgs_cf:正常图片特征:[块数[图片数[特征维度]]]
    :param model: 由正常图片特征合成的块特征
    :return: 训练好的阈值
    """
    dists = []  # 每张图片特征与model差值的平方根再平均后的列表
    for f in imgs_cf:
        sum = 0
        for i in range(len(f)):
            x = np.linalg.norm(np.array(f[i]) - model[i])
            sum = sum+x
        dists.append(sum/len(model))
    mid = dists[int(len(dists)/2)]
    # print("threshold: ", threshold)
    return mid


def image_cluster(data, k):
    """
    :param data:100 * 108（12x9）
    :param k: 我们将图片分成三类，大概为：拥堵，正常，空旷
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


def features(index, vx, vy, n_blk):
    """

    :param label: 聚类类别对应的图片索引
    :param vx: 图像块的宽
    :param vy: 图像块的长
    :param n_blk: 图像切成 n_blk 块
    :return: 按块特征的平均值，每张图像的特征
    """
    # compute features of images
    blocks_featrue = []
    imgs_feature = []
    # 将图像分块
    for i in range(n_blk):
        blocks_featrue.append([])
    for i in index:
        img_f = []
        res = cut(files_norm[i], vx, vy)
        for b in range(len(res)):
            blk_f = color_moments(res[b])
            blocks_featrue[b].append(blk_f)
            img_f.append(blk_f)

        imgs_feature.append(img_f)

    # 计算每个图像块特征的均值、标准差（所有normal图片）
    i = 0
    model = []  # 将正常图片的颜色矩特征的均值作为 model
    for b in blocks_featrue:
        # b：12 x 9 图片数量 x 特征维度
        i = i + 1
        arr = np.array(b)
        rows, cols = arr.shape
        # 每个块每一特征维度均值
        mean = np.mean(arr, 0)
        model.append(mean)
        # 每一特征维度标准差
        std = np.std(arr, 0)
        std_2 = np.linalg.norm(std)
        # print("第 %d 个块： " % i)
        # print("平均值： ", mean)
        # print("标准差: ", std_2)
    return model, imgs_feature


if __name__ == "__main__":

    file_dir_norm = "./point1/train"
    file_dir_test = "./point1/test_n"

    files_norm = []
    # get files at the current directory path
    for root, dirs, files_name in os.walk(file_dir_norm):
        for file in files_name:
            files_norm.append(root + '\\' + file)

    # 提取整张图片特征，用于聚类
    feature = []
    for f in files_norm:
        img = cv2.imread(f)
        ft = color_moments(img)
        feature.append(ft)
    idx1, idx2, idx3 = image_cluster(feature, 3)
    idx = [idx1, idx2, idx3]
    feature_num = len(idx)

    # 计算各个聚类的model特征，图像特征
    vx = 360  # 块宽
    vy = 480  # 块长
    n_blk = 4*3  # 切分成 n_blk 个块

    model = []
    imgs_feature = []
    mid = []
    flags = []
    for i in range(feature_num):
        if idx[i]:
            model_i, imgs_feature_i = features(idx[i], vx, vy, n_blk)
            model.append(model_i)
            imgs_feature.append(imgs_feature_i)
            mid_i = get_mid(imgs_feature_i, model_i)
            mid.append(mid_i)
            print('threhold{}  '.format(i+1) + str(1.5*mid_i))

    # # test image abnormal detection

    count = 0
    for root, dirs, files_name in os.walk(file_dir_test):
        for f in files_name:
            print(f, ": ")
            file_test = root + "\\" + f
            res = cut(file_test, vx, vy)
            img_feat = []
            for b in range(len(res)):
                blk_feat = color_moments(res[b])
                arr = np.array(blk_feat)
                img_feat.append(arr)
            for i in range(feature_num):
                flag = abnorm_discriminant(f.split('_')[0], img_feat, model[i], 1.5*mid[i])
                flags.append(flag)

            if flags[0] or flags[1] or flags[2]:
                print('测试正常')
                count = count + 1
            else:
                print('测试异常')

    print("测试图片总数：%d, 正确判断总数：%d, 正确率为：%f" % (len(files_name), count, count / len(files_name)))