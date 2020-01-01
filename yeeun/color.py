import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 프로젝트 다시 시작
# 코드를 뽑을 때, 색들을 동일한 비율 크기로 만들 수 있는 방법 찾기
# https://inyl.github.io/programming/2017/07/31/opencv_image_color_cluster.html 참고!!!

# 1) 이미지 파일을 image 라는 파일로 읽어오기 (현재 파일에 있는 사진 불러오기)
image = cv2.imread("porto.jpg")

# 2) 데이터 형식 확인 및 비율 줄이기
# 이미지가 너무 크면 분석에 시간이 오래 걸림
print(image.shape) #  (height, width, channel)
# ((768, 1024, 3))

# 여기서 channel는 B,G,R 3차원을 의미함

r = 5 # 5분의 1로 줄이기
image2 = cv2.resize(image, dsize=(int(image.shape[1]/r), int(image.shape[0]/r)), interpolation=cv2.INTER_AREA)
print(image2.shape)
# (153, 204, 3)


# 4) 색공간 변환
# 처음 이미지를 읽어들이면 BGR로 읽어들임
# 분석을 하기 위해 색상/채도/명도를 나타내는 HSV 색공간으로 변환
hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)


# 5) 차원 분산 -> 수치적 계산 통합하기 (width + height 통합)
image3 = hsv.reshape((hsv.shape[0] * hsv.shape[1], 3)) # height, width 통합
print(image3.shape)
# (31212, 3)

# 6) 이미지 학습 - scikit-learn k-mean
k = 8 #
clt = KMeans(n_clusters = k)
clt.fit(image3)

# 7) clustering된 컬러값 확인
for center in clt.cluster_centers_:
    print(center)

# 코드값 확인 (8개)
# [ 12.5198939   42.21927498 143.84084881]
# [101.70016327  88.24179592 221.94032653]
# [25.80571137 45.92299847 82.72463029]
# [104.22453573  23.67754643 122.32470456]
# [ 10.60024106 153.08758538 202.03254319]
# [105.77538642  46.65681863 222.52768538]
# [ 11.61642105 136.44842105 123.39368421]
# [ 10.92467359  60.87546033 209.69233344]

# 8) 각 컬러의 분율 확인
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


hist = centroid_histogram(clt)
print(hist)
# [0.07990516 0.39247725 0.06170704 0.07260028 0.07602845 0.09659746 0.05693323 0.16375112]

# 9) 추출한 color와 histogram 데이터로 화면에 그래프화
def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

bar = plot_colors(hist, clt.cluster_centers_)


# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
