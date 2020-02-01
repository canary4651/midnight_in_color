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

r = 5 # 5분의 1로 줄이기
image2 = cv2.resize(image, dsize=(int(image.shape[1]/r), int(image.shape[0]/r)), interpolation=cv2.INTER_AREA)

# 3) 차원 분산 -> 수치적 계산 통합하기 (width + height 통합)
image3 = image2.reshape((image2.shape[0] * image2.shape[1], 3)) # height, width 통합

# 4) 이미지 학습 - scikit-learn k-mean
k = 8 #
clt = KMeans(n_clusters = k)
clt.fit(image3)

# 5) clustering된 컬러값 확인
# for center in clt.cluster_centers_:
#     print(center)

# 6) 각 컬러의 분율 확인
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
nohist = hist*0+(1/8)
# 9) 추출한 color와 histogram 데이터로 화면에 그래프화
def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    colors = []
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        color = color[::-1]
        color = color.astype("uint8").tolist()
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color, -1)
        startX = endX
        colors.append(color)

    # return the bar chart
    return bar, colors

# bar 간격 똑같이 뽑으려면 HIST = False, 비중에 따라 달리 하려면 Hist = True
HIST = False
if HIST:
    bar, colors = plot_colors(hist, clt.cluster_centers_)
else:
    bar, colors = plot_colors(nohist, clt.cluster_centers_)

print(colors) # Colors = RGB

# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
