import numpy as np
from scipy.spatial import KDTree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 외접 구의 외심을 계산하는 함수
def compute_circumcenter(points):
    """
    주어진 4개의 점을 통과하는 외접 구의 중심을 계산합니다.
    
    Parameters:
    points (ndarray): shape (4, 3)
    
    Returns:
    ndarray: 외심의 좌표 (3,)
    """
    A = np.array([
        [2*(points[1][0] - points[0][0]), 2*(points[1][1] - points[0][1]), 2*(points[1][2] - points[0][2])],
        [2*(points[2][0] - points[0][0]), 2*(points[2][1] - points[0][1]), 2*(points[2][2] - points[0][2])],
        [2*(points[3][0] - points[0][0]), 2*(points[3][1] - points[0][1]), 2*(points[3][2] - points[0][2])]
    ])
    b = np.array([
        points[1][0]**2 + points[1][1]**2 + points[1][2]**2 - points[0][0]**2 - points[0][1]**2 - points[0][2]**2,
        points[2][0]**2 + points[2][1]**2 + points[2][2]**2 - points[0][0]**2 - points[0][1]**2 - points[0][2]**2,
        points[3][0]**2 + points[3][1]**2 + points[3][2]**2 - points[0][0]**2 - points[0][1]**2 - points[0][2]**2
    ])
    try:
        circumcenter = np.linalg.solve(A, b)
        return circumcenter
    except np.linalg.LinAlgError:
        # 행렬이 특이한 경우 (예: 네 점이 같은 평면에 있는 경우), None을 반환
        return None

# 3D 데이터 생성 (예: 두 클래스의 점)
def generate_3d_data(n_samples=100):
    """
    두 개의 클래스로 구성된 3D 데이터를 생성합니다.
    
    Returns:
    X (ndarray): 데이터 포인트 (n_samples*2, 3)
    y (ndarray): 레이블 (n_samples*2,)
    """
    np.random.seed(42)
    # 클래스 0: 구 중심 (0,0,0) 주위
    X0 = np.random.randn(n_samples, 3) + np.array([0, 0, 0])
    y0 = np.zeros(n_samples)
    # 클래스 1: 구 중심 (5,5,5) 주위
    X1 = np.random.randn(n_samples, 3) + np.array([5, 5, 5])
    y1 = np.ones(n_samples)
    # 결합
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    return X, y

# 데이터 생성
X, y = generate_3d_data(n_samples=100)

# KDTree를 사용하여 최근접 이웃 찾기
k = 4  # 4개의 최근접 이웃
tree = KDTree(X)
# 각 점의 4개의 최근접 이웃을 찾되, 자기 자신 제외
_, indices = tree.query(X, k=k+1)  # k+1을 하는 이유는 자기 자신을 포함하기 때문
indices = indices[:, 1:]  # 첫 번째는 자기 자신이므로 제외

# 외심 계산
circumcenters = []
circumcenters_labels = []
for i in range(X.shape[0]):
    neighbors_idx = indices[i]
    points = X[np.concatenate(([i], neighbors_idx))]
    cc = compute_circumcenter(points)
    if cc is not None:
        circumcenters.append(cc)
        # 다수결 방식으로 레이블 설정
        labels = y[np.concatenate(([i], neighbors_idx))]
        majority_label = 1 if np.sum(labels) > (len(labels) / 2) else 0
        circumcenters_labels.append(majority_label)

circumcenters = np.array(circumcenters)
circumcenters_labels = np.array(circumcenters_labels)

print(f"총 외심 개수: {circumcenters.shape[0]}")
print(f"클래스 0 외심 개수: {np.sum(circumcenters_labels == 0)}")
print(f"클래스 1 외심 개수: {np.sum(circumcenters_labels == 1)}")

# 외심 데이터에 대해 train_test_split을 적용
X_train, X_test, y_train, y_test = train_test_split(circumcenters, circumcenters_labels, test_size=0.2, random_state=42)
print(f"훈련 세트 크기: {X_train.shape[0]}, 테스트 세트 크기: {X_test.shape[0]}")

# SVM 학습
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(f"SVM 정확도: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 시각화
fig = plt.figure(figsize=(14, 7))

# 원본 데이터와 외심 시각화
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[y==0][:,0], X[y==0][:,1], X[y==0][:,2], c='blue', label='Class 0', alpha=0.5)
ax1.scatter(X[y==1][:,0], X[y==1][:,1], X[y==1][:,2], c='red', label='Class 1', alpha=0.5)
ax1.scatter(circumcenters[:,0], circumcenters[:,1], circumcenters[:,2], c='green', label='Circumcenters', marker='^')
ax1.set_title('원본 데이터 및 외심')
ax1.legend()

# SVM 결정 경계 시각화
ax2 = fig.add_subplot(122, projection='3d')

# 훈련 세트 시각화
ax2.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], X_train[y_train==0][:,2], c='blue', label='Class 0 (Train)', alpha=0.5)
ax2.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], X_train[y_train==1][:,2], c='red', label='Class 1 (Train)', alpha=0.5)

# SVM의 결정 경계 평면 시각화
# 결정 경계 평면 방정식: w.x + b = 0
# z = (-w1*x - w2*y - b) / w3
if svm.coef_.shape[1] == 3:
    w = svm.coef_[0]
    b = svm.intercept_[0]
    if w[2] != 0:
        xx, yy = np.meshgrid(
            np.linspace(X_train[:,0].min()-1, X_train[:,0].max()+1, 10),
            np.linspace(X_train[:,1].min()-1, X_train[:,1].max()+1, 10)
        )
        zz = (-w[0]*xx - w[1]*yy - b) / w[2]
        ax2.plot_surface(xx, yy, zz, alpha=0.3, color='green')
    else:
        print("SVM의 w3이 0이어서 결정 경계를 z로 표현할 수 없습니다.")
else:
    print("SVM의 계수 차원이 3이 아닙니다.")

ax2.set_title('외심 데이터 및 SVM 결정 경계')
ax2.legend()

plt.show()

# 3D 외접 구 시각화 (선택 사항)
# 예를 들어, 첫 번째 외접 구를 시각화
def plot_sphere(ax, center, points, color='cyan', alpha=0.2):
    """
    외접 구를 시각화합니다.
    
    Parameters:
    ax (Axes3D): matplotlib 3D 축
    center (ndarray): 구의 중심 (3,)
    points (ndarray): 구를 이루는 4개의 점 (4,3)
    color (str): 색상
    alpha (float): 투명도
    """
    # 구의 반지름 계산
    radius = np.linalg.norm(points[0] - center)
    # 구를 생성
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha)

# 예시: 첫 번째 외접 구를 시각화
if circumcenters.shape[0] > 0:
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='bwr', alpha=0.5)
    first_cc = circumcenters[0]
    neighbors_idx = indices[0]
    points = X[np.concatenate(([0], neighbors_idx))]
    plot_sphere(ax, first_cc, points, color='green', alpha=0.3)
    ax.scatter(first_cc[0], first_cc[1], first_cc[2], c='black', marker='x', s=100, label='Circumcenter')
    ax.set_title('첫 번째 외접 구 및 외심')
    ax.legend()
    plt.show()
