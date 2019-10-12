# coding: UTF-8
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

# 数字データの読み込み
digits = datasets.load_digits()

# データの形式を確認
# print(digits.data)
# print(digits.data.shape)

# データ数の取得
n = len(digits.data)

# 画像の正解値の表示
# images = digits.images
# labels = digits.target
# for i in range(10):
#     # 引数　：　行,列,表示位置
#     plt.subplot(2, 5, i + 1)
#     # 画像の表示　cmapで白黒　interpolationでピクセル間の補完
#     plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
#     # 軸の表示
#     plt.axis("off")
#     plt.title("Training:" + str(labels[i]))
# plt.show()

# サポートベクターマシン
clf = svm.SVC(gamma=0.01, C=100.0)
# サポートベクターマシンによる訓練 （6割のデータを使用、残りの4割は検証用）
clf.fit(digits.data[:n*6//10 ], digits.target[:n*6//10])

# # 最後の10個のデータをチェック
# # 正解　[マイナスを指定すると末尾からの範囲]
# print(digits.target[-10:])
# # 予測を行う　（数字を読み取る）
# print(clf.predict(digits.data[-10:]))


# 残りの4割の画像に対して、数字読み取り
# 正解
expected = digits.target[-n*4//10:]
# 予測
predicted = clf.predict(digits.data[-n*4//10:])
# 正解率
print(metrics.classification_report(expected, predicted))
# ご認識のマトリクス
print(metrics.confusion_matrix(expected, predicted))

# 予測と画像の対応（一部）
images = digits.images[-n*4//10:]
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.axis("off")
    plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Guess: " + str(predicted[i]))
plt.show()
