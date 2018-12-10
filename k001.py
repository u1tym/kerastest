import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout

import cv2

( img_train, num_train ), ( img_test, num_test ) = mnist.load_data()


# img_train は、78x78の画像が60000個。つまり、構造としては、
# 78x78(=784)x60000
# の3次元配列なので、それを 60000x784の2次元配列にする
# で、255で割って、0～1.0の値にする。
# astype()でfloat32にキャストしてから計算する
img_train = img_train.reshape( 60000, 784 ).astype( 'float32' ) / 255
img_test  = img_test.reshape( 10000, 784 ).astype( 'float32' ) / 255

# [ 0, 1, ... ]
# を
# [ 1, 0, ... ], [ 0, 1, 0, ... ], ...
# にする
num_train = keras.utils.np_utils.to_categorical( num_train.astype( 'int32' ), 10 )
num_test  = keras.utils.np_utils.to_categorical( num_test.astype( 'int32' ), 10 )

# モデルの作成
model = Sequential()
model.add( Dense( 512, activation='relu', input_shape=( 784, ) ) )
model.add( Dropout( 0.2 ) )
model.add( Dense( 512, activation='relu' ) )
model.add( Dropout( 0.2 ) )
model.add( Dense( 10, activation='softmax' ) )
model.summary()

#while ind < len( img_train ) :
#	num = num_train[ ind ]
#	img = img_train[ ind ]
#	ind += 1
#
#	print( num )
#	cv2.imshow( 'image', img )
#
#	c = cv2.waitKey( 0 ) & 0xff
#	if c == 27 :
#		break
#
#cv2.destroyAllWindows()

