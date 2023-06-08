import cv2
import numpy as np

colors={'red': (0,0,255),'green':(0,255,0),'gray':(100,100,100)}
shape=(1920,1080)
a=np.zeros((shape[0],shape[1],3))
base=np.ones((shape[0],shape[1]))

for elem in colors:
	for i in range(len(colors[elem])):
		print(colors[elem][i])
		a[:,:,i]=base*colors[elem][i]
		
	cv2.imshow(elem,a.astype(np.uint8))
	cv2.imwrite("dome_hdri_haven/{0}.png".format(elem),a.astype(float))
	cv2.waitKey()



cv2.destroyAllWindows()



