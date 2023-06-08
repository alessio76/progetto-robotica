YOLACT need the image to be in bgr format while DenseFusion in the rgb format,

DenseFusion outputs quaternion in the wxyz format but the train and inference code transform the input quaternion in a matrix to rotate the points so the input quaternoin could be in any format
