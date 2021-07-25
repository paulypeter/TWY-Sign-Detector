#include <iostream>
#include "TWY-Sign-Detection.h"

int SIMILAR_DIST = 50;
float MAIN_COLOUR_RATIO = 0.7;

struct SignText {
    std::string current;
};

struct SignImages {
    cv::Mat current;
    cv::Mat warning;
    cv::Mat branching;
};

SignText TwySignDetector::getSignText() {
    SignText res;
    return res;
};

cv::Mat TwySignDetector::detect_sign(cv::Mat I, int max_dim, int net_step, int out_size, int threshold) {
    int min_dim_img = (I.size[0] <= I.size[1]) ? I.size[0] : I.size[1];
	float factor = max_dim/min_dim_img;

    int w = int (I.size[1] * factor);
    int h = int (I.size[0] * factor);
	w += (w%net_step!=0)*(net_step - w%net_step);
	h += (h%net_step!=0)*(net_step - h%net_step);
	cv::Mat Iresized;
    cv::resize(I, Iresized, cv::Size(w, h));

    cv::Mat T;
	Iresized.copyTo(T);
	T = T.reshape((1,T.size[0],T.size[1],T.size[2]));

    auto model = cv::dnn::readNet("data/models/wpod_net.h5")
	Yr = model.predict(T)
	// Yr 		= np.squeeze(Yr)

	// L,TLps = reconstruct(I,Iresized,Yr,out_size,threshold)

	// return L,TLps
    return Yr;
};

//get_boundaries
SignImages getSegments(cv::Mat img) {
    int width = img.size[1];
    int height = img.size[0];
    std::vector< int > borders;
    for (int i = 0; i < width; i++) {
        std::cout << i;
    }
};

double colourDistance(cv::Mat colour1, cv::Mat colour2) {
    cv::Mat dist = colour2 - colour1;
    return cv::norm(dist);
};

// call python function?
// reconstruct(Iorig,I,Y,out_size,threshold=.9):

// 	int net_stride 	= 2**4
// 	float side 		= ((208. + 40.)/2.)/net_stride // 7.75

//  // gets first element of each subarray
// 	Probs = Y[...,0]
// 	Affines = Y[...,2:]
// 	rx,ry = Y.shape[:2]
// 	ywh = Y.shape[1::-1]
// 	iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))

// 	xx,yy = np.where(Probs>threshold)

// 	WH = getWH(I.shape)
// 	MN = WH/net_stride 

// 	vxx = vyy = 0.5 #alpha

// 	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
// 	labels = []

// 	for i in range(len(xx)):
// 		y,x = xx[i],yy[i]
// 		affine = Affines[y,x]
// 		prob = Probs[y,x]

// 		mn = np.array([float(x) + .5,float(y) + .5])

// 		A = np.reshape(affine,(2,3))
// 		A[0,0] = max(A[0,0],0.)
// 		A[1,1] = max(A[1,1],0.)

// 		pts = np.array(A*base(vxx,vyy)) #*alpha
// 		pts_MN_center_mn = pts*side
// 		pts_MN = pts_MN_center_mn + mn.reshape((2,1))

// 		pts_prop = pts_MN/MN.reshape((2,1))

// 		labels.append(DLabel(0,pts_prop,prob))

// 	final_labels = nms(labels,.1)
// 	TLps = []

// 	if len(final_labels):
// 		final_labels.sort(key=lambda x: x.prob(), reverse=True)
// 		for i,label in enumerate(final_labels):

// 			t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
// 			ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
// 			H 		= find_T_matrix(ptsh,t_ptsh)
// 			Ilp 	= cv2.warpPerspective(Iorig,H,out_size,borderValue=.0)

// 			TLps.append(Ilp)

// 	return final_labels,TLps