//
//  main.cpp
//  VSION-TP1
//
//  Created by Jean-Marie Normand on 04/02/2016.
//  Copyright © 2016 Jean-Marie Normand. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>

//#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// no idea why on the virtual box the keycode is completely wrong
#define ESC_KEY 1048603 // should be 27
#define Q_KEY 	1048689 // should be 113

using namespace std;
using namespace cv;

std::string videoName;
int camID = -1;
cv::VideoCapture cap;
int nbFrames;
double fps = 30;
int curFrame = 0;
bool videoRead = false;

cv::Mat im;
cv::Mat imGray;
cv::Mat imLab;

cv::Mat moy, diff, stdDev;
cv::Mat binRoad, morpho;
cv::Mat road, moyRoad;
cv::Mat diffMoy, seuilDM;

Mat cross = (Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);

std::vector<cv::Mat> myVideo;


std::vector<cv::Scalar> colorTable;

// Optical Flow test
cv::Mat prevImgLK;
cv::Mat curImgLK;
TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
vector<float> err;
vector<uchar> status;
Size winSize(31, 31);
vector<Point2f> prevPoints;
vector<Point2f> curPoints;
vector<Point2f> centerCars;



void moyenneStdDev(std::vector<cv::Mat> vid, int imDeb, int nbIm, cv::Mat& moy, cv::Mat& stdDev, cv::Mat& diff) {

	moy = cv::Mat::zeros(vid[imDeb].rows, vid[imDeb].cols, CV_32FC1);

	vid[imDeb].copyTo(diff);
	diff = cv::Mat::zeros(diff.rows, diff.cols, diff.type());

	int difVal;

	cv::Mat moy2;
	moy2 = cv::Mat::zeros(vid[imDeb].rows, vid[imDeb].cols, CV_32FC1);


	for (int vi = imDeb; vi < (imDeb + nbIm) - 1; vi++) {
		for (int i = 0; i < moy.rows; i++) {
			for (int j = 0; j < moy.cols; j++) {

				//cout<<"vi: "<<vi<< "i: "<<i << " j: "<<j<<endl;

				moy2.at<float>(i, j) += vid[vi].at<uchar>(i, j);
				//cout<<"moy["<<i<<","<<j<<"]="<<moy.at<uchar>(i, j)<<endl;
				//diff.at<uchar>(i, j) += abs(vid[vi].at<uchar>(i, j) - vid[vi-1].at<uchar>(i, j));
				difVal = abs(vid[vi + 1].at<uchar>(i, j) - vid[vi].at<uchar>(i, j));
				//cout<<"difVal["<<i<<","<<j<<"]="<<difVal<<endl;
				if (difVal < 50) {
					diff.at<uchar>(i, j) += 0;
				}
				else {
					diff.at<uchar>(i, j) += difVal;
				}
			}
		}

		accumulate(vid[vi], moy);

		//diff += abs(vid[vi]-vid[vi-1]);
	}

	moy /= nbIm;
	moy.convertTo(moy, CV_8U);
	/////imshow("ImMoy",moy);


	for (int i = 0; i < moy2.rows; i++) {
		for (int j = 0; j < moy2.cols; j++) {
			moy2.at<float>(i, j) /= nbIm;
		}
	}
	moy2.convertTo(moy2, CV_8U);
	/////imshow("ImMoy2",moy2);
}

/*
int getFramesFromVideo(std::string videoName) {
   int  nFrames;
   char tempSize[4];

   // Trying to open the video file
   ifstream  videoFile;
   videoFile.open( videoName , ios::in | ios::binary );
   // Checking the availablity of the file
   if ( !videoFile ) {
	  cout << "Couldn't open the input file " << videoName << endl;
	  exit( 1 );
   }

   // get the number of frames
   videoFile.seekg( 0x30 , ios::beg );
   videoFile.read( tempSize , 4 );
   nFrames = (unsigned char ) tempSize[0] + 0x100*(unsigned char ) tempSize[1] + 0x10000*(unsigned char ) tempSize[2] +    0x1000000*(unsigned char ) tempSize[3];
   cout<<"nframes: "<<nFrames<<endl;
   videoFile.close(  );

   return nFrames;
}
 */


int main(int argc, const char* argv[]) {

	if (argc > 1) {
		videoName = argv[1];
		cout << "video name: " << videoName << endl;
		if (videoName.length() == 1) {
			camID = atoi(videoName.c_str());
		}
	}
	else {
		videoName = "video.avi";
	}

	// Reading the image (and forcing it to grayscale)	   
	if (camID != -1) {
		cap.open(camID);
	}
	else {
		cap.open(videoName);
	}
	//im = imread("", CV_LOAD_IMAGE_GRAYSCALE);
	if (cap.isOpened()) {
		cap >> im;
		//int nF = getFramesFromVideo(videoName);
		nbFrames = cap.get(CAP_PROP_FRAME_COUNT);
		double fffps = cap.get(CAP_PROP_FPS);
		curFrame = 1;
		cout << " nbFrames: " << nbFrames << endl;//" et l'autre: "<<nF<<endl;
		cout << " IM cols: " << im.cols << " rows: " << im.rows << " type: " << im.type() << endl;
		cout << " FPS: " << fffps << endl;

		cvtColor(im, imGray, cv::COLOR_BGRA2GRAY);
		myVideo.push_back(imGray.clone());
		imGray.copyTo(moy);
		imGray.copyTo(diff);
		imGray.copyTo(stdDev);
	}
	else {
		return 1;
	}

	// Storing  all images in a vector
	for (int i = 1; i < nbFrames; i++) {
		cap >> im;
		cvtColor(im, imGray, cv::COLOR_BGRA2GRAY);
		myVideo.push_back(imGray.clone());
		//cout<<"Storing frame: "<<i<<endl;
	}


	videoRead = true;
	// Resetting video
	cap.set(CAP_PROP_POS_MSEC, 0);
	curFrame = 0;



	// Generating colors
	RNG rng(12345);
	for (int i = 0; i < 2048; i++) {
		colorTable.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
	}



	// Creating a window to display some images
	namedWindow("Original video");

	if (videoRead) {
		cout << "cols: " << moy.cols << " rows: " << moy.rows << endl;
		cout << "cols: " << diff.cols << " rows: " << diff.rows << endl;

		moyenneStdDev(myVideo, 0, 200, moy, stdDev, diff);

		imshow("Moyenne", moy);
		/////imshow("Diff", diff);

		cv::threshold(diff, binRoad, 50, 255, THRESH_BINARY);
		morphologyEx(binRoad, morpho, MORPH_CLOSE, cross, Point(-1, -1), 2);

		morphologyEx(morpho, morpho, MORPH_OPEN, cross, Point(-1, -1), 1);

		//invert(morpho,morpho);
		/////imshow("Binarization", binRoad);
		/////imshow("Morpho", morpho);
	}

	cout << "Moyenne computed!" << endl;

	// Waiting for the user to press ESCAPE before exiting the application	
	int key = 0;
	while ((key != ESC_KEY) && (key != Q_KEY)) {
		curFrame++;
		if (curFrame == nbFrames) {
			cap.set(CAP_PROP_POS_MSEC, 0);
			curFrame = 0;
		}
		cap >> im;

		// To Gray !
		cvtColor(im, imGray, COLOR_BGRA2GRAY);

		// TEST LAB
		/*
		   cvtColor(im,imLab,CV_BGR2YCrCb);
		   imshow("Image Lab",imLab);


		  // LAB to gray
		cv::Mat imGrayLab;
		//cvtColor(imLab,imGrayLab,CV_Lab2GRAY);
		   */

		   // TEST
		int pixG = imGray.at<uchar>(0, 0);
		Vec3b col = im.at<Vec3b>(0, 0);
		int greenCol = col[1];


		cout << "curFram:" << curFrame << endl;

		// On garde que la route
		imGray.copyTo(road, morpho);
		moy.copyTo(moyRoad, morpho);

		imshow("Route", road);
		/////imshow("Road Moyenne",moyRoad);

		// Diff moyenne
		diffMoy = abs(road - moyRoad);
		/////imshow("Diff Road Moyenne",diffMoy);


		// Remove half of the image
		cv::Mat myMask;
		myMask = cv::Mat::zeros(im.rows, im.cols, imGray.type());
		for (int i = myMask.rows / 2; i < myMask.rows; i++) {
			for (int j = 0; j < myMask.cols; j++) {
				myMask.at<uchar>(i, j) = 255;
			}
		}
		/////imshow("mask",myMask);

		// Seuillage + binarisation
		//cv::adaptiveThreshold( diffMoy, seuilDM, 255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 5 );
		cv::threshold(diffMoy, seuilDM, 100, 255, THRESH_BINARY);

		/////imshow("Seuillage Simple", seuilDM.clone());

		morphologyEx(seuilDM, seuilDM, MORPH_CLOSE, cross, Point(-1, -1), 5);//5);

		/////imshow("Seuillage Morpho",seuilDM);

		cv::Mat cut;
		seuilDM.copyTo(cut, myMask);

		/// Find contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		cv::Mat output = cv::Mat::zeros(im.size(), CV_8UC3);


		findContours(cut, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		int nbContours = contours.size();
		cout << "Nb CARS CONTOURS: " << nbContours << endl;

		int nbContoursJim = 0;
		float threshAreaCar = 250;

		/// Approximate contours to polygons + get bounding rects and circles
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<Point2f> center(contours.size());
		vector<float> radius(contours.size());

		// Clearing cars???
		centerCars.clear();

		for (int i = 0; i < contours.size(); i++) {
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);

			// Checking area
			if (contourArea(contours_poly[i]) > threshAreaCar) {
				nbContoursJim++;

				centerCars.push_back(center[i]);
			}

		}

		cout << "Nb CARS JIM: " << nbContoursJim << endl;



		/// Draw contours
		Mat drawing = Mat::zeros(im.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			/////drawContours( output, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );

			drawContours(output, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0,
				Point());
			rectangle(output, boundRect[i].tl(), boundRect[i].br(), color, 2,
				8, 0);
			circle(output, center[i], (int)radius[i], color, 2, 8, 0);
		}


		imshow("Labelled", output);

		cv::Mat testCC(cut.size(), CV_32S);
		int nLabels = connectedComponents(cut, testCC, 8);
		cout << "Nb CARS: " << nLabels - 1 << endl;
		std::vector<Vec3b> colors(nLabels);
		colors[0] = Vec3b(0, 0, 0);		//background
		for (int label = 1; label < nLabels; ++label) {
			colors[label] = Vec3b((rand() & 255), (rand() & 255),
				(rand() & 255));
		}
		Mat dstCC(cut.size(), CV_8UC3);
		for (int r = 0; r < dstCC.rows; ++r) {
			for (int c = 0; c < dstCC.cols; ++c) {
				int label = testCC.at<int>(r, c);
				Vec3b& pixel = dstCC.at<Vec3b>(r, c);
				pixel = colors[label];
			}
		}

		imshow("Connected comps", dstCC);


		// Arrêt sur image
		if (curFrame == 190) { // 614 frames in avi -- 601 in mov
			// 175+13 = 188
			cv::waitKey(0);
		}


		// Test Optical Flow

		// Retrieving previous and current frames
		if (curFrame == 0) {
			prevImgLK = myVideo[curFrame].clone();
			curImgLK = prevImgLK.clone();
		}
		else {
			prevImgLK = myVideo[curFrame - 1].clone();
			curImgLK = myVideo[curFrame].clone();
		}

		// Filling current points
		prevPoints.clear();
		for (int i = 0; i < centerCars.size(); i++) {
			//curPoints.push_back(centerCars[i]);
			prevPoints.push_back(centerCars[i]);
		}



		// Computing optical flow
		if (prevPoints.size() != 0) {
			curPoints.clear();
			cv::calcOpticalFlowPyrLK(prevImgLK, curImgLK, prevPoints, curPoints, status, err, winSize, 3, termcrit, 0, 0.001);

			// Displaying connection between points?
			for (int j = 0; j < curPoints.size(); j++) {
				cout << "j: " << j << endl;
				if (status[j]) {
					cv::line(im, prevPoints[j], curPoints[j], colorTable[j], 4);
				}


				//cv::circle(im,curPoints[j],4,colorTable[j]);
			}
		}

		// Swapping points used in LK
		//std::swap(curPoints, prevPoints);


		// Showing "original" videos
		imshow("Original video", im);
		imshow("Gray video", imGray);


		// waiting
		key = waitKey((int)(1000 / fps)); // video is 30fps
	}

	destroyAllWindows();

	return 0;
}
