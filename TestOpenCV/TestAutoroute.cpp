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
//#define ESC_KEY 1048603 // should be 27
//#define Q_KEY 	1048689 // should be 113
#define ESC_KEY 27
#define Q_KEY 	113

using namespace std;
using namespace cv;

string videoName;
VideoCapture cap;
int nbFrames;
double fps = 30;
bool videoRead = false;
int vRes;
int hRes;
float seuil = 0.4;
Mat im;
Mat imGray;
Mat imMean;
Mat imGrayMean;
Mat imMask;
vector<Mat> filmColor;
vector<Mat> filmGray;
vector<Mat> filmRoad;


Mat moyenneCouleur(int M, vector<Mat> film) {
	Mat moyenne = Mat::zeros(film[0].rows, film[0].cols, CV_32FC3);
	for (int m = 0; m < M; m++) {
		Mat img = film[m];
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				cv::Vec3b pixel = img.at<Vec3b>(i, j);
				moyenne.at<Vec3f>(i, j) += pixel;
			}
		}
	}
	for (int i = 0; i < moyenne.rows; i++) {
		for (int j = 0; j < moyenne.cols; j++) {
			moyenne.at<Vec3f>(i, j) /= (M * 256);
		}
	}
	return moyenne;
}
Mat moyenneGris(int M, vector<Mat> film) {
	Mat moyenne = Mat::zeros(film[0].rows, film[0].cols, CV_32FC1);
	for (int m = 0; m < M; m++) {
		Mat img = film[m];
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				uchar pixel = img.at<uchar>(i, j);
				moyenne.at<float>(i, j) += pixel;
			}
		}
	}
	for (int i = 0; i < moyenne.rows; i++) {
		for (int j = 0; j < moyenne.cols; j++) {
			moyenne.at<float>(i, j) /= (M * 256);
		}
	}
	return moyenne;
}
Mat creationMasque(int M, vector<Mat> film, Mat moyenne, float seuil, int begin = 0) {
	int vRes = film[0].rows;
	int hRes = film[0].cols;

	//string windowNameGrayDiffTmp = "Gray diff tmp";
	//namedWindow(windowNameGrayDiffTmp, WINDOW_AUTOSIZE);
	//string windowNameMaskFilm = "Mask film";
	//namedWindow(windowNameMaskFilm, WINDOW_AUTOSIZE);

	Mat imMask = Mat::zeros(vRes, hRes, CV_32FC1);


	for (int m = begin; m < M + begin; m++) {
		// Compute the difference between the mean and the current picture
		Mat diff = Mat::zeros(vRes, hRes, CV_32FC1);
		film[m].convertTo(diff, CV_32F);
		for (int i = 0; i < vRes; i++) {
			for (int j = 0; j < hRes; j++) {
				diff.at<float>(i, j) /= 256;
				diff.at<float>(i, j) -= moyenne.at<float>(i, j);
				diff.at<float>(i, j) = abs(diff.at<float>(i, j));
			}
		}
		//filmGrayDifference.push_back(diff.clone());

		// Compute the local mask
		Mat imMaskTmp = Mat::zeros(vRes, hRes, CV_32FC1);
		threshold(diff, imMaskTmp, seuil, 1, THRESH_BINARY);

		// Compute the global mask
		for (int i = 0; i < vRes; i++) {
			for (int j = 0; j < hRes; j++) {
				imMask.at<float>(i, j) += imMaskTmp.at<float>(i, j);
			}
		}
		// Some annimations
		//imshow(windowNameGrayDiffTmp, diff);
		//imshow(windowNameMaskFilm, imMaskTmp);
		//key = waitKey(1);
	}

	// Compute the final value of the global mask
	for (int i = 0; i < vRes; i++) {
		for (int j = 0; j < hRes; j++) {
			if (imMask.at<float>(i, j) >= 1) {
				imMask.at<float>(i, j) = 255;
			}
			else {
				imMask.at<float>(i, j) = 0;
			}
		}
	}
	imMask.convertTo(imMask, CV_8U);
	return imMask;
}


int main(int argc, const char* argv[]) {

	if (argc > 1) {
		videoName = argv[1];
		cout << "video name: " << videoName << endl;
	}
	else {
		videoName = "video.avi";
	}


	// Reading the image (and forcing it to grayscale)	   
	cap.open(videoName);

	// Making sure the capture has opened successfully
	if (cap.isOpened()) {
		cap.read(im);
		// Turning im into grayscale and storing it in imGray
		cvtColor(im, imGray, COLOR_RGB2GRAY);

		// Add the pictures to the films
		filmColor.push_back(im.clone());
		filmGray.push_back(imGray.clone());
		// test de changement
		// Read video infos
		nbFrames = cap.get(CAP_PROP_FRAME_COUNT);
		fps = cap.get(CAP_PROP_FPS);
		double duration = nbFrames / fps;
		vRes = cap.get(CAP_PROP_FRAME_HEIGHT);
		hRes = cap.get(CAP_PROP_FRAME_WIDTH);
		cout << "Nombre d images de la video :             " << nbFrames << endl;
		cout << "Nombre d images par seconde de la video : " << fps << endl;
		cout << "Duree de la video :                       " << duration << endl;
		cout << "Resolution verticale de la video :        " << vRes << endl;
		cout << "Resolution horizontale de la video :      " << hRes << endl;

	}
	else {
		// capture opening has failed we cannot do anything :'(
		return 1;
	}

	// Creating a window to display some images
	string windowName = "Original video";
	namedWindow(windowName, WINDOW_AUTOSIZE);
	string windowName_gray = "Gray video";
	namedWindow(windowName_gray, WINDOW_AUTOSIZE);

	// Waiting for the user to press ESCAPE before exiting the application	
	int key = 0;
	int noFrame = 1;
	// Read the file once and store the images in the films
	while ((key != ESC_KEY) && (key != Q_KEY) && noFrame < nbFrames) {

		imshow(windowName, im);
		imshow(windowName_gray, imGray);

		cap.read(im);
		// Turning im into grayscale and storing it in imGray
		cvtColor(im, imGray, COLOR_RGB2GRAY);

		// Add the pictures to the films
		filmColor.push_back(im.clone());
		filmGray.push_back(imGray.clone());

		// Look for waitKey documentation
		//key = waitKey(round(1000/fps)); // video is fps fps
		//key = waitKey(1);
		noFrame++;
	}
	key = waitKey(1);

	// Mean images
	/*string windowNameMean = "Colored mean";
	namedWindow(windowNameMean, WINDOW_AUTOSIZE);*/
	string windowNameGrayMean = "Gray mean";
	namedWindow(windowNameGrayMean, WINDOW_AUTOSIZE);
	//imMean = moyenneCouleur(100, filmColor); // Not realy used later, the number of frame used was reduced for execution speed
	//imshow(windowNameMean, imMean);
	key = waitKey(1);
	imGrayMean = moyenneGris(300, filmGray);
	imshow(windowNameGrayMean, imGrayMean);
	key = waitKey(1);

	//Calcul du Masque de route
	string windowNameMask = "Mask";
	namedWindow(windowNameMask, WINDOW_AUTOSIZE);

	imMask = creationMasque(nbFrames, filmGray, imGrayMean, seuil);

	imshow(windowNameMask, imMask);

	// Extract the Road from the images
	string windowNameRoad = "Road";
	namedWindow(windowNameRoad, WINDOW_AUTOSIZE);
	for (int m = 0; m < nbFrames; m++) {
		Mat imRoute;
		filmGray[m].copyTo(imRoute, imMask);
		filmRoad.push_back(imRoute.clone());

		//	imshow(windowNameRoad, imRoute);
		//	key = waitKey(1);
	}

	Mat imRoadMean;
	imGrayMean.copyTo(imRoadMean, imMask);
	imshow(windowNameRoad, imRoadMean);

	//Calcul du Masque de voitures
	string windowNameMaskCar = "Mask Cars";
	namedWindow(windowNameMaskCar, WINDOW_AUTOSIZE);
	string windowNameMaskCarMorphed = "Mask Cars Morphed";
	namedWindow(windowNameMaskCarMorphed, WINDOW_AUTOSIZE);
	string windowNameMaskCarMorphed3 = "Mask Cars Morphed3";
	namedWindow(windowNameMaskCarMorphed3, WINDOW_AUTOSIZE);
	string windowNameCars = "Cars";
	namedWindow(windowNameCars, WINDOW_AUTOSIZE);
	string windowNameContours = "Contours";
	namedWindow(windowNameContours, WINDOW_AUTOSIZE);
	int nbFramesDiff = 3;
	for (int m = 0; m < nbFrames - nbFramesDiff; m++) {
		Mat imMaskCars = creationMasque(nbFramesDiff, filmRoad, imRoadMean, 0.4, m);
		imshow(windowNameMaskCar, imMaskCars);

		Mat kernel = Mat::ones(3, 3, CV_8U);
		//morphologyEx(imMaskCars, imMaskCars, MORPH_OPEN, kernel, Point(-1, -1), 2);
		//imshow(windowNameMaskCarMorphed, imMaskCars);

		morphologyEx(imMaskCars, imMaskCars, MORPH_OPEN, kernel, Point(-1, -1), 3);
		imshow(windowNameMaskCarMorphed3, imMaskCars);

		imshow(windowNameCars, filmGray[m]);


		// Contours
		vector<vector<Point>> contours;
		findContours(imMaskCars, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		vector<vector<Point>> contours2;
		for (int k = 0; k < contours.size(); k++)
		{
			int area = contourArea(contours[k]);
			if (area > 1000) { // 1000 est pas trop mal
				contours2.push_back(contours[k]);
			}
		}

		Scalar color(0, 255, 0);
		Mat imContours = Mat::zeros(imGray.rows, imGray.cols, CV_8UC3);
		drawContours(imContours, contours2, -1, color);

		cout << "Nombre de composantes : " << contours2.size() << endl;

		imshow(windowNameContours, imContours);

		waitKey(100);
	}




	key = waitKey();
	// Destroying all OpenCV windows
	destroyAllWindows();

	return 0;
}