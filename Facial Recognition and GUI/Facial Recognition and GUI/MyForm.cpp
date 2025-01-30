//M/M///////////////////////////////////////////////////////M/M
//
// 
// 
//	   This open source project is provided under Freypt contributors.
//		 The Use, Redistribution, and Modification of the project must always include this comment
//		   If you do not agree to these conditions. Do not download or install  
//	   
// 
// 
//	   This project is part of the finals defense system for computer programming in C++ 
//		 Students: Lance Trias, Carlos Buclares, Vince Valbuena, Kristian Gopez, Fhel Bete, Johnielle Arboleda, Joshua Alviar
//		   Mentor: May Figueroa Barcelona
//
//
//
//	   All Third party libraries and software are property of their respective owners
//
//
//
//
//M/M/////////////////////////////////////////////////////////M/M

#include "MyForm.h"
#include <opencv2/highgui/highgui.hpp>		// OpenCV module for GUI (The use of webcam)
#include <opencv2/imgproc.hpp>				// OpenCV module for Image/Video Processing
#include <opencv2/objdetect.hpp>			// OpenCV module for Object/Facial Detection
#include <opencv2/face.hpp>					// OpenCV module for robust Facial Recognition


#include <Windows.h>						// 
#include <tchar.h>							// 

#include <vector>							// std module for better data management
#include <iostream>							// Input/Output for console
#include <filesystem>						// File handling

using namespace cv;
using namespace cv::face;
namespace fs = std::filesystem;
using namespace System;
using namespace System::Windows::Forms;

std::string cascadeName, nestedCascadeName;

cv::String face_cascade_dir = "c:/Users/LanceyFreypa/Documents/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
cv::String eyes_cascade_dir = "c:/Users/LanceyFreypa/Documents/Libraries/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

void Take_Sample_Faces(int, int, Mat, std::string);
void Run_Webcam(std::string);
void Train_Data_Set(const std::string, CascadeClassifier);
void Recognize_Faces(CascadeClassifier);
void directory_exists(const fs::path& p, fs::file_status s = fs::file_status{})
{
	std::cout << p << std::endl;
	if (fs::status_known(s) ? fs::exists(s) : fs::exists(p))
		std::cout << " folder exists\n";
	else
		std::cout << " Generating folder..\n";
	fs::create_directory(p);
}

Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();



[STAThreadAttribute]

void main() {
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	$safeprojectname$::MyForm form;
	Application::Run(% form);

	std::cout << "Initializing Necessary Variables and Pre-requisities.." << std::endl;

	const std::string DATASET_DIRECTORY = "c:/Users/LanceyFreypa/Documents/DataSet/";
	const std::string TRAINER_DIRECTORY = "c:/Users/LanceyFreypa/Documents/Trainer/";
	const fs::path DATASET_DIR{ DATASET_DIRECTORY };
	const fs::path TRAINER_DIR{ TRAINER_DIRECTORY };

	//  Check to see if directory exists or not. if it doesn't, create the respective folder
	directory_exists(DATASET_DIR);
	directory_exists(TRAINER_DIR);

	Run_Webcam(DATASET_DIRECTORY);
}

void Run_Webcam(std::string path)
{
	VideoCapture cap(0); // run default camera

	CascadeClassifier cascade, nestedCascade;

	Mat frame;  // A class for the natural videocapture
	Mat grayScale; // A class conversion of the frames into grayscale

	cap >> frame;  // grabs a frame from the videocapture as data for the class frame

	// Loads the .xml file from the directory, if not found, display as nothing found
	nestedCascade.load(eyes_cascade_dir);
	if (nestedCascade.empty())
	{
		std::cerr << "Could not load the .xml file" << std::endl;
		std::cin.get();
	}

	// Loads the .xml file from the directory, if not found, display as nothing found
	cascade.load(face_cascade_dir);
	if (cascade.empty())
	{
		std::cerr << "Could not load the .xml file for cascade" << std::endl;
		std::cin.get();
	}

	//  check whether the camera is open. if not, alert the user
	if (cap.isOpened() == false)
	{
		std::cerr << "Cannot Open the Camera" << std::endl;
		std::cin.get();
		return;
	}

	double width = cap.get(CAP_PROP_FRAME_WIDTH);
	double height = cap.get(CAP_PROP_FRAME_HEIGHT);
	double scale = 1;
	double face_ID = 1;

	std::cout << "Resolution of camera is " << width << " x " << height << std::endl;

	std::string window_name = "My Camera Feed";
	namedWindow(window_name); //create a window called "My Camera Feed"
	namedWindow("GrayScale");

	int count = 0;
	while (true)
	{
		bool bSuccess = cap.read(frame); // read a new frame from video 

		//Breaking the while loop if the frames cannot be captured
		if (bSuccess == false)
		{
			std::cout << "Video camera is disconnected" << std::endl;
			std::cin.get(); //Wait for any key press
			break;
		}

		std::vector<Rect> faces;
		std::vector<Rect> eyes;


		if (!frame.data) // check if frame is receiving new frames or not
		{
			std::cout << "No frame/Image is being recorded" << std::endl;
		}

		cvtColor(frame, grayScale, COLOR_BGR2GRAY); // convert to grayscale, then check if successful
		if (!grayScale.data)
		{
			std::cout << "No frame/Image is being grayScaled" << std::endl;
		}

		cascade.detectMultiScale(grayScale, faces, 1.1, 4); // Check the image/frame for faces

		for (int i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(255, 0, 0), 1.5); // draw a rectangle on detected face
			nestedCascade.detectMultiScale(grayScale, eyes, 1.1, 4);
			for (int j = 0; j < eyes.size(); j++)
			{
				rectangle(frame, eyes[j].tl(), eyes[j].br(), Scalar(255, 0, 0), 1.5); // draw a rectangle on detected eyes
				std::cout << j << std::endl;
				if (j == 1 && count < 20)
				{
					count += 1;
					Take_Sample_Faces(face_ID, count, grayScale, path); // take samples once all 2 eyes and entire face has been detected
				}
			}
		}

		//show the frame in the created window
		cv::imshow("GrayScale", grayScale);
		cv::imshow(window_name, frame);

		//wait for for 10 ms until any key is pressed.  
		//If the 'Esc' key is pressed, break the while loop.
		//If the any other key is pressed, continue the loop 
		//If any key is not pressed withing 10 ms, continue the loop 
		if (waitKey(10) == 27)
		{
			std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
			break;
		}
	}

	// clear memory
	cap.release();
	destroyAllWindows();

	Train_Data_Set(path, cascade);
}

//  Function to take sample images and store in local directory
void Take_Sample_Faces(int id, int num, Mat img, std::string path)
{
	//  Format for file naming would be : ../././path/ + num + _ + unique ID + .jpg
	imwrite(path + std::to_string(num) + "_" + std::to_string(id) + ".jpg", img);
	std::cout << "Created file at " << path << std::endl;
}

//  Function to call trainer
void Train_Data_Set(const std::string PATH, CascadeClassifier cascade)
{
	std::cout << "Training the AI.." << std::endl;

	Mat image, face;

	std::vector<int> faceIDs;
	std::vector<::Mat> faceSamples;

	// Loop for every images detecting within the directory..
	for (const auto& FILE : std::filesystem::directory_iterator(PATH))
	{
		std::string fileName = FILE.path().filename().string();
		std::cout << "Images in directory: " << fileName << "...\n";

		// Read each image from directory into memory then check if it was successful.
		image = imread(PATH + fileName);
		if (image.empty())
		{
			std::cerr << "Nothing loaded" << std::endl;
		}
		int uniqueID = std::stoi(fileName.substr(fileName.rfind("_") + 1));

		std::cout << uniqueID << std::endl;

		cvtColor(image, face, COLOR_RGB2GRAY); // Convert to grayscale

		std::vector<Rect> faceFeatures;
		cascade.detectMultiScale(face, faceFeatures);
		for (int i = 0; i < faceFeatures.size(); i++)	//  loop for every face in images/frames..
		{
			// Append each detected face into the back of the Face vector array. And append each taken Unique IDs to the back of the ID vector array
			faceSamples.emplace_back(face);
			faceIDs.emplace_back(uniqueID);
		}
	}
	// Begin Training the AI then store on a .yml file in the given directory
	model->train(faceSamples, faceIDs);
	std::cout << "training done" << std::endl;
	model->write("c:/Users/LanceyFreypa/Documents/Trainer/trainer.yml");
	std::cout << "Total Trained images: " << faceSamples.size() << std::endl;

}
