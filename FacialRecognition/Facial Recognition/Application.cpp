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

#include <opencv2/highgui/highgui.hpp>		// OpenCV module for GUI (The use of webcam)
#include <opencv2/imgproc.hpp>				// OpenCV module for Image/Video Processing
#include <opencv2/objdetect.hpp>			// OpenCV module for Object/Facial Detection
#include <opencv2/face.hpp>					// OpenCV module for robust Facial Recognition

#include <Windows.h>						// Controller for windows applications 
#include <tchar.h>							// Internationalization of Unicode

#include <vector>							// std module for better data management
#include <iostream>							// Input/Output for console
#include <fstream>							// Basic File Handling
#include <filesystem>						// Advanced File handling

using namespace cv;
using namespace cv::face;
namespace fs = std::filesystem;

std::string cascadeName, nestedCascadeName;

String face_cascade_dir = "c:/Users/LanceyFreypa/Documents/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_dir = "c:/Users/LanceyFreypa/Documents/Libraries/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

void Run_Webcam(std::string);
void Train_Data_Set(const std::string, CascadeClassifier);
void Recognize_Faces(CascadeClassifier, std::vector<std::string>);
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

int main()
{
	std::cout << "Initializing Necessary Variables and Pre-requisities.." << std::endl;

	const std::string DATASET_DIRECTORY = "c:/Users/LanceyFreypa/Documents/DataSet/";
	const std::string TRAINER_DIRECTORY = "c:/Users/LanceyFreypa/Documents/Trainer/";
	const fs::path DATASET_DIR{ DATASET_DIRECTORY };
	const fs::path TRAINER_DIR{ TRAINER_DIRECTORY };

	//  Check to see if directory exists or not. if it doesn't, create the respective folder
	directory_exists(DATASET_DIR);
	directory_exists(TRAINER_DIR);

	Run_Webcam(DATASET_DIRECTORY);

	return 0;
}

void Run_Webcam(std::string path)
{
	VideoCapture cap(0); // run default camera

	CascadeClassifier cascade, nestedCascade;

	Mat frame;  // A class for the natural videocapture
	Mat grayScale; // A class conversion of the frames into grayscale

	cap >> frame;  // grabs a frame from the videocapture as data for the class frame

	int n = 0;
	
	//  Read file containing lists of names and store them into memory
	std::string name;
	std::vector<std::string> names;
	std::ifstream nameTxt;
	nameTxt.open("c:/Users/LanceyFreypa/Documents/names.txt");
	if (!nameTxt.is_open())
	{
		std::cerr << "No names.txt file detected " << std::endl;
		std::cin.get();
	}
	else
	{
		while (std::getline(nameTxt, name))
		{
			names.emplace_back(name);
		}
	}

	nameTxt.close();
	for (int i = 0; i < names.size(); i++)
	{
		std::cout << names[i] << std::endl;
	}

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

	// Grab details about the camera
	double width = cap.get(CAP_PROP_FRAME_WIDTH);
	double height = cap.get(CAP_PROP_FRAME_HEIGHT);

	std::cout << "Resolution of camera is " << width << " x " << height << std::endl;
	
	// Clear memory
	cap.release();
	cv::destroyAllWindows();

	Train_Data_Set(path, cascade);

	Recognize_Faces(cascade, names);
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


//  Function to call the recognizer algorithm
void Recognize_Faces(CascadeClassifier cascade, std::vector<std::string> names)
{

	// load the recognizer's training file
	model->read("c:/Users/LanceyFreypa/Documents/Trainer/trainer.yml");

	Mat frame, gray, crop;

	VideoCapture cap(0);
	//  check whether the camera is open. if not, alert the user
	if (cap.isOpened() == false)
	{
		std::cerr << "Cannot Open the Camera" << std::endl;
		std::cin.get();
		return;
	}

	cap >> frame;

	//  define methods for minimum window for detecting faces
	//  double minH = 0.1 * cap.get(CAP_PROP_FRAME_HEIGHT);
	//  double minW = 0.1 * cap.get(CAP_PROP_FRAME_WIDTH);

	std::string windowName = "Face Recognizer";
	namedWindow(windowName);
	namedWindow("Camera");

	// Check if nothing was loaded
	if (model.empty())
	{
		std::cerr << "No face Recognizer found" << std::endl;
		std::cin.get();
	}

	while (true)
	{
		std::vector<Rect> faces;

		bool bSuccess = cap.read(frame); // read a new frame from video 

		//Breaking the while loop if the frames cannot be captured
		if (bSuccess == false)
		{
			std::cout << "Video camera is disconnected" << std::endl;
			std::cin.get(); //Wait for any key press
			break;
		}

		// convert to grayscale
		cvtColor(frame, gray, COLOR_RGB2GRAY);

		int predictLabel = -1;
		double confidence = 0.0;
		cascade.detectMultiScale(gray, faces, 1.2, 5);	// Detect fraces from images/frames
		for (int i = 0; i < faces.size(); i++)	// Loop For every face in the image/frame..
		{
			crop = gray(faces[i]);	//  Seperate the detected faces from the inputted images/frames
			if (crop.empty())
			{
				std::cin.get();
				continue;
			}
			else
			{
				imshow(windowName, crop);
			}
			rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(0, 255, 0), 1.5);
			model->predict(crop, predictLabel, confidence);  //  The main method for predicting whose face the inputted frame is from
			if (confidence < 100)  // If the confidence is lower than 100, the prediction could be accurate
			{
				std::cout << predictLabel << std::endl;
				cv::putText(frame, names[predictLabel], cv::Point(10, frame.rows / 2),
					cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 255, 0), 1.2);
				std::cout << "This face belongs to " << names[predictLabel] << std::endl;
			}
			else
			{
				std::cout << "This face belongs to " << names[0] << std::endl;
			}

		}
		imshow("Camera", frame);

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
	cv::destroyAllWindows();
}