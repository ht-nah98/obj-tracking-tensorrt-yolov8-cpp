#include "yolov8.h"
#include "cmd_line_util.h"
#include <opencv2/cudaimgproc.hpp>
#include "trackerv1.cpp"


// Runs object detection on video stream then displays annotated results.
int main(int argc, char* argv[]) {
	std::map<int, std::string> classMap = {
		{0, "motobike"},
		{1, "car"},
	};


    YoloV8Config config;
    std::string onnxModelPath;
    std::string inputVideo;
	EuclideanDistTracker tracker;
	// Parse the command line arguments
	if (!parseArgumentsVideo(argc, argv, config, onnxModelPath, inputVideo)) {
		return -1;
    }

	// Create the YoloV8 engine
	YoloV8 yoloV8(onnxModelPath, config);

	// Initialize the video stream
	cv::VideoCapture cap;

	// Open video capture
	try {
		cap.open(std::stoi(inputVideo));
	} catch (const std::exception& e) {
		cap.open(inputVideo);
	}

	// Try to use HD resolution (or closest resolution)
	auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	// std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	// std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl;

	if (!cap.isOpened())
		throw std::runtime_error("Unable to open video capture with input '" + inputVideo + "'");

	
	// Set up video writer
    const std::string outputVideoPath = "../images/output_video.avi";
    cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(resW, resH));
	int frameCount = 0;
	
	auto start = std::chrono::high_resolution_clock::now();

	// Initialize counter
	std::vector<int> list_moto;
	std::vector<int> list_car;
	while (true) {
		// Grab frame
		cv::Mat img;
		cap >> img;

		if (img.empty())
			throw std::runtime_error("Unable to decode image from video stream.");

		// Extract the ROI from Image
		int roi_x = 100;
		int roi_y = 125;
		int roi_width = 400;
		int roi_height = 200;
		cv::Rect roi_rect(roi_x,roi_y,roi_width,roi_height);
		cv::Mat roi = img(roi_rect);

		// Run inference
		const auto objects = yoloV8.detectObjects(roi);
		

		// std::cout << "Detected " << objects.size() << " objects" << std::endl;
		std::vector<std::vector<int>> detections;
		// Print details of each detected object
		for (const auto& obj : objects) {
			int x = obj.rect.x;
			int y = obj.rect.y;
			int width = obj.rect.width;
			int height = obj.rect.height;
			int label = obj.label;

			std::vector<int> detection = {x,y,width, height, label};
			detections.push_back(detection);
		}    

		// Use the tracker to update detections
		std::vector<std::vector<int>> updated_objects = tracker.update(detections);
		int thickness = 2;
		// Display updated object information
		for (const auto& obj : updated_objects) {
			// std::cout << "Object ID: " << obj[4] << " at (" << obj[0] << ", " << obj[1] << ", " << obj[2] << ", " << obj[3] << ")\n";
			int id = obj[4];		
			int x = obj[0];
			int y = obj[1];
			int width = obj[2];
			int height = obj[3];
			int label = obj[5];
			std::string obj_label = std::to_string(id);
			int cx = (x + x + width) / 2;
            int cy = (y + y + height) / 2;

			// Draw rectangle in ROI 
			cv::Point position_text(x,y-15);
			int font_weight = 2;
			float font_size = 0.5f;
			cv::Point p1(x,y);
			cv::Point p2(x+width, y+height);

			// Check draw for Moto or Car
			if(label == 0){
				cv::circle(roi, cv::Point(cx,cy), 3, cv::Scalar(150,150,0),-1);
				cv::putText(roi, obj_label, position_text, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(150,150,0), font_weight);
				cv::rectangle(roi, p1, p2, cv::Scalar(150,150,0), thickness, cv::LINE_8);
			}
			else{
				cv::circle(roi, cv::Point(cx,cy), 3, cv::Scalar(0,150,150),-1);
				cv::putText(roi, obj_label, position_text, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(0,150,150), font_weight);
				cv::rectangle(roi, p1, p2, cv::Scalar(0,150,150), thickness, cv::LINE_8);	
			}

			// Define the line position
			cv::Point startPoint(300, 5);
			cv::Point endPoint(300, 250);
			// Define the color of the line (BGR format)
			cv::Scalar color(0, 0, 200); // Red color
			// Draw the line on the image
			cv::line(roi, startPoint, endPoint, color, 2);

			if (cx > 295 && cx < 305 && cy > 5 && cy < 250 && label == 0) {
				// Check if id is in the vector
				auto it = std::find(list_moto.begin(), list_moto.end(), id);
				// If can not find the id tracker, add into list
				if (it == list_moto.end()) {
					// std::cout << "Found " << id << " at position " << std::distance(list_moto.begin(), it) << std::endl;
					list_moto.push_back(id);
				}	
			}

			if (cx > 295 && cx < 305 && cy > 5 && cy < 250 && label == 1) {
				// Check if id is in the vector
				auto it = std::find(list_car.begin(), list_car.end(), id);
				if (it == list_car.end()) {
					// Element not found
					// std::cout << "Found " << id << " at position " << std::distance(list_moto.begin(), it) << std::endl;
					list_car.push_back(id);
				}	
			}
			
		}
		
		// Display ROI
		cv::imshow("ROI", roi);
		
		// Draw the bounding boxes on the image
		// yoloV8.drawObjectLabels(img, objects);

        frameCount++;

		// Display the FPS on the video frame
        double elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
        double fps = frameCount / elapsedSeconds;
        std::ostringstream fpsText;
        fpsText << "FPS: " << static_cast<int>(fps);
        cv::putText(img, fpsText.str(), cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

		// Display the results
		int font_size = 1;
		int font_weight = 2;
		cv::Point position_moto(50,50);
		cv::Point position_car(200,50);
		int total_moto = list_moto.size();
		std::string total_moto_str = std::to_string(total_moto);
		int total_car = list_car.size();
		std::string total_car_str = std::to_string(total_car);
		cv::putText(img, "Moto"+total_moto_str, position_moto, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(150,150,0), font_weight);
		cv::putText(img, "Car"+total_car_str, position_car, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(0,150,150), font_weight);
		cv::imshow("Object Detection", img);

		// Write frame to video
        videoWriter.write(img);

		// cv::waitKey(0);
		if (cv::waitKey(20) >= 0)
			break;
	}
	// Release the video writer
    videoWriter.release();
	return 0;
}