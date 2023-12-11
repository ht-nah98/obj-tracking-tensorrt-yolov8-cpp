#include "yolov8.h"
#include "cmd_line_util.h"
#include "trackerv1.cpp"

// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string inputImage;

    EuclideanDistTracker tracker;

    // Parse the command line arguments
	if (!parseArguments(argc, argv, config, onnxModelPath, inputImage)) {
		return -1;
    }

    // Create the YoloV8 engine
    YoloV8 yoloV8(onnxModelPath, config);

    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    const auto objects = yoloV8.detectObjects(img);

    std::cout << "Detected " << objects.size() << " objects" << std::endl;
    std::vector<std::vector<int>> detections;
   // Print details of each detected object
    for (const auto& obj : objects) {
        std::cout << "Object Label: " << obj.label << std::endl;
        std::cout << "Object Probability: " << obj.probability << std::endl;
        std::cout << "Object Rect: " << obj.rect << std::endl;
        std::cout << "Object Rectangle: x=" << obj.rect.x << ", y=" << obj.rect.y
                << ", width=" << obj.rect.width << ", height=" << obj.rect.height << std::endl;
        // Add more prints for other attributes if needed
        std::cout << "------------------------------------" << std::endl;
        int x = obj.rect.x;
        int y = obj.rect.y;
        int width = obj.rect.width;
        int height = obj.rect.height;

        std::vector<int> detection = {x,y,width, height};
        detections.push_back(detection);
    }    
    // Check detections of bbox value
    // for(const auto& bbox: detections){
    //     std::cout << "Bbox: "<< bbox[0] << ", " << bbox[1] << ", " << bbox[2] << ", " << bbox[3] << ")\n";
    // }


    // Use the tracker to update detections
    std::vector<std::vector<int>> updated_objects = tracker.update(detections);
    int thickness = 2;
    // Display updated object information
    for (const auto& obj : updated_objects) {
        std::cout << "Object ID: " << obj[4] << " at (" << obj[0] << ", " << obj[1] << ", " << obj[2] << ", " << obj[3] << ")\n";
        
        int id = obj[4];
        std::string obj_id = std::to_string(id);
        int x = obj[0];
        int y = obj[1];
        int width = obj[2];
        int height = obj[3];

        cv::Point position_text(x,y-15);
        cv::Scalar font_color(200,0,0);
        int font_weight = 2;
        int font_size = 2;

 
        cv::Point p1(x,y);
        cv::Point p2(x+width, y+height);
        
        cv::putText(img, obj_id, position_text, cv::FONT_HERSHEY_SIMPLEX, font_size, font_color, font_weight);
        cv::rectangle(img, p1, p2, cv::Scalar(0,255,0), thickness, cv::LINE_8);
    }

    // Draw the bounding boxes on the image
    // yoloV8.drawObjectLabels(img, objects);

    // Define the start and end points of the line
    cv::Point startPoint(2000, 650);
    cv::Point endPoint(2000, 1600);
    // Define the color of the line (BGR format)
    cv::Scalar color(0, 0, 200); // Red color
    // Draw the line on the image
    cv::line(img, startPoint, endPoint, color, 5);

    int down_width = 640;
    int down_height = 480;
    cv::Mat resize_down;
    cv::resize(img, resize_down, cv::Size(down_width, down_height), cv::INTER_LINEAR);

    cv::imshow("Image", resize_down);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}




