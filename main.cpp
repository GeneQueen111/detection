#include <iostream>
#include "detection.h"
#include <thread>
#include <vector>
#include <functional>
#include "timeCounter.hpp"
#include <random>  // C++11的随机数库
#include <ctime> 
#include <opencv2/opencv.hpp>
#include <string>


int main(int, char**)
{
    string model_path = "/Users/linzenggeng/Desktop/detection/model/new.onnx";
    detection::DetectionArmor detectionArmor(model_path, true);

    detectionArmor.start_detection();
    detectionArmor.showImage();

    return 0;
}
