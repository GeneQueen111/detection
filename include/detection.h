#ifndef __DETECTION_H__
#define __DETECTION_H__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <thread>
#include <atomic>
#include "timeCounter.hpp"
#include <shared_mutex>
#include "BYTETracker.h"

#define __TEST__

using namespace std;
using namespace cv;             
using namespace ov;
using namespace cv::dnn;

namespace detection
{
    typedef struct ArmorData
    {
        float center_x; //中心点x
        float center_y; //中心点y

        float length;   //长度
        float width;    //宽度

        int ID = 0;
        int color = 0;

    } ArmorData;

    class DetectionArmor 
    {

    // 一些基本的量
    Core core;
    VideoCapture cap;
    CompiledModel compiled;
    InferRequest infer_request;
    ov::Output<const ov::Node> input_port;

    Mat frame;
    Mat img;
    Mat input_blob;

    // 一个优雅又不太优雅的计时器
    timeCounter counter = timeCounter("run a frame");
    bool ifCountTime = false; // 是否计时

    // 指定识别的颜色
    int detect_color = 0; // 0: 红色，1: 蓝色

    // 当前帧的装甲板数据
    vector<ArmorData> armorsDatas; //

    // 整个推理的线程
    thread infer_thread;

    // 一个用来保护识别到的装甲板的数据的锁 （不知道要不要用到）
    std::mutex _mtx;
    
    // 控制run线程的原子变量
    std::atomic<bool> isRunning{false};

    //数据就绪状态
    std::atomic<bool> data_ready{false};

    public:
        
        DetectionArmor() = default; //默认构造函数
        DetectionArmor(const DetectionArmor&) = delete; // 禁止拷贝
        DetectionArmor(string& model_path, bool ifCountTime=false, string video_path="/Users/linzenggeng/Desktop/detection/video/2.mp4");
        ~DetectionArmor();

        BYTETracker tracker = BYTETracker(10, 10); // 初始化BYTETracker

        static void drawObject(Mat& image, const ArmorData& d);
        static double sigmoid(double x);

        void start_detection();

        // 外部线程可以开一个循环连续调用此成员函数以实时获取实时装甲板的检测结果
        const vector<ArmorData> getdata();

        void __TEST__ format_print_data_test();

        // 开启识别
        void run();

        // 显示图像
        void __TEST__ showImage();

    private:
        void infer();

        void clearHeap();
    };

}  // namespace detection

#include "detection.hpp"

#endif