#ifndef __DETECTION_HPP__
#define __DETECTION_HPP__

#include "detection.h"
// #include "STrack.h"
#include <thread>    // std::this_thread
#include <chrono>    // std::chrono::seconds
#include <system_error>
#include <pthread.h>

using namespace detection;

bool setThreadPriority(std::thread& thread, int priority) {
    pthread_t pthread = thread.native_handle();
    
    // 获取当前调度策略
    int policy;
    struct sched_param param;
    if (pthread_getschedparam(pthread, &policy, &param) != 0) {
        std::cerr << "获取线程调度参数失败" << std::endl;
        return false;
    }
    
    // 设置新优先级（macOS 和 Linux 通用）
    // 注意：优先级范围通常为 1-99，值越大优先级越高
    param.sched_priority = priority;
    
    // 应用新参数（使用 SCHED_RR 实时调度策略）
    if (pthread_setschedparam(pthread, SCHED_RR, &param) != 0) {
        std::cerr << "设置线程优先级失败，可能需要 root 权限" << std::endl;
        return false;
    }
    
    return true;
}

DetectionArmor::DetectionArmor(string& model_path, bool ifcountTime, string video_path)
    : ifCountTime(ifcountTime)
{
    cap = VideoCapture(video_path);

    ov::AnyMap config = {
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}, // 设置性能模式为延迟优化
        {ov::inference_num_threads(4)}, // 使用4个线程进行推理
        {ov::num_streams(1)}, // 允许同时执行1个推理流
        {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY)}, // 性能核心绑定
        {ov::hint::enable_hyper_threading(false)}, // 关闭超线程
        {ov::hint::enable_cpu_pinning(false)} // 关闭CPU固定
    };

    auto network = core.read_model(model_path);
    compiled = core.compile_model(network, "CPU", config);
    infer_request = compiled.create_infer_request();
    input_port = compiled.input();

    input_blob = Mat(640, 640, CV_32F, Scalar(0)); // 初始化输入blob

    tracker = BYTETracker(10, 10); // 初始化BYTETracker

    // armorsDatas = new ArmorData[20]; // 最多装20个装甲板
}

DetectionArmor::~DetectionArmor() 
{
    // std::cout << "quit from detection" << std::endl;
    clearHeap();
}

void DetectionArmor::clearHeap()
{
    cap.release();
    cv::destroyAllWindows();
}

void DetectionArmor::drawObject(Mat& image, const ArmorData& d)
{
    // 绘制装甲板的边界框
    float s_x = d.center_x - d.length / 2.0;
    float s_y = d.center_y - d.width / 2.0;
    float e_x = d.center_x + d.length / 2.0; 
    float e_y = d.center_y + d.width / 2.0;
    cv::rectangle(
        image, 
        Point(s_x, s_y), 
        Point(e_x, e_y), 
        Scalar(0, 0, 255),
        3
    );
}

inline double DetectionArmor::sigmoid(double x) 
{
    return (1 / (1 + exp(-x)));
}

void DetectionArmor::run()
{
    size_t frame_count = 0;

    while (1) 
    {
        armorsDatas.clear(); // 清空当前帧的装甲板数据
        cap >> frame;

        resize(frame, img, Size(640, 640));
        
        // 推理
        infer();

        frame_count += 1;
        if (frame_count == int(cap.get(cv::CAP_PROP_FRAME_COUNT)))
        {
            frame_count = 0;
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        }

        showImage();

        if (cv::waitKey(1) == 27)
        {
            isRunning = false; // 设置线程停止标志
            clearHeap();
            break;
        } // 按下ESC键退出

        // cout << "Detected armors: " << getdata().size() << endl;

        // for (auto i : getdata()) 
        // {
        //     drawObject(img, i); // 绘制检测结果
        // }
        // imshow("Detection", img); // 显示图像

        // if (cv::waitKey(20) == 27)
        // {
        //     isRunning = false; // 设置线程停止标志
        //     break;
        // } // 按下ESC键退出
    }
}

void DetectionArmor::infer()
{

    Timer t(counter);

    // 归一化
    input_blob = blobFromImage(
        img, 
        1 / 255.0
    );

    // 固定八股
    Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), input_blob.data);
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();    
    auto outputs = compiled.outputs();
    Tensor output = infer_request.get_tensor(outputs[0]);
    ov::Shape output_shape = output.get_shape();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());
    
    float conf_threshold = 0.75;  // 置信度阈值
    float nms_threshold = 0.45;   // NMS重叠阈值
    
    // 存储临时结果
    std::vector<cv::Rect> boxes;
    std::vector<int> num_class;
    std::vector<int> color_class;
    std::vector<float> confidences;     
    std::vector<int> indices;

    // 遍历所有的网络输出
    for (int i = 0; i < output_buffer.rows; ++i) 
    {
        // 获取当前的置信度
        float confidence = output_buffer.at<float>(i, 8);

        // 激活到0-1之间
        confidence = sigmoid(confidence);

        // 过滤低置信度检测框
        if (confidence < conf_threshold) continue;  

        // 检查出颜色和数字的类别
        cv::Mat color_scores = output_buffer.row(i).colRange(9, 13);   // 颜色概率（红/蓝等）
        cv::Mat classes_scores = output_buffer.row(i).colRange(13, 22);// 类别概率
        cv::Point class_id, color_id;
        cv::minMaxLoc(classes_scores, nullptr, nullptr, nullptr, &class_id);
        cv::minMaxLoc(color_scores, nullptr, nullptr, nullptr, &color_id);
        // 加入预测出来的数字和颜色
        num_class.push_back(class_id.x);
        color_class.push_back(color_id.x);

        // 检测颜色
        if ((detect_color == 0 && color_id.x == 1) || (detect_color == 1 && color_id.x == 0)) continue;
        
        // 获取第一个输出向量的指针
        float* f_ptr = output_buffer.ptr<float>(i);
        // 创建rect
        cv::Rect rect(
            f_ptr[0], // x
            f_ptr[1], // y
            f_ptr[4] - f_ptr[0], // width
            f_ptr[5] - f_ptr[1]  // height
        );
        
        // 加入
        boxes.push_back(rect);
        confidences.push_back(confidence);
    }

    // 非什么几把极大值抑制
    cv::dnn::NMSBoxes(
        boxes,                // 输入边界框（std::vector<cv::Rect>）
        confidences,          // 输入置信度（std::vector<float>）
        conf_threshold,       // 得分阈值（如 0.5f）
        nms_threshold,        // NMS 阈值（如 0.4f）
        indices               // 输出索引（必须传入引用）
    );

    // cout << "NMS indices size: " << indices.size() << endl;

    // 保留最终的数据
    std::vector<ArmorData> data;
    for (int valid_index = 0; valid_index < indices.size(); ++valid_index) 
    {
        ArmorData d;
        d.center_x = boxes[indices[valid_index]].x + boxes[indices[valid_index]].width / 2.0;
        d.center_y = boxes[indices[valid_index]].y + boxes[indices[valid_index]].height / 2.0;
        d.length = boxes[indices[valid_index]].width;
        d.width = boxes[indices[valid_index]].height;
        d.ID = num_class[indices[valid_index]];
        d.color = color_class[indices[valid_index]];

        armorsDatas.push_back(d);
    }
}

inline const vector<ArmorData> DetectionArmor::getdata()
{
    return armorsDatas; // 返回当前帧的装甲板数据
}

void DetectionArmor::start_detection()
{
    this->isRunning = true; // 设置线程运行标志
    run();

    // run(); // 直接调用run函数
}

void __TEST__ DetectionArmor::showImage()
{
    if (!img.empty()) 
    {
        for (auto i : getdata()) 
        {
            drawObject(img, i); // 绘制检测结果
        }
        cv::imshow("Detection Armor", img); // 显示图像
        // format_print_data_test();
    }

    // std::lock_guard<std::mutex> lock(_mtx);
}

void __TEST__ DetectionArmor::format_print_data_test()
{
    cout << "armor Num: " << getdata().size() << endl;
    for (auto d : getdata())
    {
        cout << "center X: " << d.center_x << " ";
        cout << "center Y: " << d.center_y << endl;
    }
}

#endif // __DETECTION_HPP__