/**
 * @author: strinkin
 * @date: 2025-3-23
 * @description: mnist模型 定义 序列化 反序列化 推理
*/
#include <iostream>
#include <map>
#include <fstream>
#include <cassert>
#include <vector>
#include <algorithm> // 包含std::max_element

#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "logging.h"
#include "commom.h"
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int INPUT_C = 1;
static const int OUTPUT_SIZE = 10;

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace std;
using namespace cv;

static Logger logger;

map<string, nvinfer1::Weights> readWtsFromFile(const string& wts_path) {
    ifstream wts_file(wts_path, ios::in);
    assert(wts_file.is_open() && "unable to read weight file");
    map<string, nvinfer1::Weights> weights_map;
    uint32_t num_weights = 0; // wts行数
    wts_file >> dec >> num_weights;
    while(num_weights--) {
        string weight_name; // 该网络层名称
        int64_t weight_count; // 该网络层权重参数数量
        wts_file >> weight_name >> dec >> weight_count;
        uint32_t* weight_value = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * weight_count)); // 该网络层权重参数值
        cout << "weight name: " << weight_name << " \nweight count: " << weight_count << endl;
        // read weight values
        for (size_t i = 0; i < weight_count; i++) {
            wts_file >> hex >> weight_value[i];
            // cout << hex << weight_value[i] << " ";
        }
        cout << dec << endl;
        nvinfer1::Weights weight = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, weight_value, weight_count};
        weights_map[weight_name] = weight;
    }
    return weights_map;
}

SampleUniquePtr<nvinfer1::INetworkDefinition> defineNetwork(map<string, nvinfer1::Weights>& weights_map, SampleUniquePtr<nvinfer1::IBuilder>& builder) {
    auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
                   // 0：
                   // 表示不使用任何特殊标志，通常用于创建一个普通的静态网络。
                   // 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)：
                   // 表示启用显式批量（Explicit Batch）模式。这是TensorRT 7及以上版本引入的一个重要特性，允许网络支持动态输入尺寸。
    SampleUniquePtr<nvinfer1::INetworkDefinition>network(builder->createNetworkV2(flag)); // 创建网络
    nvinfer1::ITensor* data = network->addInput(INPUT_BLOB_NAME, nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, INPUT_C, INPUT_H, INPUT_W}); // 定义输入数据格式
    assert(data);
    // conv1 layer
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 32, nvinfer1::DimsHW{3, 3}, weights_map["conv1.weight"], weights_map["conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(nvinfer1::DimsHW{1, 1}); // 步长1
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1}); // 填充1
    // relu1 layer
    nvinfer1::IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);
    // maxpool1 layer
    nvinfer1::IPoolingLayer* maxpool1 = network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    assert(maxpool1);
    maxpool1->setStrideNd(nvinfer1::DimsHW{2, 2}); // 步长2
    // conv2 layer
    nvinfer1::IConvolutionLayer* conv2 = network->addConvolutionNd(*maxpool1->getOutput(0), 64, nvinfer1::DimsHW{3, 3}, weights_map["conv2.weight"], weights_map["conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(nvinfer1::DimsHW{1, 1}); // 步长1
    conv2->setPaddingNd(nvinfer1::DimsHW{1, 1}); // 填充1
    // relu2 layer
    nvinfer1::IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu2);
    // maxpool2 layer
    nvinfer1::IPoolingLayer* maxpool2 = network->addPoolingNd(*relu2->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    assert(maxpool2);
    maxpool2->setStrideNd(nvinfer1::DimsHW{2, 2}); // 步长2
    {/* 全连接层 trt 8.4 弃用该方法
    // fc1 layer
    nvinfer1::IFullyConnectedLayer* fc1 = network->addFullyConnected(*maxpool2->getOutput(0), 128, weights_map["linear1.weight"], weights_map["linear1.bias"]);
    assert(fc1);
    // fc2 layer
    nvinfer1::IFullyConnectedLayer* fc2 = network->addFullyConnected(*fc1->getOutput(0), 10, weights_map["linear2.weight"], weights_map["linear2.bias"]);
    assert(fc2);
    */}

    /* 全连接层 官方示例
    // int32_t const batch = maxpool2->getOutput(0)->getDimensions().d[0];
    // int32_t const mmInputs = maxpool2->getOutput(0)->getDimensions().d[1] * maxpool2->getOutput(0)->getDimensions().d[2] * maxpool2->getOutput(0)->getDimensions().d[3]; // channels*h*w=64*7*7
    // auto inputReshape = network->addShuffle(*maxpool2->getOutput(0));
    // inputReshape->setReshapeDimensions(nvinfer1::Dims{2, {batch, mmInputs}});
    // uint32_t nbOutputs = 128;
    // nvinfer1::IConstantLayer* filterConst = network->addConstant(nvinfer1::Dims{2, {nbOutputs, mmInputs}}, weights_map["linear1.weight"]);
    // nvinfer1::IMatrixMultiplyLayer* mm = network->addMatrixMultiply(*inputReshape->getOutput(0), nvinfer1::MatrixOperation::kNONE, *filterConst->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
    // auto biasConst = network->addConstant(nvinfer1::Dims{2, {1, nbOutputs}}, mWeightMap["linear1.bias"]);
    // auto biasAdd = network->addElementWise(*mm->getOutput(0), *biasConst->getOutput(0), ElementWiseOperation::kSUM);
    */
    
    // flatten layer
    nvinfer1::IShuffleLayer* flatten = network->addShuffle(*maxpool2->getOutput(0));
    assert(flatten);
    flatten->setReshapeDimensions(nvinfer1::Dims{2, {1, 7 * 7 * 64}}); // {dim, {batch, channels*h*w}} 等价Dims2{1, 7 * 7 * 64}
    
    // fc1 layer
    // weight tensor
    nvinfer1::IConstantLayer* weight_layer = network->addConstant(nvinfer1::Dims{2, {128, 7 * 7 * 64}}, weights_map["linear1.weight"]);
    nvinfer1::ITensor* weight_tensor = weight_layer->getOutput(0);
    // tensor multiply {2, {batch, 128}} * {2, {7 * 7 * 64, 128}} = {2, {batch, 128}}
    nvinfer1::IMatrixMultiplyLayer* mat_multi = network->addMatrixMultiply(*flatten->getOutput(0), nvinfer1::MatrixOperation::kNONE, *weight_tensor, nvinfer1::MatrixOperation::kTRANSPOSE);
    assert(mat_multi);
    // bias tensor {2, {batch, 128}} + {2, {1, 128}} = {2, {batch, 128}}
    nvinfer1::IConstantLayer* bias_layer = network->addConstant(nvinfer1::Dims{2, {1, 128}}, weights_map["linear1.bias"]);
    nvinfer1::ITensor* bias_tensor = bias_layer->getOutput(0);
    // add bias
    nvinfer1::IElementWiseLayer* add = network->addElementWise(*mat_multi->getOutput(0), *bias_tensor, nvinfer1::ElementWiseOperation::kSUM);
    assert(add);

    // fc2 layer {batch, output_num} {H, W}
    // weight and bias tensor
    weight_layer = network->addConstant(nvinfer1::Dims{2, {10, 128}}, weights_map["linear2.weight"]);
    weight_tensor = weight_layer->getOutput(0);
    bias_layer = network->addConstant(nvinfer1::Dims{2, {1, 10}}, weights_map["linear2.bias"]);
    bias_tensor = bias_layer->getOutput(0);
    // tensor multiply
    mat_multi = network->addMatrixMultiply(*add->getOutput(0), nvinfer1::MatrixOperation::kNONE, *weight_tensor, nvinfer1::MatrixOperation::kTRANSPOSE);
    assert(mat_multi);
    // add bias
    add = network->addElementWise(*mat_multi->getOutput(0), *bias_tensor, nvinfer1::ElementWiseOperation::kSUM);
    assert(add);

    // change dimensions
    // nvinfer1::IShuffleLayer* reshape = network->addShuffle(*add->getOutput(0));
    // assert(reshape);
    // reshape->setReshapeDimensions(nvinfer1::Dims{2, {1, 10}}); 

    // softmax layer
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*add->getOutput(0));
    softmax->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    assert(softmax);
    softmax->setAxes(1<<1); // softmax要处理的维度的索引，例如{N, C, H, W}中，N=1<<0, C=1<<1, H=1<<2, W=1<<3. 这里上一层的输出是{batch, output_size}，所以取索引1,即1<<1
    network->markOutput(*softmax->getOutput(0));
    return network;
}

SampleUniquePtr<nvinfer1::IBuilderConfig> setConfig(SampleUniquePtr<nvinfer1::IBuilder>& builder) {
    SampleUniquePtr<nvinfer1::IBuilderConfig>config(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(64U << 20); // 64MB 构建引擎时分配的最大工作空间
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20); // 1MB 运行时分配的最大显存工作空间
    // config->setFlag(nvinfer1::BuilderFlag::kFP16); // 设置为FP16精度
    return config;
}

bool saveEngine(SampleUniquePtr<nvinfer1::IHostMemory>& serialized_model, const string& save_engine_path) {
    std::ifstream check(save_engine_path); // 检查文件是否存在
    if (check) {
        check.close();
        std::string new_file_path = save_engine_path + ".new";
        std::ofstream p(new_file_path, std::ios::binary);
        if (!p.is_open()) {
            std::cerr << "Failed to open new file: " << new_file_path << std::endl;
            return false;
        }
        std::cout << "engine exist, Writing to new file: " << new_file_path << std::endl;
        p.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
        p.close();
    } else {
        std::ofstream p(save_engine_path, std::ios::binary);
        if (!p.is_open()) {
            std::cerr << "Failed to open file: " << save_engine_path << std::endl;
            return false;
        }
        std::cout << "Writing to file: " << save_engine_path << std::endl;
        p.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
        p.close();
    }
    return true;
}

bool serializeEngine(const string& wts_file_path, const string& save_engine_path="default_name.engine") {
    map<string, nvinfer1::Weights> weights_map = readWtsFromFile(wts_file_path);
    SampleUniquePtr<nvinfer1::IBuilder>builder(nvinfer1::createInferBuilder(logger)); // 构建器构建网络和配置
    SampleUniquePtr<nvinfer1::INetworkDefinition>network = defineNetwork(weights_map, builder); // 定义网络
    SampleUniquePtr<nvinfer1::IBuilderConfig>config = setConfig(builder); // 配置
    SampleUniquePtr<nvinfer1::IHostMemory>serialized_model(builder->buildSerializedNetwork(*network, *config)); // 构建序列化模型
    assert(serialized_model != nullptr);
    // Release host memory
    for (auto& mem : weights_map)
    {
        free((void*) (mem.second.values)); // first是key，second是value
    }
    if (saveEngine(serialized_model, save_engine_path)) {
        cout << "save engine success" << endl;
    } else {
        cout << "save engine failed" << endl;
        return false;
    }

    return true;
}

std::vector<char> readModelFromFile(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate); // 打开文件，以二进制模式并定位到文件末尾
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + engine_path);
    }

    std::streamsize file_size = file.tellg(); // 获取文件大小
    file.seekg(0, std::ios::beg); // 定位到文件开头

    std::vector<char> model_data(file_size); // 创建一个足够大的向量来存储文件内容
    if (!file.read(model_data.data(), file_size)) { // 读取文件内容
        throw std::runtime_error("Failed to read file: " + engine_path);
    }

    return model_data;
}

cv::Mat resizeAndPad(const cv::Mat& src, int targetSize) {
    // 获取原始图像的宽高
    int srcWidth = src.cols;
    int srcHeight = src.rows;

    // 计算长边和短边
    int longSide = std::max(srcWidth, srcHeight);
    int shortSide = std::min(srcWidth, srcHeight);

    // 计算缩放比例
    float scale = static_cast<float>(targetSize) / longSide;

    // 计算缩放后的宽高
    int newWidth = static_cast<int>(srcWidth * scale);
    int newHeight = static_cast<int>(srcHeight * scale);

    // 缩放图像
    cv::Mat resizedImage;
    cv::resize(src, resizedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);

    // 创建一个目标大小的黑色背景图像
    cv::Mat paddedImage = cv::Mat::zeros(targetSize, targetSize, src.type());

    // 计算填充的起始位置
    int startX = (targetSize - newWidth) / 2;
    int startY = (targetSize - newHeight) / 2;

    // 将缩放后的图像复制到黑色背景图像的中心位置
    resizedImage.copyTo(paddedImage(cv::Rect(startX, startY, newWidth, newHeight)));

    return paddedImage;
}

int main(const int argc, const char** argv) {
    string wts_file_path = "../models/mnist.wts";
    string save_engine_path = "../models/mnist.engine";
    // if(!serializeEngine(wts_file_path, save_engine_path)) { return -1; };
    // if(0) 
    if(1) 
    {
        string engine_path = "../models/mnist.engine";
        SampleUniquePtr<nvinfer1::IRuntime>runtime(nvinfer1::createInferRuntime(logger)); // 运行时记录器
        assert(runtime != nullptr);
        std::vector<char> model_data = readModelFromFile(engine_path);
        SampleUniquePtr<nvinfer1::ICudaEngine>engine(runtime->deserializeCudaEngine(model_data.data(), model_data.size(), nullptr));
        assert(engine != nullptr);
        SampleUniquePtr<nvinfer1::IExecutionContext>context(engine->createExecutionContext());
        assert(context != nullptr);

        assert(engine->getNbIOTensors() == 2); // 检查输入和输出张量的数量
        void* device_buffers[2] = {nullptr, nullptr}; // {input_buffer, output_buffer}
        CHECK(cudaMalloc(&device_buffers[0], sizeof(float) * INPUT_C * INPUT_H * INPUT_W));
        CHECK(cudaMalloc(&device_buffers[1], sizeof(float) * OUTPUT_SIZE));
        context->setTensorAddress(INPUT_BLOB_NAME, device_buffers[0]);
        context->setTensorAddress(OUTPUT_BLOB_NAME, device_buffers[1]);
        nvinfer1::Dims input_dims = {4, {1, INPUT_C, INPUT_H, INPUT_W}};
        context->setInputShape(INPUT_BLOB_NAME, input_dims);
        
        Mat img_src = imread("../pic/num3.png", IMREAD_GRAYSCALE);
        Mat img = resizeAndPad(img_src, INPUT_H);
        imshow("img_src", img_src);
        float data[INPUT_C * INPUT_H * INPUT_W];
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[i] = img.data[i] / 255.0;
        }

        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        CHECK(cudaMemcpy(device_buffers[0], data, sizeof(float) * INPUT_C * INPUT_H * INPUT_W, cudaMemcpyHostToDevice));
        // CHECK(cudaMemcpy2D((void*)device_buffers[0], sizeof(float) * INPUT_W, data, sizeof(float) * INPUT_W, sizeof(float) * INPUT_W, sizeof(float) * INPUT_H, cudaMemcpyHostToDevice));
        context->enqueueV3(stream);
        float prob[OUTPUT_SIZE] = {0};
        CHECK(cudaMemcpy(prob, device_buffers[1], sizeof(float) * OUTPUT_SIZE, cudaMemcpyDeviceToHost));

        // Print histogram of the output distribution
        std::cout << "\nOutput:\n";
        for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
        {
            std::cout << prob[i] << " ";
        }
        std::cout << std::endl;
        cout << max_element(prob, prob + OUTPUT_SIZE) - prob << endl;
        cudaStreamDestroy(stream);
        CHECK(cudaFree(device_buffers[0]));
        CHECK(cudaFree(device_buffers[1]));
        waitKey(0);
    }
    return 0;
}