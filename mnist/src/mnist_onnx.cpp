/**
 * @author: strinkin
 * @date: 2025-4-29
 * @description: mnist模型 导入onnx进行序列化 已整合到src/mnist.cpp
*/
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "logging.h"
#include "commom.h"

#include <iostream>
#include <fstream>

using namespace std;

static Logger logger;

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

int main(int argc, char** argv) {
    cout << "start" << endl;
    string onnx_path = "../models/mnist.onnx";
    string save_engine_path = "../models/mnist_onnx.engine";
    SampleUniquePtr<nvinfer1::IBuilder>builder(nvinfer1::createInferBuilder(logger)); // 构建器构建网络和配置
    assert(builder != nullptr);

    auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
        // 0：  
        // 表示不使用任何特殊标志，通常用于创建一个普通的静态网络。
        // 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)：
        // 表示启用显式批量（Explicit Batch）模式。这是TensorRT 7及以上版本引入的一个重要特性，允许网络支持动态输入尺寸。
    SampleUniquePtr<nvinfer1::INetworkDefinition>network(builder->createNetworkV2(flag)); // 创建网络
    assert(network != nullptr);

    SampleUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger)); // 解析器解析onnx模型
    assert(parser != nullptr);

    parser->parseFromFile(onnx_path.c_str(), // onnx模型路径
        static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING)); // 日志级别
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) { // 遍历错误信息
        std::cout << parser->getError(i)->desc() << std::endl; // 打印错误信息
    }

    SampleUniquePtr<nvinfer1::IBuilderConfig>config = setConfig(builder); // 配置
    assert(config != nullptr);

    SampleUniquePtr<nvinfer1::IHostMemory>serialized_model(builder->buildSerializedNetwork(*network, *config)); // 构建序列化模型
    assert(serialized_model != nullptr);

    if (saveEngine(serialized_model, save_engine_path)) {
        cout << "save engine success" << endl;
    } else {
        cout << "save engine failed" << endl;
        return false;
    }
    cout << "end" << endl;
}