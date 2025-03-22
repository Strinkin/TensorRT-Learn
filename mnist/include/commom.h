/**
 * @author: strinkin
 * @date: 2025-3-23
 * @description: TensorRT相关类的智能指针Deleter(高版本TensorRT可能不需要); CUDA错误检查宏
*/

struct TrtDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, TrtDeleter>;

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %d: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)