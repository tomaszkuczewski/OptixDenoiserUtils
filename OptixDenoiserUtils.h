//Optix 7.0.0 Denoiser Implementation
//Tomasz Kuczewski, 2019

#ifndef OPTIX_DENOISER_UTILS
#define OPTIX_DENOISER_UTILS

#ifdef _MSC_VER
#pragma comment(lib, "cudart_static.lib")
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <lodepng.h>
#include <tinyexr.h>

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw sutil::Exception( res, ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw sutil::Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )


using byte = unsigned char;
using uint = unsigned int;

using decode_function = std::function<void(const std::vector<byte>&, std::vector<float>&, unsigned int&, unsigned int&)>;
using encode_function = std::function<void(const std::vector<float>&, std::vector<byte>&, const unsigned int&, const unsigned int&)>;

namespace sutil
{
	class Exception : public std::runtime_error
	{
	public:
		Exception(const char* msg)
			: std::runtime_error(msg)
		{ }

		Exception(OptixResult res, const char* msg)
			: std::runtime_error(createMessage(res, msg).c_str())
		{ }

	private:
		std::string createMessage(OptixResult res, const char* msg)
		{
			std::ostringstream out;
			out << optixGetErrorName(res) << ": " << msg;
			return out.str();
		}
	};
}

static inline void LodePngWrapperDecoder(const std::vector<byte>& input, std::vector<float>& output, unsigned int& width, unsigned int& height)
{
	std::vector<byte> rawOutput{};
	auto lodeState = lodepng::State{};
	if (lodepng::decode(rawOutput, width, height, lodeState, input))
		throw std::exception("Could not load this file with LodePNG");

	output.resize(rawOutput.size());
	for (uint i = 0; i < rawOutput.size(); ++i)
	*(((float*)output.data()) + i) = *(rawOutput.data() + i) / 255.0f;  //To minimize access times
}

static inline void LodePngWrapperEncoder(const std::vector<float>& input, std::vector<byte>& output, const unsigned int& width, const unsigned int& height)
{
	std::vector<byte> rawData{};
	rawData.resize(input.size());
	for (uint i = 0; i < input.size(); ++i)
	{
		//To minimize access times
		float bytePixel = *(((float*)input.data()) + i) * 255.0f;
		*(rawData.data() + i) = (unsigned char)(bytePixel > 255.0f ? 255.0f : bytePixel);
	}

	auto lodeState = lodepng::State{};
	if (lodepng::encode(output, rawData, width, height))
		throw std::exception("Could not load this file with LodePNG");
}

static inline void ExrWrapperDecoder(const std::vector<byte>& input, std::vector<float>& output, unsigned int& width, unsigned int& height)
{
	float* dataVector;
	const char* error;
	if (LoadEXRFromMemory(&dataVector, reinterpret_cast<int*>(&width), reinterpret_cast<int*>(&height), input.data(), input.size(), &error))
		throw std::exception("Could not load this file with TinyEXR");

	uint imageLength = width * height * 4;
	output.resize(imageLength);
	memcpy((void*)output.data(), (void*)dataVector, imageLength);

	//uint imageLength = width * height * 4;
	//auto maxPixelValue = std::max_element(dataVector, dataVector + imageLength);
	//output.resize(imageLength);
	//for (uint i = 0; i < imageLength; i+=4)
	//{
	//	output[i] = dataVector[i] / *maxPixelValue;
	//	output[i+1] = dataVector[i+1] / *maxPixelValue;
	//	output[i+2] = dataVector[i+2] / *maxPixelValue;
	//	output[i + 3] = dataVector[i + 3];
	//}
}

enum class OptixImageType
{
	LDR,
	HDR
};

class OptixDenoiserUtils
{
private:
	CUcontext _mCContext;
	OptixDeviceContext _mODeviceContext;
	OptixDeviceContextOptions _mOOptions;

	OptixDenoiser _mODenoiser;
	OptixDenoiserOptions _mODenoiserOptions;
	OptixDenoiserSizes _mOSizes;

	CUstream _mCStream;
	CUdeviceptr _mCDenoiserState;
	CUdeviceptr _mCScratch;
	CUdeviceptr _mCIntensityBuffer;

	uint _mWidth;
	uint _mHeight;

	bool _mCleared;

	std::vector<float> _mFBufferRBG;
	std::vector<float> _mFBufferAlbedo;
	std::vector<float> _mFBufferNormal;
	std::vector<float> _mFBufferDenoised;

	inline void				ReadImage(std::string fileName, std::vector<float>& imageData, uint& width, uint& height) const;

	inline decode_function	GetDecoder(std::string fileName) const;
	inline encode_function  GetEncoder(std::string fileName) const;

	inline void				GenerateOptixInputBuffer(const std::vector<float>& buffer, OptixImage2D& output) const;
	inline void				GenerateOptixOutputBuffer(size_t size, OptixImage2D& output) const;
public:
	inline OptixDenoiserUtils() :
		_mCContext{ 0 },
		_mODeviceContext{}, 
		_mOOptions{ NULL, nullptr, 4 }, //Log level doesn't matter because I don't use log function
		_mODenoiser{},
		_mODenoiserOptions{},
		_mOSizes{},
		_mCStream{},
		_mCDenoiserState{},
		_mCScratch{},
		_mWidth{ 0 },
		_mHeight{ 0 },
		_mCleared{false}
	{ ; }

	inline ~OptixDenoiserUtils() { Clear(); }

	inline void				Init();
	inline void				CreateDenoiser(OptixImageType type);
	inline void				Denoise(float blendFactor, unsigned int denoiseAlpha, bool precalculateIntenstity);
	inline void				SetRGB(const char* fileName);
	inline void				SetAlbedo(const char* fileName);
	inline void				SetNormal(const char* fileName);
	inline void				SaveImage(std::string fileName) const;
	inline void				Clear();

};

inline decode_function OptixDenoiserUtils::GetDecoder(std::string fileName) const
{
	auto extention = fileName.substr(fileName.find_first_of('.') + 1);
	if (_stricmp(extention.c_str(), "png") == 0)
		return LodePngWrapperDecoder;	
	else if(_stricmp(extention.c_str(), "exr") == 0)
		return ExrWrapperDecoder;

	return 0;
}

inline encode_function OptixDenoiserUtils::GetEncoder(std::string fileName) const
{
	auto extention = fileName.substr(fileName.find_first_of('.') + 1);
	if (_stricmp(extention.c_str(), ".png"))
		return LodePngWrapperEncoder;

	return 0;
}

inline void OptixDenoiserUtils::GenerateOptixInputBuffer(const std::vector<float>& buffer, OptixImage2D& output) const
{
	//Preallocate buffer in GPU
	CUdeviceptr gpuImage{};
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuImage), _mFBufferRBG.size() * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(gpuImage), (const void*)_mFBufferRBG.data(), _mFBufferRBG.size() * sizeof(float), cudaMemcpyHostToDevice));
	
	output.data = gpuImage;
	output.format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4;
	output.width = _mWidth;
	output.height = _mHeight;
	output.pixelStrideInBytes = 16;
	output.rowStrideInBytes = _mWidth * 16;
}

inline void OptixDenoiserUtils::GenerateOptixOutputBuffer(size_t size, OptixImage2D& output) const
{
	CUdeviceptr gpuImage{};
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuImage), size));
	output.data = gpuImage;
	output.format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4;
	output.width = _mWidth;
	output.height = _mHeight;
	output.pixelStrideInBytes = 16;
	output.rowStrideInBytes = _mWidth * 16;
}


inline void OptixDenoiserUtils::ReadImage(std::string fileName, std::vector<float>& imageData, uint& width, uint& height) const
{
	auto formatData = std::vector<byte>{};

	std::ifstream stream{ fileName, std::ios::binary };
	if (!stream.is_open())
	{
		std::stringstream errorStream;
		errorStream << "Could not find file " << fileName;
		throw std::exception{ errorStream.str().c_str() };
	}

	stream.seekg(0, std::ios::end);
	formatData.resize(stream.tellg());	//Resize byte buffer
	stream.seekg(0, std::ios::beg);

	//Reading the file from stream
	stream.read((char*)formatData.data(), formatData.size());
	stream.close();

	//Get the loader of raw image
	auto decoder = GetDecoder(fileName);
	if (!decoder) throw std::exception("Could not find decoder for source image");

	//Read the raw image
	decoder(formatData, imageData, width, height);

	////Resize image data
	//imageData.resize(rawData.size());

	////Recalculate float buffer
	//for (uint i = 0; i < rawData.size(); ++i)
	//	*(((float*)imageData.data()) + i) = *(rawData.data() + i) / 255.0f;  //To minimize access times
}

inline void OptixDenoiserUtils::SaveImage(std::string fileName) const
{
	auto formatData = std::vector<byte>{};
	auto encoder = GetEncoder(fileName);
	if (!encoder) throw std::exception("Could not find encoder for destination image");

	encoder(_mFBufferDenoised, formatData, _mWidth, _mHeight);

	std::ofstream stream{ fileName.c_str(), std::ios::binary };
	if (!stream.is_open())
	{
		std::stringstream errorStream;
		errorStream << "Could not create file " << fileName;
		throw std::exception{ errorStream.str().c_str() };
	}
	//Write to file
	stream.write(reinterpret_cast<char*>(formatData.data()), formatData.size());
	stream.close();
}

inline void OptixDenoiserUtils::Clear()
{
	if (!_mCleared) return;
	_mCleared = true;
	_mFBufferDenoised.clear();
	_mFBufferNormal.clear();
	_mFBufferAlbedo.clear();
	_mFBufferRBG.clear();
	CUDA_CHECK(cudaFree((void*)_mCIntensityBuffer));
	CUDA_CHECK(cudaFree((void*)_mCScratch));
	CUDA_CHECK(cudaFree((void*)_mCDenoiserState));
	CUDA_CHECK(cudaStreamDestroy(_mCStream));
	OPTIX_CHECK(optixDenoiserDestroy(_mODenoiser));
	OPTIX_CHECK(optixDeviceContextDestroy(_mODeviceContext));
	CUDA_CHECK(cudaDeviceReset());
	_mCIntensityBuffer = 0;
	_mCScratch = 0;
	_mCDenoiserState = 0;
	_mCStream = 0;
	_mODenoiser = 0;
	_mODeviceContext = 0;
}

inline void OptixDenoiserUtils::Init()
{
	_mCleared = false;
	CUDA_CHECK(cudaFree(0)); //Init CUDA
	OPTIX_CHECK(optixInit()); //Init Optix
	OPTIX_CHECK(optixDeviceContextCreate(_mCContext, &_mOOptions, &_mODeviceContext)); //Init Optix Context
}

inline void OptixDenoiserUtils::CreateDenoiser(OptixImageType type)
{
	if (_mFBufferRBG.size() > 1 && _mFBufferAlbedo.size() <= 1)
		_mODenoiserOptions.inputKind = OptixDenoiserInputKind::OPTIX_DENOISER_INPUT_RGB;
	else if (_mFBufferRBG.size() > 1 && _mFBufferAlbedo.size() > 1 && _mFBufferNormal.size() <= 1)
		_mODenoiserOptions.inputKind = OptixDenoiserInputKind::OPTIX_DENOISER_INPUT_RGB_ALBEDO;
	else if (_mFBufferRBG.size() > 1 && _mFBufferAlbedo.size() > 1 && _mFBufferNormal.size() > 1)
		_mODenoiserOptions.inputKind = OptixDenoiserInputKind::OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
	else
		throw std::exception("Supported input kind format was not found");

	_mODenoiserOptions.pixelFormat = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4;
	OPTIX_CHECK(optixDenoiserCreate(_mODeviceContext, &_mODenoiserOptions, &_mODenoiser)); //Create denoiser

	//Set AI model 
	if(type == OptixImageType::HDR) OPTIX_CHECK(optixDenoiserSetModel(_mODenoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));
	else OPTIX_CHECK(optixDenoiserSetModel(_mODenoiser, OPTIX_DENOISER_MODEL_KIND_LDR, nullptr, 0));

	//Compute sizes of resources
	OPTIX_CHECK(optixDenoiserComputeMemoryResources(_mODenoiser, _mWidth, _mHeight, &_mOSizes));

	//NOTE as the setup function doesn't preallocate data for calculation
	//it has to be done manually
	CUDA_CHECK(cudaStreamCreate(&_mCStream));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_mCDenoiserState), _mOSizes.stateSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_mCScratch), _mOSizes.recommendedScratchSizeInBytes));

	//Setup the denoiser
	OPTIX_CHECK(optixDenoiserSetup(_mODenoiser, _mCStream, _mWidth, _mHeight, _mCDenoiserState,
	_mOSizes.stateSizeInBytes, _mCScratch, _mOSizes.recommendedScratchSizeInBytes));
}

inline void OptixDenoiserUtils::Denoise(float blendFactor, unsigned int denoiseAlpha, bool precalculateIntenstity)
{
	auto bufferRBG = OptixImage2D{};
	auto bufferAlbedo = OptixImage2D{};
	auto bufferNormal = OptixImage2D{};
	auto bufferOutput = OptixImage2D{};

	//Input buffers
	auto bufferCount = 0u;
	if (_mFBufferRBG.size() > 1 && ++bufferCount) GenerateOptixInputBuffer(_mFBufferRBG, bufferRBG);
	else throw std::exception("RGB buffer was not found");
	if (_mFBufferAlbedo.size() > 1 && ++bufferCount) GenerateOptixInputBuffer(_mFBufferAlbedo, bufferAlbedo);
	if (_mFBufferNormal.size() > 1 && ++bufferCount) GenerateOptixInputBuffer(_mFBufferNormal, bufferNormal);
	
	//Output buffer
	GenerateOptixOutputBuffer(_mFBufferRBG.size() * sizeof(float), bufferOutput);

	auto denoiserParams = OptixDenoiserParams{};
	denoiserParams.blendFactor = blendFactor;
	denoiserParams.denoiseAlpha = denoiseAlpha;

	if (precalculateIntenstity)
	{
		//Reserve data for intensity
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_mCIntensityBuffer), _mFBufferRBG.size() * sizeof(float)));
		denoiserParams.hdrIntensity = _mCIntensityBuffer; //Set itenstity to denoiser params
		//Calculate itenstity
		OPTIX_CHECK(optixDenoiserComputeIntensity(_mODenoiser, _mCStream, &bufferRBG, _mCIntensityBuffer,
			_mCScratch, _mOSizes.recommendedScratchSizeInBytes));
	}

	OptixImage2D* images[3] = { &bufferRBG , &bufferAlbedo , &bufferNormal };
	OPTIX_CHECK(optixDenoiserInvoke(_mODenoiser, _mCStream, &denoiserParams, _mCDenoiserState, _mOSizes.stateSizeInBytes, *images, bufferCount, 0, 0,
				&bufferOutput, _mCScratch, _mOSizes.recommendedScratchSizeInBytes));

	//Copy GPU result to CPU
	_mFBufferDenoised.resize(_mFBufferRBG.size());
	cudaMemcpy((void*)_mFBufferDenoised.data(), (void*)bufferOutput.data, _mFBufferRBG.size() * sizeof(float), cudaMemcpyDeviceToHost);

	//Free the buffers
	//CUDA_CHECK(cudaFree((void*)_mCIntensityBuffer));
	CUDA_CHECK(cudaFree((void*)bufferOutput.data));
	CUDA_CHECK(cudaFree((void*)bufferNormal.data));
	CUDA_CHECK(cudaFree((void*)bufferAlbedo.data));
	CUDA_CHECK(cudaFree((void*)bufferRBG.data));
}

inline void OptixDenoiserUtils::SetRGB(const char* fileName)
{
	unsigned int width, height;
	ReadImage(fileName, _mFBufferRBG, width, height); //Readding up RGB buffer
	_mWidth = width;
	_mHeight = height;
}

inline void OptixDenoiserUtils::SetAlbedo(const char* fileName)
{
	unsigned int width, height;
	ReadImage(fileName, _mFBufferAlbedo, width, height); //Readding up Albedo buffer
	if (_mWidth != width || height != _mHeight) throw std::exception("Albedo image size is not the same as RGB one");
}

inline void OptixDenoiserUtils::SetNormal(const char* fileName)
{
	unsigned int width, height;
	ReadImage(fileName, _mFBufferNormal, width, height); //Readding up Albedo buffer
	if (_mWidth != width || height != _mHeight) throw std::exception("Normal image size is not the same as RGB one");
}

#endif