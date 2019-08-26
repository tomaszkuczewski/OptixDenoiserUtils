# Summary
Example implementation of new Optix 7.0.0 denoiser feature withing single header C++ file.
Code was compiled with Visual Studio Compiler v1922.

# Important
 - For now this implementation supports only 4 channel pixel data (R,G,B,A) and only PNG file format.
 - Optix Denoiser uses the format OPTIX_PIXEL_FORMAT_FLOAT4 so the raw pixel data has to be recalculated (from unsigned char to float) and it takes some time
 - Supports Albedo and Normal bitmaps

# Requirements
- C++ 14
- Optix 7.0 SDK (https://developer.nvidia.com/designworks/optix/download)
- CUDA Toolkit 10.1 Update 2 (https://developer.nvidia.com/cuda-downloads)
- PNG Encoder/Decoder LodePNG (https://github.com/lvandeve/lodepng)

# Denoiser options
- For function CreateDenoiser there are two supported trained AI models (OptixImageType::HDR, OptixImageType::LDR)
- For function Denoise there are three parameters to adjust 
1) float blendFactor - "blendFactor to interpolate between the noisy input image (1.0) and the denoised output image (0.0)."
2) unsigned int denoiseAlpha - "can be used to disable the denoising of the (optional) alpha channelof the noisy image"
3) bool calculateItenstity - If true then it runs the intenstity calculation function for image

# Example code
It is worth to mention that denoising multiple images requires to use all of the 
functions from Init() to Clear() to successfully release all of the used memory during denoising process.

```C++
	OptixDenoiserUtils optix{};

	optix.Init();
	optix.SetRGB("image2.png");
	//Albedo and Normals are supported
	//optix.SetAlbedo("image2_albedo.png"); 
	//optix.SetNormal("image2_normal.png");
	optix.CreateDenoiser(OptixImageType::HDR);
	optix.Denoise(0.0f, 0, true);
	optix.SaveImage("denoised2.png");
	optix.Clear();

	optix.Init();
	optix.SetRGB("image3.png");
	optix.CreateDenoiser(OptixImageType::HDR);
	optix.Denoise(0.0f, 0, true);
	optix.SaveImage("denoised3.png");
	optix.Clear();
```

#Results
With noise
![Noise1](Images/image2.png)
Denoised
![Denoise1](Images/denoised2.png)
With noise
![Noise2](Images/image3.png)
Denoised
![Denoise2](Images/denoised3.png)