//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#pragma once

#include "StepTimer.h"
#include "DXFramework.h"
#include "SuperResolution.h"

using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().

class SuperResolutionX : public DXFramework
{
public:
	SuperResolutionX(uint32_t width, uint32_t height, std::wstring name);
	virtual ~SuperResolutionX();

	virtual void OnInit();
	virtual void OnUpdate();
	virtual void OnRender();
	virtual void OnDestroy();

	virtual void OnKeyUp(uint8_t /*key*/);

	virtual void ParseCommandLineArgs(wchar_t* argv[], int argc);

private:
	static const uint8_t FrameCount = 3;

	XUSG::DescriptorTableLib::sptr	m_descriptorTableLib;

	XUSG::SwapChain::uptr			m_swapChain;
	XUSG::CommandAllocator::uptr	m_commandAllocators[FrameCount];
	XUSG::CommandQueue::uptr		m_commandQueue;

	XUSG::Device::uptr				m_device;
	XUSG::RenderTarget::uptr		m_renderTargets[FrameCount];
	XUSG::CommandList::uptr			m_commandList;

	// DML
	std::unique_ptr<SuperResolution> m_superResolution;
	XUSG::ML::Device::uptr m_mlDevice;
	XUSG::ML::CommandRecorder::uptr m_mlCommandRecorder;

	uint32_t	m_vendorId;

	// Synchronization objects.
	uint32_t	m_frameIndex;
	HANDLE		m_fenceEvent;
	XUSG::Fence::uptr m_fence;
	uint64_t	m_fenceValues[FrameCount];

	// Application state
	bool		m_updateImage;
	bool		m_showFPS;
	bool		m_pausing;
	StepTimer	m_timer;

	// User external settings
	std::wstring m_fileName;

	// Screen-shot helpers and state
	XUSG::Buffer::uptr	m_readBuffer;
	uint32_t			m_rowPitch;
	uint8_t				m_screenShot;

	void LoadPipeline(std::vector<XUSG::Resource::uptr>& uploaders);
	void LoadAssets();
	bool InitTensors(std::vector<XUSG::Resource::uptr>& uploaders);

	void PopulateCommandList();
	void WaitForGpu();
	void MoveToNextFrame();
	void SaveImage(char const* fileName, XUSG::Buffer* imageBuffer,
		uint32_t w, uint32_t h, uint32_t rowPitch, uint8_t comp = 3);
	double CalculateFrameStats(float* fTimeStep = nullptr);
};
