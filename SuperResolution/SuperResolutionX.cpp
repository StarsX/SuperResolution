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

#include "SuperResolutionX.h"
#include "stb_image_write.h"

using namespace std;
using namespace XUSG;
using namespace XUSG::ML;

SuperResolutionX::SuperResolutionX(uint32_t width, uint32_t height, std::wstring name) :
	DXFramework(width, height, name),
	m_frameIndex(0),
	m_deviceType(DEVICE_DISCRETE),
	m_updateImage(false),
	m_showFPS(true),
	m_pausing(false),
	m_fileName(L"Assets/Sashimi128.dds"),
	m_screenShot(0)
{
#if defined (_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	AllocConsole();
	FILE* stream;
	freopen_s(&stream, "CONOUT$", "w+t", stdout);
	freopen_s(&stream, "CONIN$", "r+t", stdin);
#endif
}

SuperResolutionX::~SuperResolutionX()
{
#if defined (_DEBUG)
	FreeConsole();
#endif
}

void SuperResolutionX::OnInit()
{
	vector<Resource::uptr> uploaders(0);
	LoadPipeline(uploaders);
	LoadAssets();
}

// Load the rendering pipeline dependencies.
void SuperResolutionX::LoadPipeline(vector<Resource::uptr>& uploaders)
{
	auto dxgiFactoryFlags = 0u;

#if defined(_DEBUG)
	// Enable the debug layer (requires the Graphics Tools "optional feature").
	// NOTE: Enabling the debug layer after device creation will invalidate the active device.
	{
		com_ptr<ID3D12Debug1> debugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
		{
			debugController->EnableDebugLayer();
			//debugController->SetEnableGPUBasedValidation(TRUE);

			// Enable additional debug layers.
			dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
		}
	}
#endif

	com_ptr<IDXGIFactory5> factory;
	ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

	DXGI_ADAPTER_DESC1 dxgiAdapterDesc;
	com_ptr<IDXGIAdapter1> dxgiAdapter;
	const auto useUMA = m_deviceType == DEVICE_UMA;
	const auto useWARP = m_deviceType == DEVICE_WARP;
	auto checkUMA = true, checkWARP = true;
	auto hr = DXGI_ERROR_NOT_FOUND;
	for (uint8_t n = 0; n < 3; ++n)
	{
		if (FAILED(hr)) hr = DXGI_ERROR_UNSUPPORTED;
		for (auto i = 0u; hr == DXGI_ERROR_UNSUPPORTED; ++i)
		{
			dxgiAdapter = nullptr;
			hr = factory->EnumAdapters1(i, &dxgiAdapter);

			if (SUCCEEDED(hr) && dxgiAdapter)
			{
				dxgiAdapter->GetDesc1(&dxgiAdapterDesc);
				if (checkWARP) hr = dxgiAdapterDesc.VendorId == 0x1414 && dxgiAdapterDesc.DeviceId == 0x8c ?
					(useWARP ? hr : DXGI_ERROR_UNSUPPORTED) : (useWARP ? DXGI_ERROR_UNSUPPORTED : hr);
			}

			if (SUCCEEDED(hr))
			{
				m_device = XUSG::Device::MakeUnique();
				if (SUCCEEDED(m_device->Create(dxgiAdapter.get(), D3D_FEATURE_LEVEL_11_0)) && checkUMA)
				{
					D3D12_FEATURE_DATA_ARCHITECTURE feature = {};
					const auto pDevice = static_cast<ID3D12Device*>(m_device->GetHandle());
					if (SUCCEEDED(pDevice->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &feature, sizeof(feature))))
						hr = feature.UMA ? (useUMA ? hr : DXGI_ERROR_UNSUPPORTED) : (useUMA ? DXGI_ERROR_UNSUPPORTED : hr);
				}
			}
		}

		checkUMA = false;
		if (n) checkWARP = false;
	}

	if (dxgiAdapterDesc.VendorId == 0x1414 && dxgiAdapterDesc.DeviceId == 0x8c) m_title += L" (WARP)";
	else if (dxgiAdapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) m_title += L" (Software)";
	//else m_title += wstring(L" - ") + dxgiAdapterDesc.Description;
	ThrowIfFailed(hr);

	// Create the command queue.
	m_commandQueue = CommandQueue::MakeUnique();
	XUSG_N_RETURN(m_commandQueue->Create(m_device.get(), CommandListType::DIRECT, CommandQueueFlag::NONE,
		0, 0, L"CommandQueue"), ThrowIfFailed(E_FAIL));

	// This sample does not support fullscreen transitions.
	ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(), DXGI_MWA_NO_ALT_ENTER));

	// Create a command allocator for each frame.
	for (uint8_t n = 0; n < FrameCount; ++n)
	{
		m_commandAllocators[n] = CommandAllocator::MakeUnique();
		XUSG_N_RETURN(m_commandAllocators[n]->Create(m_device.get(), CommandListType::DIRECT,
			(L"CommandAllocator" + to_wstring(n)).c_str()), ThrowIfFailed(E_FAIL));
	}

	// Create the command list.
	m_commandList = CommandList::MakeUnique();
	const auto pCommandList = m_commandList.get();
	XUSG_N_RETURN(pCommandList->Create(m_device.get(), 0, CommandListType::DIRECT,
		m_commandAllocators[m_frameIndex].get(), nullptr), ThrowIfFailed(E_FAIL));

	// Create descriptor-table lib.
	m_descriptorTableLib = DescriptorTableLib::MakeShared(m_device.get(), L"DescriptorTableLib");

	XUSG_N_RETURN(InitTensors(uploaders), ThrowIfFailed(E_FAIL));
	m_width = m_superResolution->GetOutWidth();
	m_height = m_superResolution->GetOutHeight();

	// Resize window
	{
		RECT windowRect;
		GetWindowRect(Win32Application::GetHwnd(), &windowRect);
		windowRect.right = windowRect.left + m_width;
		windowRect.bottom = windowRect.top + m_height;

		AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);
		SetWindowPos(Win32Application::GetHwnd(), HWND_TOP, windowRect.left, windowRect.top,
			windowRect.right - windowRect.left, windowRect.bottom - windowRect.top, 0);
	}

	// Describe and create the swap chain.
	m_swapChain = SwapChain::MakeUnique();
	XUSG_N_RETURN(m_swapChain->Create(factory.get(), Win32Application::GetHwnd(), m_commandQueue->GetHandle(),
		FrameCount, m_width, m_height, Format::R8G8B8A8_UNORM, SwapChainFlag::ALLOW_TEARING), ThrowIfFailed(E_FAIL));

	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

	// Create frame resources.
	// Create a RTV for each frame.
	for (uint8_t n = 0; n < FrameCount; ++n)
	{
		m_renderTargets[n] = RenderTarget::MakeUnique();
		XUSG_N_RETURN(m_renderTargets[n]->CreateFromSwapChain(m_device.get(), m_swapChain.get(), n), ThrowIfFailed(E_FAIL));
	}
}

// Load the sample assets.
void SuperResolutionX::LoadAssets()
{
	// Close the command list and execute it to begin the initial GPU setup.
	XUSG_N_RETURN(m_commandList->Close(), ThrowIfFailed(E_FAIL));
	m_commandQueue->ExecuteCommandList(m_commandList.get());

	// Create synchronization objects and wait until assets have been uploaded to the GPU.
	{
		if (!m_fence)
		{
			m_fence = Fence::MakeUnique();
			XUSG_N_RETURN(m_fence->Create(m_device.get(), m_fenceValues[m_frameIndex]++, FenceFlag::NONE, L"Fence"), ThrowIfFailed(E_FAIL));
		}

		// Create an event handle to use for frame synchronization.
		m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		if (!m_fenceEvent) ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));

		// Wait for the command list to execute; we are reusing the same command 
		// list in our main loop but for now, we just want to wait for setup to 
		// complete before continuing.
		WaitForGpu();
	}
}

bool SuperResolutionX::InitTensors(vector<Resource::uptr>& uploaders)
{
	CreateDeviceFlag createDeviceFlags = CreateDeviceFlag::NONE;

#if defined (_DEBUG)
	// If the project is in a debug build, then enable debugging via DirectML debug layers with this flag.
	createDeviceFlags |= CreateDeviceFlag::DEBUG;
#endif

	m_mlDevice = ML::Device::MakeUnique();
	XUSG_N_RETURN(m_mlDevice->Create(m_device.get(), createDeviceFlags), false);

	DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT fp16Query = { DML_TENSOR_DATA_TYPE_FLOAT16 };
	DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT fp16Supported = {};
	const auto pDevice = static_cast<IDMLDevice*>(m_mlDevice->GetHandle());
	const auto hr = pDevice->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(fp16Query),
		&fp16Query, sizeof(fp16Supported), &fp16Supported);

	// The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
	m_mlCommandRecorder = CommandRecorder::MakeUnique();
	XUSG_N_RETURN(m_mlDevice->GetCommandRecorder(m_mlCommandRecorder.get()), false);

	XUSG_X_RETURN(m_superResolution, make_unique<SuperResolution>(), false);

	return m_superResolution->Init(m_commandList.get(), m_mlCommandRecorder.get(),
		m_descriptorTableLib, m_vendorId, uploaders, m_fileName.c_str(),
		fp16Supported.IsSupported);
}

// Update frame-based values.
void SuperResolutionX::OnUpdate()
{
	// Timer
	static auto time = 0.0, pauseTime = 0.0;

	m_timer.Tick();
	const auto totalTime = CalculateFrameStats();
	pauseTime = m_pausing ? totalTime - time : pauseTime;
	time = totalTime - pauseTime;
}

// Render the scene.
void SuperResolutionX::OnRender()
{
	// Record all the commands we need to render the scene into the command list.
	PopulateCommandList();

	// Execute the command list.
	m_commandQueue->ExecuteCommandList(m_commandList.get());

	// Present the frame.
	XUSG_N_RETURN(m_swapChain->Present(0, PresentFlag::ALLOW_TEARING), ThrowIfFailed(E_FAIL));

	MoveToNextFrame();
}

void SuperResolutionX::OnDestroy()
{
	// Ensure that the GPU is no longer referencing resources that are about to be
	// cleaned up by the destructor.
	WaitForGpu();

	CloseHandle(m_fenceEvent);
}

// User hot-key interactions.
void SuperResolutionX::OnKeyUp(uint8_t key)
{
	switch (key)
	{
	case VK_SPACE:
		m_pausing = !m_pausing;
		break;
	case VK_F1:
		m_showFPS = !m_showFPS;
		break;
	case VK_F11:
		m_screenShot = 1;
		break;
	case 'U':
		m_updateImage = !m_updateImage;
		break;
	}
}

void SuperResolutionX::ParseCommandLineArgs(wchar_t* argv[], int argc)
{
	const auto str_tolower = [](wstring s)
		{
			transform(s.begin(), s.end(), s.begin(), [](wchar_t c) { return towlower(c); });

			return s;
		};

	const auto isArgMatched = [&argv, &str_tolower](int i, const wchar_t* paramName)
		{
			const auto& arg = argv[i];

			return (arg[0] == L'-' || arg[0] == L'/')
				&& str_tolower(&arg[1]) == str_tolower(paramName);
		};

	const auto hasNextArgValue = [&argv, &argc](int i)
		{
			const auto& arg = argv[i + 1];

			return i + 1 < argc && arg[0] != L'/' &&
				(arg[0] != L'-' || (arg[1] >= L'0' && arg[1] <= L'9') || arg[1] == L'.');
		};

	DXFramework::ParseCommandLineArgs(argv, argc);

	for (auto i = 1; i < argc; ++i)
	{
		if (isArgMatched(i, L"warp")) m_deviceType = DEVICE_WARP;
		else if (isArgMatched(i, L"uma")) m_deviceType = DEVICE_UMA;
		else if (isArgMatched(i, L"i") || isArgMatched(i, L"image"))
		{
			if (hasNextArgValue(i)) m_fileName = argv[++i];
		}
	}
}

void SuperResolutionX::PopulateCommandList()
{
	// Command list allocators can only be reset when the associated 
	// command lists have finished execution on the GPU; apps should use 
	// fences to determine GPU execution progress.
	const auto pCommandAllocator = m_commandAllocators[m_frameIndex].get();
	XUSG_N_RETURN(pCommandAllocator->Reset(), ThrowIfFailed(E_FAIL));

	// However, when ExecuteCommandList() is called on a particular command 
	// list, that command list can then be reset at any time and must be before 
	// re-recording.
	const auto pCommandList = m_commandList.get();
	XUSG_N_RETURN(pCommandList->Reset(pCommandAllocator, nullptr), ThrowIfFailed(E_FAIL));

	const auto descriptorHeap = m_descriptorTableLib->GetDescriptorHeap(CBV_SRV_UAV_HEAP);
	pCommandList->SetDescriptorHeaps(1, &descriptorHeap);

	static auto isFirstFrame = true;
	if (isFirstFrame || m_updateImage)
	{
		m_superResolution->ImageToTensors(pCommandList);
		m_superResolution->Process(pCommandList, m_mlCommandRecorder.get());
		isFirstFrame = false;
	}

	const auto pRenderTarget = m_renderTargets[m_frameIndex].get();
	m_superResolution->Render(pCommandList, pRenderTarget);

	ResourceBarrier barrier;
	const auto numBarriers = pRenderTarget->SetBarrier(&barrier, ResourceState::PRESENT);
	pCommandList->Barrier(numBarriers, &barrier);

	// Screen-shot helper
	if (m_screenShot == 1)
	{
		if (!m_readBuffer) m_readBuffer = Buffer::MakeUnique();
		pRenderTarget->ReadBack(pCommandList, m_readBuffer.get(), &m_rowPitch);
		m_screenShot = 2;
	}

	XUSG_N_RETURN(pCommandList->Close(), ThrowIfFailed(E_FAIL));
}

// Wait for pending GPU work to complete.
void SuperResolutionX::WaitForGpu()
{
	// Schedule a Signal command in the queue.
	XUSG_N_RETURN(m_commandQueue->Signal(m_fence.get(), m_fenceValues[m_frameIndex]), ThrowIfFailed(E_FAIL));

	// Wait until the fence has been processed.
	XUSG_N_RETURN(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent), ThrowIfFailed(E_FAIL));
	WaitForSingleObject(m_fenceEvent, INFINITE);

	// Increment the fence value for the current frame.
	m_fenceValues[m_frameIndex]++;
}

// Prepare to render the next frame.
void SuperResolutionX::MoveToNextFrame()
{
	// Schedule a Signal command in the queue.
	const auto currentFenceValue = m_fenceValues[m_frameIndex];
	XUSG_N_RETURN(m_commandQueue->Signal(m_fence.get(), currentFenceValue), ThrowIfFailed(E_FAIL));

	// Update the frame index.
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

	// If the next frame is not ready to be rendered yet, wait until it is ready.
	if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
	{
		XUSG_N_RETURN(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent), ThrowIfFailed(E_FAIL));
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}

	// Set the fence value for the next frame.
	m_fenceValues[m_frameIndex] = currentFenceValue + 1;

	// Screen-shot helper
	if (m_screenShot)
	{
		if (m_screenShot > FrameCount)
		{
			char timeStr[15];
			tm dateTime;
			const auto now = time(nullptr);
			if (!localtime_s(&dateTime, &now) && strftime(timeStr, sizeof(timeStr), "%Y%m%d%H%M%S", &dateTime))
				SaveImage((string("SuperResolutionX_") + timeStr + ".png").c_str(), m_readBuffer.get(), m_width, m_height, m_rowPitch);
			m_screenShot = 0;
		}
		else ++m_screenShot;
	}
}

void SuperResolutionX::SaveImage(char const* fileName, Buffer* pImageBuffer, uint32_t w, uint32_t h, uint32_t rowPitch, uint8_t comp)
{
	assert(comp == 3 || comp == 4);
	const auto pData = static_cast<const uint8_t*>(pImageBuffer->Map(nullptr));

	//stbi_write_png_compression_level = 1024;
	vector<uint8_t> imageData(comp * w * h);
	const auto sw = rowPitch / 4; // Byte to pixel
	for (auto i = 0u; i < h; ++i)
		for (auto j = 0u; j < w; ++j)
		{
			const auto s = sw * i + j;
			const auto d = w * i + j;
			for (uint8_t k = 0; k < comp; ++k)
				imageData[comp * d + k] = pData[4 * s + k];
		}

	stbi_write_png(fileName, w, h, comp, imageData.data(), 0);

	pImageBuffer->Unmap();
}

double SuperResolutionX::CalculateFrameStats(float* pTimeStep)
{
	static auto frameCnt = 0u;
	static auto previousTime = 0.0;
	const auto totalTime = m_timer.GetTotalSeconds();
	++frameCnt;

	const auto timeStep = totalTime - previousTime;

	// Compute averages over one second period.
	if (timeStep >= 1.0)
	{
		const auto fps = static_cast<float>(frameCnt / timeStep);	// Normalize to an exact second.

		frameCnt = 0;
		previousTime = totalTime;

		wstringstream windowText;
		windowText << L"    fps: ";
		if (m_showFPS) windowText << setprecision(2) << fixed << fps;
		else windowText << L"[F1]";

		windowText << L"    [F11] screen shot";

		SetCustomWindowText(windowText.str().c_str());
	}

	if (pTimeStep) *pTimeStep = static_cast<float>(m_timer.GetElapsedSeconds());

	return totalTime;
}
