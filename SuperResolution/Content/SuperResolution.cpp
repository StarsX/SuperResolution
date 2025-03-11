//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "SuperResolution.h"

#define _ENABLE_STB_IMAGE_LOADER_ONLY_
#include "Advanced/XUSGTextureLoader.h"

using namespace std;
using namespace XUSG;
using namespace XUSG::ML;

SuperResolution::SuperResolution() :
	m_tensorLayout(TensorLayout::DEFAULT),
	m_tensorDataType(TensorDataType::FLOAT32)
{
	m_shaderLib = ShaderLib::MakeUnique();
}

SuperResolution::~SuperResolution()
{
}

bool SuperResolution::Init(CommandList* pCommandList, const CommandRecorder* pCommandRecorder,
	const DescriptorTableLib::sptr& descriptorTableLib, uint32_t vendorId,
	vector<Resource::uptr>& uploaders, const char* fileName, bool isFP16Supported)
{
	const auto pDevice = pCommandList->GetDevice();
	m_pipelineLayoutLib = PipelineLayoutLib::MakeUnique(pDevice);
	m_graphicsPipelineLib = Graphics::PipelineLib::MakeUnique(pDevice);
	m_computePipelineLib = Compute::PipelineLib::MakeUnique(pDevice);
	m_descriptorTableLib = descriptorTableLib;

	// Load input image
	m_inputImage = Texture::MakeUnique();
	uploaders.emplace_back(Resource::MakeUnique());
	XUSG_N_RETURN(CreateTextureFromFile(pCommandList, fileName, m_inputImage.get(),
		uploaders.back().get(), ResourceState::COMMON, MemoryFlag::NONE, L"Source"), false);

	m_imageLayoutIn.Width = static_cast<uint32_t>(m_inputImage->GetWidth());
	m_imageLayoutIn.Height = m_inputImage->GetHeight();

	XUSG_N_RETURN(createResources(pCommandList, pCommandRecorder, vendorId, uploaders, isFP16Supported), false);
	XUSG_N_RETURN(initResources(pCommandList, pCommandRecorder), false);

	XUSG_N_RETURN(createPipelineLayouts(), false);
	XUSG_N_RETURN(createPipelines(), false);

	m_imageLayoutOut.Width = m_imageLayoutIn.Width << 1;
	m_imageLayoutOut.Height = m_imageLayoutIn.Height << 1;
	m_imageLayoutOut.UseNhwc = m_imageLayoutIn.UseNhwc = m_tensorLayout == TensorLayout::NHWC;

	return true;
}

void SuperResolution::ImageToTensors(CommandList* pCommandList)
{
	ResourceBarrier barrier;
	auto numBarriers = m_modelInput->SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);
	pCommandList->Barrier(numBarriers, &barrier);

	pCommandList->SetComputePipelineLayout(m_pipelineLayouts[0]);
	pCommandList->SetCompute32BitConstants(0, 3, &m_imageLayoutIn);
	pCommandList->SetComputeDescriptorTable(1, m_uavSrvTable);
	pCommandList->SetPipelineState(m_pipelines[0]);
	pCommandList->Dispatch(XUSG_DIV_UP(m_imageLayoutIn.Width, 8), XUSG_DIV_UP(m_imageLayoutIn.Height, 8), 1);

	numBarriers = m_modelInput->SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);
	pCommandList->Barrier(numBarriers, &barrier);
}

void SuperResolution::Process(CommandList* pCommandList, const CommandRecorder* pCommandRecorder)
{
	ResourceBarrier barrier;
	auto numBarriers = m_modelOutput->SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);
	pCommandList->Barrier(numBarriers, &barrier);

	// Create an upsampled (nearest neighbor) version of the image first
	pCommandRecorder->Dispatch(pCommandList, m_upsampleOps[0]->GetDispatchable(), m_upsampleBindings[0]->GetDispatchableBindingTable());
	// No UAV barrier is required here since we don't use the result right away.

	// Run the intermediate model steps: 3 convolutions (with premultiplied batch normalization
	// baked into the weights), an upsample, 3 convolutions w/ premultiplied batch norm, 1 final convolution.
	// This generates a residual image.
	const ResourceBarrier allUavBarrier = { nullptr, ResourceState::UNORDERED_ACCESS };
	for (auto i = 0u; i < c_numConvLayers; ++i)
	{
		// Convolution
		pCommandRecorder->Dispatch(pCommandList, m_convOps[i]->GetDispatchable(), m_convBindings[i]->GetDispatchableBindingTable());
		pCommandList->Barrier(1, &allUavBarrier);

		if (i == 2)
		{
			// Intermediate upsample
			pCommandRecorder->Dispatch(pCommandList, m_upsampleOps[1]->GetDispatchable(), m_upsampleBindings[1]->GetDispatchableBindingTable());
			pCommandList->Barrier(1, &allUavBarrier);
		}
	}

	// Add the residual image to the original nearest-neighbor upscale
	pCommandRecorder->Dispatch(pCommandList, m_addResidualOp->GetDispatchable(), m_addResidualBinding->GetDispatchableBindingTable());
}

void SuperResolution::Render(CommandList* pCommandList, RenderTarget* pRenderTarget)
{
	ResourceBarrier barriers[2];
	auto numBarriers = pRenderTarget->SetBarrier(barriers, ResourceState::RENDER_TARGET);
	numBarriers = m_modelOutput->SetBarrier(barriers, ResourceState::PIXEL_SHADER_RESOURCE, numBarriers);
	pCommandList->Barrier(numBarriers, barriers);

	// Set pipeline
	pCommandList->SetGraphicsPipelineLayout(m_pipelineLayouts[1]);
	pCommandList->SetGraphics32BitConstants(0, XUSG_UINT32_SIZE_OF(m_imageLayoutOut), &m_imageLayoutOut);
	pCommandList->SetGraphicsDescriptorTable(1, m_srvTable);
	pCommandList->SetPipelineState(m_pipelines[1]);

	// Set viewport
	const auto& width = m_imageLayoutOut.Width;
	const auto& height = m_imageLayoutOut.Height;
	Viewport viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
	RectRange scissorRect(0, 0, width, height);
	pCommandList->RSSetViewports(1, &viewport);
	pCommandList->RSSetScissorRects(1, &scissorRect);

	pCommandList->OMSetRenderTargets(1, &pRenderTarget->GetRTV());

	pCommandList->IASetPrimitiveTopology(PrimitiveTopology::TRIANGLELIST);
	pCommandList->Draw(3, 1, 0, 0);
}

uint32_t SuperResolution::GetOutWidth() const
{
	return m_imageLayoutOut.Width;
}

uint32_t SuperResolution::GetOutHeight() const
{
	return m_imageLayoutOut.Height;
}

bool SuperResolution::createResources(CommandList* pCommandList, const CommandRecorder* pCommandRecorder,
	uint32_t vendorId, vector<Resource::uptr>& uploaders, bool isFP16Supported)
{
	// ML device
	const auto pDevice = pCommandList->GetDevice();
	const auto pMLDevice = pCommandRecorder->GetDevice();

	auto format = Format::R32_FLOAT;
	uint32_t dataStride = sizeof(float);
	{
#if FORCE_NCHW
		m_tensorLayout = TensorLayout::Default;
#else
		m_tensorLayout = vendorId == 0x10DE ? // Nvidia
			// This is faster on recent Nvidia hardware, but may be a problem on older hardware.
			// If necessary, set FORCE_NCHW to override this.
			TensorLayout::NHWC :
			TensorLayout::DEFAULT;
#endif
		m_tensorDataType = isFP16Supported ? TensorDataType::FLOAT16 : TensorDataType::FLOAT32;
		format = isFP16Supported ? Format::R16_FLOAT : Format::R32_FLOAT;
		dataStride = isFP16Supported ? sizeof(uint16_t) : sizeof(float);
	}

	uint64_t modelInputBufferSize = 0;
	uint64_t modelOutputBufferSize = 0;
	uint64_t intermediateBufferMaxSize[] = { 0, 0 };

	// ML operator resources--implementation of the super-resolution model
	{
		const auto mlUtil = ML::Util::MakeUnique(m_tensorDataType, m_tensorLayout);

		// Create an upscaled (nearest neighbor) version of the image first
		const auto& width = m_imageLayoutIn.Width;
		const auto& height = m_imageLayoutIn.Height;
		const uint32_t modelInputSizes[] = { 1, 3, height, width };
		uint32_t upscaledInputSizes[4];
		m_upsampleOps[0] = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateUpsampleLayer(pMLDevice, modelInputSizes, modelInputBufferSize,
			modelOutputBufferSize, upscaledInputSizes, *m_upsampleOps[0]), false);

		// Create the residual with three convolutions, an upsample, and four more convolutions
		WeightMapType weights;
		if (!LoadWeights("Assets/weights.bin", weights))
			throw std::exception("loadWeights");

		uint32_t filterSizes[] = { 32, 3, 5, 5 };
		uint32_t intermediateInputSizes[2][4];
		m_convOps[0] = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateConvolutionLayer(pMLDevice, modelInputSizes, filterSizes, true, modelInputBufferSize,
			intermediateBufferMaxSize[0], intermediateInputSizes[0], *m_convOps[0]), false);
		XUSG_N_RETURN(createWeightTensors(pCommandList, weights, "conv1/weights", "conv1/BatchNorm/scale",
			"conv1/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[0],
			m_modelConvBiasWeights[0]), false);

		// Which intermediate resource to use as input for the current operation. The other will be
		// used as output. Then the next op will swap the order.
		uint8_t inputIndex = 0;

		filterSizes[0] = 64;	// output filters
		filterSizes[1] = 32;	// input channels
		filterSizes[2] = 3;		// filter height
		filterSizes[3] = 3;		// filter width
		m_convOps[1] = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateConvolutionLayer(pMLDevice, intermediateInputSizes[inputIndex], filterSizes,
			true, intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], *m_convOps[1]), false);
		XUSG_N_RETURN(createWeightTensors(pCommandList, weights, "conv2/weights", "conv2/BatchNorm/scale",
			"conv2/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[1],
			m_modelConvBiasWeights[1]), false);
		inputIndex = 1 - inputIndex;

		filterSizes[1] = 64;
		m_convOps[2] = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateConvolutionLayer(pMLDevice, intermediateInputSizes[inputIndex], filterSizes,
			true, intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], *m_convOps[2]), false);
		XUSG_N_RETURN(createWeightTensors(pCommandList, weights, "conv3/weights", "conv3/BatchNorm/scale",
			"conv3/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[2],
			m_modelConvBiasWeights[2]), false);
		inputIndex = 1 - inputIndex;

		m_upsampleOps[1] = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateUpsampleLayer(pMLDevice, intermediateInputSizes[inputIndex], intermediateBufferMaxSize[inputIndex],
			intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], *m_upsampleOps[1]), false);
		inputIndex = 1 - inputIndex;

		filterSizes[0] = 32;
		filterSizes[2] = 5;
		filterSizes[3] = 5;
		m_convOps[3] = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateConvolutionLayer(pMLDevice, intermediateInputSizes[inputIndex], filterSizes,
			true, intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], *m_convOps[3]), false);
		XUSG_N_RETURN(createWeightTensors(pCommandList, weights, "conv_up1/conv/weights",
			"conv_up1/conv/BatchNorm/scale", "conv_up1/conv/BatchNorm/shift", filterSizes,
			uploaders, m_modelConvFilterWeights[3], m_modelConvBiasWeights[3]), false);
		inputIndex = 1 - inputIndex;

		filterSizes[1] = 32;
		filterSizes[2] = 3;
		filterSizes[3] = 3;
		m_convOps[4] = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateConvolutionLayer(pMLDevice, intermediateInputSizes[inputIndex], filterSizes,
			true, intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], *m_convOps[4]), false);
		XUSG_N_RETURN(createWeightTensors(pCommandList, weights, "conv4/weights", "conv4/BatchNorm/scale",
			"conv4/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[4],
			m_modelConvBiasWeights[4]), false);
		inputIndex = 1 - inputIndex;

		m_convOps[5] = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateConvolutionLayer(pMLDevice, intermediateInputSizes[inputIndex], filterSizes,
			true, intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], *m_convOps[5]), false);
		XUSG_N_RETURN(createWeightTensors(pCommandList, weights, "conv5/weights", "conv5/BatchNorm/scale",
			"conv5/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[5],
			m_modelConvBiasWeights[5]), false);
		inputIndex = 1 - inputIndex;

		filterSizes[0] = 3;
		m_convOps[6] = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateConvolutionLayer(pMLDevice, intermediateInputSizes[inputIndex], filterSizes,
			false, intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], *m_convOps[6]), false);
		XUSG_N_RETURN(createWeightTensors(pCommandList, weights, "conv6/weights", nullptr, nullptr,
			filterSizes, uploaders, m_modelConvFilterWeights[6], m_modelConvBiasWeights[6]), false);
		inputIndex = 1 - inputIndex;

		// Finally add the residual to the original upsampled image
		assert(memcmp(upscaledInputSizes, intermediateInputSizes[inputIndex], 4 * dataStride) == 0);

		m_addResidualOp = Operator::MakeShared();
		XUSG_N_RETURN(mlUtil->CreateAdditionLayer(pMLDevice, upscaledInputSizes, *m_addResidualOp), false);
	}

	// Buffers for ML inputs and outputs
	{
		// Resource for input tensor
		auto numElements = static_cast<uint32_t>(modelInputBufferSize / dataStride);
		m_modelInput = TypedBuffer::MakeUnique();
		XUSG_N_RETURN(m_modelInput->Create(pDevice, numElements, dataStride, format,
			ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0,
			nullptr, 1, nullptr, MemoryFlag::NONE, L"InputBuffer"), false);

		// Model result tensor is 2x larger in both dimensions
		numElements = static_cast<uint32_t>(modelOutputBufferSize / dataStride);
		m_modelOutput = TypedBuffer::MakeUnique();
		XUSG_N_RETURN(m_modelOutput->Create(pDevice, numElements, dataStride, format,
			ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 1,
			nullptr, 0, nullptr, MemoryFlag::NONE, L"OutputBuffer"), false);

		// Create two resources for intermediate layer results. Each layer will ping-pong between these. They're each large
		// enough to hold the largest intermediate result required.
		for (uint8_t i = 0; i < 2; ++i)
		{
			m_modelIntermediateResult[i] = Buffer::MakeUnique();
			XUSG_N_RETURN(m_modelIntermediateResult[i]->Create(pDevice, intermediateBufferMaxSize[i],
				ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0, nullptr, 0, nullptr,
				MemoryFlag::NONE, (L"IntermediateResultBuffer" + to_wstring(i)).c_str()), false);
		}
	}

	return true;
}

bool SuperResolution::createWeightTensors(CommandList* pCommandList, WeightMapType& weights, const char* convLayerName,
	const char* scaleLayerName, const char* shiftLayerName, const uint32_t filterSizes[4], vector<Resource::uptr>& uploaders,
	Buffer::uptr& filterWeightBuffer, Buffer::uptr& biasWeightBuffer)
{
	const auto pDevice = pCommandList->GetDevice();

	const auto mlUtil = ML::Util::MakeUnique(m_tensorDataType, m_tensorLayout);
	vector<uint8_t> filterWeights;
	vector<uint8_t> biasWeights;

	mlUtil->CreateWeightTensors(weights, convLayerName, scaleLayerName,
		shiftLayerName, filterSizes, filterWeights, biasWeights);

	auto useScaleShift = true;
	if (scaleLayerName == nullptr)
	{
		assert(shiftLayerName == nullptr);
		useScaleShift = false;
	}

	uploaders.emplace_back(Resource::MakeUnique());
	XUSG_N_RETURN(createWeightResource(pDevice, filterSizes, filterWeightBuffer), false);
	XUSG_N_RETURN(filterWeightBuffer->Upload(pCommandList, uploaders.back().get(), filterWeights.data(),
		filterWeights.size(), 0, ResourceState::UNORDERED_ACCESS), false);

	if (useScaleShift)
	{
		const uint32_t biasSizes[] = { 1, filterSizes[0], 1, 1 };	// One bias per output channel
		uploaders.emplace_back(Resource::MakeUnique());
		XUSG_N_RETURN(createWeightResource(pDevice, biasSizes, biasWeightBuffer), false);
		XUSG_N_RETURN(biasWeightBuffer->Upload(pCommandList, uploaders.back().get(), biasWeights.data(),
			biasWeights.size(), 0, ResourceState::UNORDERED_ACCESS), false);

		// The scale weights will be premultiplied into the filter weights, so they don't need
		// a separate resource.
	}
	else biasWeightBuffer = nullptr;

	return true;
}

bool SuperResolution::createWeightResource(const XUSG::Device* pDevice, const uint32_t tensorSizes[4], Buffer::uptr& resourceOut)
{
	uint32_t strides[4];
	ML::Util::GetStrides(tensorSizes, m_tensorLayout, strides);
	
	const auto tensor = Tensor::MakeUnique();
	const auto bufferSize = tensor->Create(m_tensorDataType, static_cast<uint32_t>(size(strides)), tensorSizes, strides);

	resourceOut = Buffer::MakeUnique();

	return resourceOut->Create(pDevice, bufferSize, ResourceFlag::ALLOW_UNORDERED_ACCESS,
		MemoryType::DEFAULT, 0, nullptr, 0, nullptr, MemoryFlag::NONE, L"WeightBuffer");
}

bool SuperResolution::createPipelineLayouts()
{
	{
		const auto pipelineLayout = XUSG::Util::PipelineLayout::MakeUnique();
		pipelineLayout->SetRange(0, DescriptorType::CONSTANT, XUSG_UINT32_SIZE_OF(ImageLayout), 0);
		pipelineLayout->SetRange(1, DescriptorType::UAV, 1, 0, 0,
			DescriptorFlag::DATA_STATIC_WHILE_SET_AT_EXECUTE | DescriptorFlag::DESCRIPTORS_VOLATILE);
		pipelineLayout->SetRange(1, DescriptorType::SRV, 1, 0,
			0, DescriptorFlag::DATA_STATIC);
		XUSG_X_RETURN(m_pipelineLayouts[0], pipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::DENY_VERTEX_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_HULL_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_DOMAIN_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_GEOMETRY_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_PIXEL_SHADER_ROOT_ACCESS,
			L"ImageToTensorLayout"), false);
	}

	{
		const auto pipelineLayout = XUSG::Util::PipelineLayout::MakeUnique();
		pipelineLayout->SetRange(0, DescriptorType::CONSTANT, XUSG_UINT32_SIZE_OF(ImageLayout), 0);
		pipelineLayout->SetRange(1, DescriptorType::SRV, 1, 0, 0, DescriptorFlag::DESCRIPTORS_VOLATILE);
		pipelineLayout->SetShaderStage(0, Shader::Stage::PS);
		pipelineLayout->SetShaderStage(1, Shader::Stage::PS);
		XUSG_X_RETURN(m_pipelineLayouts[1], pipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::DENY_VERTEX_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_HULL_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_DOMAIN_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_GEOMETRY_SHADER_ROOT_ACCESS,
			L"TensorToImageLayout"), false);
	}

	return true;
}

bool SuperResolution::createPipelines()
{
	{
		const auto cs = m_shaderLib->CreateShader(Shader::Stage::CS, 0, L"CSImageToTensor.cso");
		XUSG_N_RETURN(cs, false);

		const auto state = Compute::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[0]);
		state->SetShader(cs);
		XUSG_X_RETURN(m_pipelines[0], state->GetPipeline(m_computePipelineLib.get(), L"ImageToTensor"), false);
	}

	{
		const auto vs = m_shaderLib->CreateShader(Shader::Stage::VS, 0, L"VSTensorToImage.cso");
		const auto ps = m_shaderLib->CreateShader(Shader::Stage::PS, 0, L"PSTensorToImage.cso");
		XUSG_N_RETURN(vs, false);
		XUSG_N_RETURN(ps, false);

		const auto state = Graphics::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[1]);
		state->SetShader(Shader::Stage::VS, vs);
		state->SetShader(Shader::Stage::PS, ps);
		state->DSSetState(Graphics::DEPTH_STENCIL_NONE, m_graphicsPipelineLib.get());
		state->IASetPrimitiveTopologyType(PrimitiveTopologyType::TRIANGLE);
		state->OMSetNumRenderTargets(1);
		state->OMSetRTVFormat(0, Format::R8G8B8A8_UNORM);
		XUSG_X_RETURN(m_pipelines[1], state->GetPipeline(m_graphicsPipelineLib.get(), L"TensorToImage"), false);
	}

	return true;
}

bool SuperResolution::createDescriptorTables()
{
	{
		const Descriptor descriptors[] =
		{
			m_modelInput->GetUAV(),
			m_inputImage->GetSRV()
		};
		const auto descriptorTable = XUSG::Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_uavSrvTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	{
		const auto descriptorTable = XUSG::Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_modelOutput->GetSRV());
		XUSG_X_RETURN(m_srvTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	return true;
}

bool SuperResolution::initResources(CommandList* pCommandList,// const CommandAllocator& commandAllocator,
	const CommandRecorder* pCommandRecorder)
{
	// ML device
	const auto pDevice = pCommandList->GetDevice();
	const auto pMLDevice = pCommandRecorder->GetDevice();

	//pCommandList->Reset(commandAllocator, nullptr);

	// Create operator initializers and descriptor heap for binding
	uint32_t upsampleOpDescriptorCount, convOpDescriptorCount, additionOpDescriptorCount;
	uint32_t upsampleDescriptorsIdx, convDescriptorsIdx, additionDescriptorsIdx;

	OperatorInitializer::uptr opInitializers[NUM_OP];
	DescriptorHeap descriptorHeap;
	{
		// The same descriptor heap will be used for both initializing and executing operators. These each happen
		// at different times, so we reuse the same descriptor slots. GetDescriptorCount() ensures there are enough
		// slots for both cases.
		opInitializers[OP_UP_SAMPLE] = OperatorInitializer::MakeUnique();
		XUSG_N_RETURN(opInitializers[OP_UP_SAMPLE]->Create(pMLDevice, m_upsampleOps, c_numUpsampleLayers), false);
		upsampleOpDescriptorCount = opInitializers[OP_UP_SAMPLE]->GetDescriptorCount();
		
		opInitializers[OP_CONV] = OperatorInitializer::MakeUnique();
		XUSG_N_RETURN(opInitializers[OP_CONV]->Create(pMLDevice, m_convOps, c_numConvLayers), false);
		convOpDescriptorCount = opInitializers[OP_CONV]->GetDescriptorCount();

		opInitializers[OP_ADD] = OperatorInitializer::MakeUnique();
		XUSG_N_RETURN(opInitializers[OP_ADD]->Create(pMLDevice, &m_addResidualOp, 1), false);
		additionOpDescriptorCount = opInitializers[OP_ADD]->GetDescriptorCount();

		upsampleDescriptorsIdx = 3;
		convDescriptorsIdx = upsampleDescriptorsIdx + upsampleOpDescriptorCount * c_numUpsampleLayers;
		additionDescriptorsIdx = convDescriptorsIdx + convOpDescriptorCount * c_numConvLayers;
		const auto descriptorCount = additionDescriptorsIdx + additionOpDescriptorCount;

		XUSG_N_RETURN(m_descriptorTableLib->AllocateDescriptorHeap(CBV_SRV_UAV_HEAP, descriptorCount), false);
		XUSG_N_RETURN(createDescriptorTables(), false);
		descriptorHeap = m_descriptorTableLib->GetDescriptorHeap(CBV_SRV_UAV_HEAP);

		// Operator initialization dispatches will use this heap right away
		pCommandList->SetDescriptorHeaps(1, &descriptorHeap);
	}

	// Create any persistent resources required for the operators.
	{
		for (auto i = 0u; i < c_numUpsampleLayers; ++i)
		{
			const auto persistentResourceSize = m_upsampleOps[i]->GetPersistentResourceSize();
			if (persistentResourceSize > 0)
			{
				m_modelUpsamplePersistentResources[i] = Buffer::MakeUnique();
				XUSG_N_RETURN(m_modelUpsamplePersistentResources[i]->Create(pDevice, persistentResourceSize,
					ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0, nullptr, 0, nullptr,
					MemoryFlag::NONE, (L"UpSamplePersistent" + to_wstring(i)).c_str()), false);
			}
		}

		for (auto i = 0u; i < c_numConvLayers; ++i)
		{
			const auto persistentResourceSize = m_convOps[i]->GetPersistentResourceSize();
			if (persistentResourceSize > 0)
			{
				m_modelConvPersistentResources[i] = Buffer::MakeUnique();
				XUSG_N_RETURN(m_modelConvPersistentResources[i]->Create(pDevice, persistentResourceSize,
					ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0, nullptr, 0, nullptr,
					MemoryFlag::NONE, (L"ConvPersistent" + to_wstring(i)).c_str()), false);
			}
		}

		{
			const auto persistentResourceSize = m_addResidualOp->GetPersistentResourceSize();
			if (persistentResourceSize > 0)
			{
				m_modelAddPersistentResource = Buffer::MakeUnique();
				XUSG_N_RETURN(m_modelAddPersistentResource->Create(pDevice, persistentResourceSize,
					ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0, nullptr,
					0, nullptr, MemoryFlag::NONE, L"AddPersistent"), false);
			}	
		}
	}

	// When binding input and output resources, take note of which temp resource is used at the time:
	// Layer		| Input							| Output
	// Upsample[0]	| m_modelInput					| m_modelOutput
	// Conv[0]		| m_modelInput					| m_modelIntermediateResult[0]
	// Conv[1]		| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
	// Conv[2]		| m_modelIntermediateResult[1]	| m_modelIntermediateResult[0]
	// Upsample[1]	| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
	// Conv[3]		| m_modelIntermediateResult[1]	| m_modelIntermediateResult[0]
	// Conv[4]		| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
	// Conv[5]		| m_modelIntermediateResult[1]	| m_modelIntermediateResult[0]
	// Conv[6]		| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
	// Addition		| m_modelIntermediateResult[1], m_modelOutput | m_modelOutput

	const auto bindTempResourceIfNeeded = [this, pDevice](uint64_t temporaryResourceSize,
		Binding* pBinding, Buffer::uptr& tempResource, const wchar_t* name)
	{
		if (temporaryResourceSize > 0)
		{
			tempResource = Buffer::MakeUnique();
			XUSG_N_RETURN(tempResource->Create(pDevice, temporaryResourceSize, ResourceFlag::ALLOW_UNORDERED_ACCESS,
				MemoryType::DEFAULT, 0, nullptr, 0, nullptr, MemoryFlag::NONE, name), false);
			pBinding->BindTemporary(tempResource.get());
		}

		return true;
	};

	// Upsample layers
	{
		const auto initBindingTable = Binding::MakeUnique();

		// Bind resources for initialization.
		// The ML API guarantees that initialization never uses a persistent resource.
		assert(opInitializers[OP_UP_SAMPLE]->GetPersistentResourceSize() == 0);
		XUSG_N_RETURN(initBindingTable->Create(pMLDevice, *opInitializers[OP_UP_SAMPLE], descriptorHeap,
			opInitializers[OP_UP_SAMPLE]->GetDescriptorCount(), upsampleDescriptorsIdx), false);

		// If the operator requires a persistent resource, it must be bound as output for the initializer.
		// The inputs will vary each frame, so don't bind inputs at initialization.
		for (auto i = 0u; i < c_numUpsampleLayers; ++i)
			if (m_modelUpsamplePersistentResources[i])
				initBindingTable->AppendOutput(m_modelUpsamplePersistentResources[i].get());
			else initBindingTable->AppendOutput(nullptr);
		initBindingTable->GetDispatchableBindingTable();
		XUSG_N_RETURN(bindTempResourceIfNeeded(opInitializers[OP_UP_SAMPLE]->AsOperator()->GetTemporaryResourceSize(),
			initBindingTable.get(), m_modelInitTemporaryResources[OP_UP_SAMPLE], L"UpSampleInitTemporary"), false);

		// Run initialization
		pCommandRecorder->Dispatch(pCommandList, opInitializers[OP_UP_SAMPLE]->GetDispatchable(), initBindingTable->GetDispatchableBindingTable());

		// Bind resources for execution
		for (auto i = 0u; i < c_numUpsampleLayers; ++i)
		{
			const auto descriptorOffset = upsampleDescriptorsIdx + i * upsampleOpDescriptorCount;
			m_upsampleBindings[i] = Binding::MakeUnique();
			XUSG_N_RETURN(m_upsampleBindings[i]->Create(pMLDevice, *m_upsampleOps[i], descriptorHeap,
				m_upsampleOps[i]->GetDescriptorCount(), descriptorOffset), false);

			const auto pInputResource = (i == 0) ? m_modelInput.get() : m_modelIntermediateResult[0].get();
			const auto pOutputResource = (i == 0) ? m_modelOutput.get() : m_modelIntermediateResult[1].get();

			m_upsampleBindings[i]->AppendInput(pInputResource);
			m_upsampleBindings[i]->AppendOutput(pOutputResource);
			m_upsampleBindings[i]->GetDispatchableBindingTable();
			XUSG_N_RETURN(bindTempResourceIfNeeded(m_upsampleOps[i]->GetTemporaryResourceSize(), m_upsampleBindings[i].get(),
				m_modelUpsampleTemporaryResources[i], (L"UpSampleTemporary" + to_wstring(i)).c_str()), false);

			if (m_modelUpsamplePersistentResources[i])
				m_upsampleBindings[i]->BindPersistent(m_modelUpsamplePersistentResources[i].get());
		}
	}

	// Convolution layers
	{
		const auto initBindingTable = Binding::MakeUnique();

		// Bind resources for initialization
		assert(opInitializers[OP_CONV]->GetPersistentResourceSize() == 0);
		XUSG_N_RETURN(initBindingTable->Create(pMLDevice, *opInitializers[OP_CONV], descriptorHeap,
			opInitializers[OP_CONV]->GetDescriptorCount(), convDescriptorsIdx), false);

#if ML_MANAGED_WEIGHTS
		// Bind the weight tensors at initialization instead of at execution. This lets DirectML reformat them
		// and improve performance on some hardware.
		for (auto i = 0u; i < c_numConvLayers; ++i)
		{
			const auto idx = 3 * i;
			initBindingTable->BindInputBuffer(idx, nullptr);
			initBindingTable->BindInputBuffer(idx + 1, m_modelConvFilterWeights[i].get());
			initBindingTable->BindInputBuffer(idx + 2, m_modelConvBiasWeights[i].get());

			initBindingTable->AppendInput(idx, 3);
		}
#endif

		// If the operator requires a persistent resource, it must be bound as output for the initializer.
		for (auto i = 0u; i < c_numConvLayers; ++i)
			if (m_modelConvPersistentResources[i])
				initBindingTable->AppendOutput(m_modelConvPersistentResources[i].get());
			else initBindingTable->AppendOutput(nullptr);
		initBindingTable->GetDispatchableBindingTable();
		XUSG_N_RETURN(bindTempResourceIfNeeded(opInitializers[OP_CONV]->AsOperator()->GetTemporaryResourceSize(),
			initBindingTable.get(), m_modelInitTemporaryResources[OP_CONV], L"ConvInitTemporary"), false);

		// Run initialization
		pCommandRecorder->Dispatch(pCommandList, opInitializers[OP_CONV]->GetDispatchable(), initBindingTable->GetDispatchableBindingTable());

		// Bind resources for execution
		for (auto i = 0u; i < c_numConvLayers; ++i)
		{
			const auto descriptorOffset = convDescriptorsIdx + i * convOpDescriptorCount;
			m_convBindings[i] = Binding::MakeUnique();
			XUSG_N_RETURN(m_convBindings[i]->Create(pMLDevice, *m_convOps[i], descriptorHeap,
				m_convOps[i]->GetDescriptorCount(), descriptorOffset), false);

			// See table at the beginning of the function for the mapping of ops to resources.
			const auto& pInputResource = (i == 0) ? m_modelInput.get() : ((i == 1 || i == 4 || i == 6) ?
				m_modelIntermediateResult[0].get() : m_modelIntermediateResult[1].get());
			const auto& pOutputResource = (i == 1 || i == 4 || i == 6) ?
				m_modelIntermediateResult[1].get() : m_modelIntermediateResult[0].get();

			m_convBindings[i]->AppendInput(pInputResource);
#if ML_MANAGED_WEIGHTS
			m_convBindings[i]->AppendInput(nullptr);
			m_convBindings[i]->AppendInput(nullptr);
#else
			m_convBindings[i]->AppendInput(m_modelConvFilterWeights[i].get());
			if (i == 6) m_convBindings[i]->AppendInput(nullptr); // last layer has no bias;
			else m_convBindings[i]->AppendInput(m_modelConvBiasWeights[i].get());
#endif
			m_convBindings[i]->AppendOutput(pOutputResource);
			m_convBindings[i]->GetDispatchableBindingTable();
			XUSG_N_RETURN(bindTempResourceIfNeeded(m_convOps[i]->GetTemporaryResourceSize(), m_convBindings[i].get(),
				m_modelConvTemporaryResources[i], (L"ConvTemporary" + to_wstring(i)).c_str()), false);

			if (m_modelConvPersistentResources[i])
				m_convBindings[i]->BindPersistent(m_modelConvPersistentResources[i].get());
		}
	}

	// Addition layer
	{
		const auto initBindingTable = Binding::MakeUnique();

		// Bind resources for initialization.
		assert(opInitializers[OP_ADD]->GetPersistentResourceSize() == 0);
		XUSG_N_RETURN(initBindingTable->Create(pMLDevice, *opInitializers[OP_ADD], descriptorHeap,
			opInitializers[OP_ADD]->GetDescriptorCount(), additionDescriptorsIdx), false);

		// If the operator requires a persistent resource, it must be bound as output for the initializer.
		if (m_modelAddPersistentResource)
			initBindingTable->AppendOutput(m_modelAddPersistentResource.get());
		XUSG_N_RETURN(bindTempResourceIfNeeded(opInitializers[OP_ADD]->AsOperator()->GetTemporaryResourceSize(),
			initBindingTable.get(), m_modelInitTemporaryResources[OP_ADD], L"AddInitTemporary"), false);

		// Run initialization
		pCommandRecorder->Dispatch(pCommandList, opInitializers[OP_ADD]->GetDispatchable(), initBindingTable->GetDispatchableBindingTable());

		// Bind resources for execution
		{
			m_addResidualBinding = Binding::MakeUnique();
			XUSG_N_RETURN(m_addResidualBinding->Create(pMLDevice, *m_addResidualOp, descriptorHeap,
				m_addResidualOp->GetDescriptorCount(), additionDescriptorsIdx), false);

			// m_modelOutput will already hold the result of the first upsample operation. We add the result of
			// the last convolution (the residual) to it in-place to get the final result.
			m_addResidualBinding->AppendInput(m_modelIntermediateResult[1].get());
			m_addResidualBinding->AppendInput(m_modelOutput.get());
			m_addResidualBinding->AppendOutput(m_modelOutput.get());
			XUSG_N_RETURN(bindTempResourceIfNeeded(m_addResidualOp->GetTemporaryResourceSize(), m_addResidualBinding.get(),
				m_modelAddTemporaryResource, L"AddTemporary"), false);

			if (m_modelAddPersistentResource)
				m_addResidualBinding->BindPersistent(m_modelAddPersistentResource.get());
		}
	}

	return true;
}
