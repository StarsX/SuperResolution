//--------------------------------------------------------------------------------------
// By XU, Tianchen
//--------------------------------------------------------------------------------------

#include "SuperResolution.h"
#include "Advanced/XUSGDDSLoader.h"

using namespace std;
using namespace XUSG;
using namespace XUSG::ML;

#define SizeOfInUint32(obj)	DIV_UP(sizeof(obj), sizeof(uint32_t))

SuperResolution::SuperResolution(const XUSG::Device& device, const ML::Device& mlDevice) :
	m_device(device),
	m_mlDevice(mlDevice),
	m_tensorLayout(TensorLayout::DEFAULT),
	m_tensorDataType(TensorDataType::FLOAT32)
{
	m_pipelineLayoutCache.SetDevice(m_device);
	m_graphicsPipelineCache.SetDevice(m_device);
	m_computePipelineCache.SetDevice(m_device);
	m_descriptorTableCache.SetDevice(device);
}

SuperResolution::~SuperResolution()
{
}

bool SuperResolution::Init(CommandList& commandList, const CommandRecorder& commandRecorder,
	uint32_t vendorId, vector<Resource>& uploaders, const wchar_t* fileName)
{
	// Load input image
	{
		DDS::Loader textureLoader;
		DDS::AlphaMode alphaMode;

		uploaders.push_back(nullptr);
		N_RETURN(textureLoader.CreateTextureFromFile(m_device, commandList, fileName,
			8192, false, m_inputImage, uploaders.back(), &alphaMode), false);
	}

	m_imageLayoutIn.Width = static_cast<uint32_t>(m_inputImage->GetResource()->GetDesc().Width);
	m_imageLayoutIn.Height = m_inputImage->GetResource()->GetDesc().Height;

	N_RETURN(createResources(commandList, commandRecorder, vendorId, uploaders), false);
	N_RETURN(initResources(commandList, commandRecorder), false);

	N_RETURN(createPipelineLayouts(), false);
	N_RETURN(createPipelines(), false);
	N_RETURN(createDescriptorTables(), false);

	m_imageLayoutOut.Width = m_imageLayoutIn.Width << 1;
	m_imageLayoutOut.Height = m_imageLayoutIn.Height << 1;
	m_imageLayoutOut.UseNhwc = m_imageLayoutIn.UseNhwc = m_tensorLayout == TensorLayout::NHWC;

	return true;
}

void SuperResolution::ImageToTensors(const CommandList& commandList)
{
	const DescriptorPool descriptorPools[] = { m_descriptorTableCache.GetDescriptorPool(CBV_SRV_UAV_POOL, GRAPHICS_POOL) };
	commandList.SetDescriptorPools(static_cast<uint32_t>(size(descriptorPools)), descriptorPools);

	ResourceBarrier barrier;
	auto numBarriers = m_modelInput.SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);
	commandList.Barrier(numBarriers, &barrier);

	commandList.SetComputePipelineLayout(m_pipelineLayouts[0]);
	commandList.SetCompute32BitConstants(0, 3, &m_imageLayoutIn);
	commandList.SetComputeDescriptorTable(1, m_uavSrvTable);
	commandList.SetPipelineState(m_pipelines[0]);
	commandList.Dispatch(DIV_UP(m_imageLayoutIn.Width, 8), DIV_UP(m_imageLayoutIn.Height, 8), 1);

	numBarriers = m_modelInput.SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);
	commandList.Barrier(numBarriers, &barrier);
}

void SuperResolution::Process(CommandList& commandList, const CommandRecorder& commandRecorder)
{
	const DescriptorPool descriptorPools[] = { m_descriptorTableCache.GetDescriptorPool(CBV_SRV_UAV_POOL, ML_POOL) };
	commandList.SetDescriptorPools(static_cast<uint32_t>(size(descriptorPools)), descriptorPools);

	ResourceBarrier barrier;
	auto numBarriers = m_modelOutput.SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);
	commandList.Barrier(numBarriers, &barrier);

	// Create an upsampled (nearest neighbor) version of the image first
	commandRecorder->RecordDispatch(commandList.GetCommandList().get(), m_upsampleOps[0].GetDispatchable().get(),
		m_upsampleBindings[0].GetDispatchableBindingTable().get());
	// No UAV barrier is required here since we don't use the result right away.

	// Run the intermediate model steps: 3 convolutions (with premultiplied batch normalization
	// baked into the weights), an upsample, 3 convolutions w/ premultiplied batch norm, 1 final convolution.
	// This generates a residual image.
	for (auto i = 0u; i < c_numConvLayers; ++i)
	{
		// Convolution
		commandRecorder->RecordDispatch(commandList.GetCommandList().get(), m_convOps[i].GetDispatchable().get(),
			m_convBindings[i].GetDispatchableBindingTable().get());
		commandList.Barrier(1, &ResourceBarrier::UAV(nullptr));

		if (i == 2)
		{
			// Intermediate upsample
			commandRecorder->RecordDispatch(commandList.GetCommandList().get(), m_upsampleOps[1].GetDispatchable().get(),
				m_upsampleBindings[1].GetDispatchableBindingTable().get());
			commandList.Barrier(1, &ResourceBarrier::UAV(nullptr));
		}
	}

	// Add the residual image to the original nearest-neighbor upscale
	commandRecorder->RecordDispatch(commandList.GetCommandList().get(), m_addResidualOp.GetDispatchable().get(),
		m_addResidualBinding.GetDispatchableBindingTable().get());
}

void SuperResolution::Render(CommandList& commandList, RenderTarget& renderTarget)
{
	const DescriptorPool descriptorPools[] = { m_descriptorTableCache.GetDescriptorPool(CBV_SRV_UAV_POOL, GRAPHICS_POOL) };
	commandList.SetDescriptorPools(static_cast<uint32_t>(size(descriptorPools)), descriptorPools);

	ResourceBarrier barriers[2];
	auto numBarriers = renderTarget.SetBarrier(barriers, ResourceState::RENDER_TARGET);
	numBarriers = m_modelOutput.SetBarrier(barriers, ResourceState::PIXEL_SHADER_RESOURCE, numBarriers);
	commandList.Barrier(numBarriers, barriers);

	// Set pipeline
	commandList.SetGraphicsPipelineLayout(m_pipelineLayouts[1]);
	commandList.SetGraphics32BitConstants(0, SizeOfInUint32(m_imageLayoutOut), &m_imageLayoutOut);
	commandList.SetGraphicsDescriptorTable(1, m_srvTable);
	commandList.SetPipelineState(m_pipelines[1]);

	// Set viewport
	const auto& width = m_imageLayoutOut.Width;
	const auto& height = m_imageLayoutOut.Height;
	Viewport viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
	RectRange scissorRect(0, 0, width, height);
	commandList.RSSetViewports(1, &viewport);
	commandList.RSSetScissorRects(1, &scissorRect);

	commandList.OMSetRenderTargets(1, &renderTarget.GetRTV());

	commandList.IASetPrimitiveTopology(PrimitiveTopology::TRIANGLELIST);
	commandList.Draw(3, 1, 0, 0);
}

uint32_t SuperResolution::GetOutWidth() const
{
	return m_imageLayoutOut.Width;
}

uint32_t SuperResolution::GetOutHeight() const
{
	return m_imageLayoutOut.Height;
}

bool SuperResolution::createResources(CommandList& commandList, const CommandRecorder& commandRecorder,
	uint32_t vendorId, vector<Resource>& uploaders)
{
	// ML device
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
		DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT fp16Query = { DML_TENSOR_DATA_TYPE_FLOAT16 };
		DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT fp16Supported = {};
		V_RETURN(m_mlDevice->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(fp16Query),
			&fp16Query, sizeof(fp16Supported), &fp16Supported), cerr, false);

		m_tensorDataType = fp16Supported.IsSupported ? TensorDataType::FLOAT16 : TensorDataType::FLOAT32;
		format = fp16Supported.IsSupported ? Format::R16_FLOAT : Format::R32_FLOAT;
		dataStride = fp16Supported.IsSupported ? sizeof(uint16_t) : sizeof(float);
	}

	uint64_t modelInputBufferSize = 0;
	uint64_t modelOutputBufferSize = 0;
	uint64_t intermediateBufferMaxSize[] = { 0, 0 };

	// ML operator resources--implementation of the super-resolution model
	{
		ML::Util mlUtil(m_mlDevice, m_tensorDataType, m_tensorLayout);

		// Create an upscaled (nearest neighbor) version of the image first
		const auto& width = m_imageLayoutIn.Width;
		const auto& height = m_imageLayoutIn.Height;
		const uint32_t modelInputSizes[] = { 1, 3, height, width };
		uint32_t upscaledInputSizes[4];
		N_RETURN(mlUtil.CreateUpsampleLayer(modelInputSizes, modelInputBufferSize,
			modelOutputBufferSize, upscaledInputSizes, m_upsampleOps[0]), false);

		// Create the residual with three convolutions, an upsample, and four more convolutions
		WeightMapType weights;
		if (!LoadWeights("Assets/weights.bin", weights))
			throw std::exception("loadWeights");

		uint32_t filterSizes[] = { 32, 3, 5, 5 };
		uint32_t intermediateInputSizes[2][4];
		N_RETURN(mlUtil.CreateConvolutionLayer(modelInputSizes, filterSizes, true, modelInputBufferSize,
			intermediateBufferMaxSize[0], intermediateInputSizes[0], m_convOps[0]), false);
		N_RETURN(createWeightTensors(commandList, weights, "conv1/weights", "conv1/BatchNorm/scale",
			"conv1/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[0],
			&m_modelConvBiasWeights[0]), false);

		// Which intermediate resource to use as input for the current operation. The other will be
		// used as output. Then the next op will swap the order.
		auto inputIndex = 0ui8;

		filterSizes[0] = 64;	// output filters
		filterSizes[1] = 32;	// input channels
		filterSizes[2] = 3;		// filter height
		filterSizes[3] = 3;		// filter width
		N_RETURN(mlUtil.CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true,
			intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], m_convOps[1]), false);
		N_RETURN(createWeightTensors(commandList, weights, "conv2/weights", "conv2/BatchNorm/scale",
			"conv2/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[1],
			&m_modelConvBiasWeights[1]), false);
		inputIndex = 1 - inputIndex;

		filterSizes[1] = 64;
		N_RETURN(mlUtil.CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true,
			intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], m_convOps[2]), false);
		N_RETURN(createWeightTensors(commandList, weights, "conv3/weights", "conv3/BatchNorm/scale",
			"conv3/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[2],
			&m_modelConvBiasWeights[2]), false);
		inputIndex = 1 - inputIndex;

		N_RETURN(mlUtil.CreateUpsampleLayer(intermediateInputSizes[inputIndex], intermediateBufferMaxSize[inputIndex],
			intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], m_upsampleOps[1]), false);
		inputIndex = 1 - inputIndex;

		filterSizes[0] = 32;
		filterSizes[2] = 5;
		filterSizes[3] = 5;
		N_RETURN(mlUtil.CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true,
			intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], m_convOps[3]), false);
		N_RETURN(createWeightTensors(commandList, weights, "conv_up1/conv/weights",
			"conv_up1/conv/BatchNorm/scale", "conv_up1/conv/BatchNorm/shift", filterSizes,
			uploaders, m_modelConvFilterWeights[3], &m_modelConvBiasWeights[3]), false);
		inputIndex = 1 - inputIndex;

		filterSizes[1] = 32;
		filterSizes[2] = 3;
		filterSizes[3] = 3;
		N_RETURN(mlUtil.CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true,
			intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], m_convOps[4]), false);
		N_RETURN(createWeightTensors(commandList, weights, "conv4/weights", "conv4/BatchNorm/scale",
			"conv4/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[4],
			&m_modelConvBiasWeights[4]), false);
		inputIndex = 1 - inputIndex;

		N_RETURN(mlUtil.CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true,
			intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], m_convOps[5]), false);
		N_RETURN(createWeightTensors(commandList, weights, "conv5/weights", "conv5/BatchNorm/scale",
			"conv5/BatchNorm/shift", filterSizes, uploaders, m_modelConvFilterWeights[5],
			&m_modelConvBiasWeights[5]), false);
		inputIndex = 1 - inputIndex;

		filterSizes[0] = 3;
		N_RETURN(mlUtil.CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, false,
			intermediateBufferMaxSize[inputIndex], intermediateBufferMaxSize[1 - inputIndex],
			intermediateInputSizes[1 - inputIndex], m_convOps[6]), false);
		N_RETURN(createWeightTensors(commandList, weights, "conv6/weights", nullptr, nullptr,
			filterSizes, uploaders, m_modelConvFilterWeights[6], nullptr), false);
		inputIndex = 1 - inputIndex;

		// Finally add the residual to the original upsampled image
		assert(memcmp(upscaledInputSizes, intermediateInputSizes[inputIndex], 4 * dataStride) == 0);

		N_RETURN(mlUtil.CreateAdditionLayer(upscaledInputSizes, m_addResidualOp), false);
	}

	// Buffers for ML inputs and outputs
	{
		// Resource for input tensor
		auto numElements = static_cast<uint32_t>(modelInputBufferSize / dataStride);
		N_RETURN(m_modelInput.Create(m_device, numElements, dataStride, format,
			ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0,
			nullptr, 1, nullptr, L"InputBuffer"), false);

		// Model result tensor is 2x larger in both dimensions
		numElements = static_cast<uint32_t>(modelOutputBufferSize / dataStride);
		N_RETURN(m_modelOutput.Create(m_device, numElements, dataStride, format,
			ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 1,
			nullptr, 0, nullptr, L"OutputBuffer"), false);

		// Create two resources for intermediate layer results. Each layer will ping-pong between these. They're each large
		// enough to hold the largest intermediate result required.
		for (auto i = 0ui8; i < 2; ++i)
			N_RETURN(m_modelIntermediateResult[i].Create(m_device, intermediateBufferMaxSize[i],
				ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0, nullptr, 0,
				nullptr, (L"IntermediateResultBuffer" + to_wstring(i)).c_str()), false);
	}

	return true;
}

bool SuperResolution::createWeightTensors(const CommandList& commandList, WeightMapType& weights, const char* convLayerName,
	const char* scaleLayerName, const char* shiftLayerName, const uint32_t filterSizes[4],
	vector<Resource>& uploaders, RawBuffer& filterWeightBuffer, RawBuffer* pBiasWeightBuffer)
{
	ML::Util mlUtil(m_mlDevice, m_tensorDataType, m_tensorLayout);
	vector<uint8_t> filterWeights;
	vector<uint8_t> biasWeights;

	mlUtil.CreateWeightTensors(weights, convLayerName, scaleLayerName, shiftLayerName,
		filterSizes, filterWeights, biasWeights);

	auto useScaleShift = true;
	if (scaleLayerName == nullptr)
	{
		assert(shiftLayerName == nullptr);
		useScaleShift = false;
	}

	uploaders.push_back(nullptr);
	N_RETURN(createWeightResource(filterSizes, filterWeightBuffer), false);
	N_RETURN(filterWeightBuffer.Upload(commandList, uploaders.back(), ResourceState::UNORDERED_ACCESS,
		filterWeights.data(), filterWeights.size()), false);

	if (useScaleShift)
	{
		const uint32_t biasSizes[] = { 1, filterSizes[0], 1, 1 };	// One bias per output channel
		uploaders.push_back(nullptr);
		N_RETURN(createWeightResource(biasSizes, *pBiasWeightBuffer), false);
		N_RETURN(pBiasWeightBuffer->Upload(commandList, uploaders.back(), ResourceState::UNORDERED_ACCESS,
			biasWeights.data(), biasWeights.size()), false);

		// The scale weights will be premultiplied into the filter weights, so they don't need
		// a separate resource.
	}
	//else if (pBiasWeightBuffer) pBiasWeightBuffer = nullptr;

	return true;
}

bool SuperResolution::createWeightResource(const uint32_t tensorSizes[4], RawBuffer& resourceOut)
{
	uint32_t strides[4];
	ML::Util::GetStrides(tensorSizes, m_tensorLayout, strides);
	
	Tensor tensor;
	const auto bufferSize = tensor.Create(m_tensorDataType, static_cast<uint32_t>(size(strides)), tensorSizes, strides);

	return resourceOut.Create(m_device, bufferSize, ResourceFlag::ALLOW_UNORDERED_ACCESS,
		MemoryType::DEFAULT, 0, nullptr, 0, nullptr, L"WeightBuffer");
}

bool SuperResolution::createPipelineLayouts()
{
	{
		XUSG::Util::PipelineLayout pipelineLayout;
		pipelineLayout.SetRange(0, DescriptorType::CONSTANT, SizeOfInUint32(ImageLayout), 0);
		pipelineLayout.SetRange(1, DescriptorType::UAV, 1, 0,
			0, DescriptorRangeFlag::DATA_STATIC_WHILE_SET_AT_EXECUTE);
		pipelineLayout.SetRange(1, DescriptorType::SRV, 1, 0,
			0, DescriptorRangeFlag::DATA_STATIC);
		X_RETURN(m_pipelineLayouts[0], pipelineLayout.GetPipelineLayout(m_pipelineLayoutCache,
			PipelineLayoutFlag::DENY_VERTEX_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_HULL_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_DOMAIN_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_GEOMETRY_SHADER_ROOT_ACCESS |
			PipelineLayoutFlag::DENY_PIXEL_SHADER_ROOT_ACCESS,
			L"ImageToTensorLayout"), false);
	}

	{
		XUSG::Util::PipelineLayout pipelineLayout;
		pipelineLayout.SetRange(0, DescriptorType::CONSTANT, SizeOfInUint32(ImageLayout), 0);
		pipelineLayout.SetRange(1, DescriptorType::SRV, 1, 0);
		pipelineLayout.SetShaderStage(0, Shader::Stage::PS);
		pipelineLayout.SetShaderStage(1, Shader::Stage::PS);
		X_RETURN(m_pipelineLayouts[1], pipelineLayout.GetPipelineLayout(m_pipelineLayoutCache,
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
		const auto cs = m_shaderPool.CreateShader(Shader::Stage::CS, 0, L"CSImageToTensor.cso");
		N_RETURN(cs, false);

		Compute::State state;
		state.SetPipelineLayout(m_pipelineLayouts[0]);
		state.SetShader(cs);
		X_RETURN(m_pipelines[0], state.GetPipeline(m_computePipelineCache, L"ImageToTensor"), false);
	}

	{
		const auto vs = m_shaderPool.CreateShader(Shader::Stage::VS, 0, L"VSTensorToImage.cso");
		const auto ps = m_shaderPool.CreateShader(Shader::Stage::PS, 0, L"PSTensorToImage.cso");
		N_RETURN(vs, false);
		N_RETURN(ps, false);

		Graphics::State state;
		state.SetPipelineLayout(m_pipelineLayouts[1]);
		state.SetShader(Shader::Stage::VS, vs);
		state.SetShader(Shader::Stage::PS, ps);
		state.DSSetState(Graphics::DEPTH_STENCIL_NONE, m_graphicsPipelineCache);
		state.IASetPrimitiveTopologyType(PrimitiveTopologyType::TRIANGLE);
		state.OMSetNumRenderTargets(1);
		state.OMSetRTVFormat(0, Format::R8G8B8A8_UNORM);
		X_RETURN(m_pipelines[1], state.GetPipeline(m_graphicsPipelineCache, L"TensorToImage"), false);
	}

	return true;
}

bool SuperResolution::createDescriptorTables()
{
	{
		const Descriptor descriptors[] =
		{
			m_modelInput.GetUAV(),
			m_inputImage->GetSRV()
		};
		XUSG::Util::DescriptorTable uavSrvTable;
		uavSrvTable.SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors, GRAPHICS_POOL);
		X_RETURN(m_uavSrvTable, uavSrvTable.GetCbvSrvUavTable(m_descriptorTableCache), false);
	}

	{
		XUSG::Util::DescriptorTable srvTable;
		srvTable.SetDescriptors(0, 1, &m_modelOutput.GetSRV(), GRAPHICS_POOL);
		X_RETURN(m_srvTable, srvTable.GetCbvSrvUavTable(m_descriptorTableCache), false);
	}

	return true;
}

bool SuperResolution::initResources(CommandList& commandList,// const CommandAllocator& commandAllocator,
	const CommandRecorder& commandRecorder)
{
	//commandList.Reset(commandAllocator, nullptr);

	// Create operator initializers and descriptor heap for binding
	uint32_t upsampleOpDescriptorCount, convOpDescriptorCount, additionOpDescriptorCount;
	uint32_t upsampleDescriptorsIdx, convDescriptorsIdx, additionDescriptorsIdx;

	OperatorInitializer opInitializers[NUM_OP];
	DescriptorPool descriptorPool;
	{
		// The same descriptor heap will be used for both initializing and executing operators. These each happen
		// at different times, so we reuse the same descriptor slots. GetDescriptorCount() ensures there are enough
		// slots for both cases.
		N_RETURN(opInitializers[OP_UP_SAMPLE].Create(m_mlDevice, m_upsampleOps, c_numUpsampleLayers), false);
		upsampleOpDescriptorCount = opInitializers[OP_UP_SAMPLE].GetDescriptorCount();
		
		N_RETURN(opInitializers[OP_CONV].Create(m_mlDevice, m_convOps, c_numConvLayers), false);
		convOpDescriptorCount = opInitializers[OP_CONV].GetDescriptorCount();

		N_RETURN(opInitializers[OP_ADD].Create(m_mlDevice, &m_addResidualOp, 1), false);
		additionOpDescriptorCount = opInitializers[OP_ADD].GetDescriptorCount();

		upsampleDescriptorsIdx = 0;
		convDescriptorsIdx = upsampleDescriptorsIdx + upsampleOpDescriptorCount * c_numUpsampleLayers;
		additionDescriptorsIdx = convDescriptorsIdx + convOpDescriptorCount * c_numConvLayers;
		const auto descriptorCount = additionDescriptorsIdx + additionOpDescriptorCount;

		N_RETURN(m_descriptorTableCache.AllocateDescriptorPool(CBV_SRV_UAV_POOL, descriptorCount, ML_POOL), false);
		descriptorPool = m_descriptorTableCache.GetDescriptorPool(CBV_SRV_UAV_POOL, ML_POOL);

		// Operator initialization dispatches will use this heap right away
		const DescriptorPool descriptorPools[] = { descriptorPool };
		commandList.SetDescriptorPools(static_cast<uint32_t>(size(descriptorPools)), descriptorPools);
	}

	// Create any persistent resources required for the operators.
	{
		for (auto i = 0u; i < c_numUpsampleLayers; ++i)
		{
			const auto persistentResourceSize = m_upsampleOps[i].GetPersistentResourceSize();
			if (persistentResourceSize > 0)
				N_RETURN(m_modelUpsamplePersistentResources[i].Create(m_device, persistentResourceSize,
					ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0, nullptr, 0,
					nullptr, (L"UpSamplePersistent" + to_wstring(i)).c_str()), false);
		}

		for (auto i = 0u; i < c_numConvLayers; ++i)
		{
			const auto persistentResourceSize = m_convOps[i].GetPersistentResourceSize();
			if (persistentResourceSize > 0)
				N_RETURN(m_modelConvPersistentResources[i].Create(m_device, persistentResourceSize,
					ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0, nullptr, 0,
					nullptr, (L"ConvPersistent" + to_wstring(i)).c_str()), false);
		}

		{
			const auto persistentResourceSize = m_addResidualOp.GetPersistentResourceSize();
			if (persistentResourceSize > 0)
				N_RETURN(m_modelAddPersistentResource.Create(m_device, persistentResourceSize,
					ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT, 0, nullptr,
					0, nullptr, L"AddPersistent"), false);
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

	const auto bindTempResourceIfNeeded = [this](uint64_t temporaryResourceSize,
		Binding& binding, RawBuffer& tempResource, const wchar_t* name)
	{
		if (temporaryResourceSize > 0)
		{
			N_RETURN(tempResource.Create(m_device, temporaryResourceSize, ResourceFlag::ALLOW_UNORDERED_ACCESS,
				MemoryType::DEFAULT, 0, nullptr, 0, nullptr, name), false);
			binding.BindTemporary(tempResource);
		}

		return true;
	};

	// Upsample layers
	{
		Binding initBindingTable;

		// Bind resources for initialization.
		// The ML API guarantees that initialization never uses a persistent resource.
		assert(opInitializers[OP_UP_SAMPLE].GetPersistentResourceSize() == 0);
		N_RETURN(initBindingTable.Create(m_mlDevice, opInitializers[OP_UP_SAMPLE], descriptorPool,
			opInitializers[OP_UP_SAMPLE].GetDescriptorCount(), upsampleDescriptorsIdx), false);

		// If the operator requires a persistent resource, it must be bound as output for the initializer.
		// The inputs will vary each frame, so don't bind inputs at initialization.
		for (auto i = 0u; i < c_numUpsampleLayers; ++i)
			if (m_modelUpsamplePersistentResources[i].GetResource())
				initBindingTable.AppendOutput(m_modelUpsamplePersistentResources[i]);
			else initBindingTable.AppendOutput(nullptr);
		initBindingTable.GetDispatchableBindingTable();
		N_RETURN(bindTempResourceIfNeeded(opInitializers[OP_UP_SAMPLE].Operator::GetTemporaryResourceSize(),
			initBindingTable, m_modelInitTemporaryResources[OP_UP_SAMPLE], L"UpSampleInitTemporary"), false);

		// Run initialization
		commandRecorder->RecordDispatch(commandList.GetCommandList().get(),
			opInitializers[OP_UP_SAMPLE].GetDispatchable().get(),
			initBindingTable.GetDispatchableBindingTable().get());

		// Bind resources for execution
		for (auto i = 0u; i < c_numUpsampleLayers; ++i)
		{
			const auto descriptorOffset = upsampleDescriptorsIdx + i * upsampleOpDescriptorCount;
			N_RETURN(m_upsampleBindings[i].Create(m_mlDevice, m_upsampleOps[i], descriptorPool,
				m_upsampleOps[i].GetDescriptorCount(), descriptorOffset), false);

			const auto& inputResource = (i == 0) ? m_modelInput : m_modelIntermediateResult[0];
			const auto& outputResource = (i == 0) ? m_modelOutput : m_modelIntermediateResult[1];

			m_upsampleBindings[i].AppendInput(inputResource);
			m_upsampleBindings[i].AppendOutput(outputResource);
			m_upsampleBindings[i].GetDispatchableBindingTable();
			N_RETURN(bindTempResourceIfNeeded(m_upsampleOps[i].GetTemporaryResourceSize(), m_upsampleBindings[i],
				m_modelUpsampleTemporaryResources[i], (L"UpSampleTemporary" + to_wstring(i)).c_str()), false);

			if (m_modelUpsamplePersistentResources[i].GetResource())
				m_upsampleBindings[i].BindPersistent(m_modelUpsamplePersistentResources[i]);
		}
	}

	// Convolution layers
	{
		Binding initBindingTable;

		// Bind resources for initialization
		assert(opInitializers[OP_CONV].GetPersistentResourceSize() == 0);
		N_RETURN(initBindingTable.Create(m_mlDevice, opInitializers[OP_CONV], descriptorPool,
			opInitializers[OP_CONV].GetDescriptorCount(), convDescriptorsIdx), false);

#if ML_MANAGED_WEIGHTS
		// Bind the weight tensors at initialization instead of at execution. This lets DirectML reformat them
		// and improve performance on some hardware.
		for (auto i = 0u; i < c_numConvLayers; ++i)
		{
			const auto idx = 3 * i;
			initBindingTable.BindInputBuffer(idx, nullptr);
			initBindingTable.BindInputBuffer(idx + 1, m_modelConvFilterWeights[i]);
			initBindingTable.BindInputBuffer(idx + 2, m_modelConvBiasWeights[i]);

			initBindingTable.AppendInput(idx, 3);
		}
#endif

		// If the operator requires a persistent resource, it must be bound as output for the initializer.
		for (auto i = 0u; i < c_numConvLayers; ++i)
			if (m_modelConvPersistentResources[i].GetResource())
				initBindingTable.AppendOutput(m_modelConvPersistentResources[i]);
			else initBindingTable.AppendOutput(nullptr);
		initBindingTable.GetDispatchableBindingTable();
		N_RETURN(bindTempResourceIfNeeded(opInitializers[OP_CONV].Operator::GetTemporaryResourceSize(),
			initBindingTable, m_modelInitTemporaryResources[OP_CONV], L"ConvInitTemporary"), false);

		// Run initialization
		commandRecorder->RecordDispatch(commandList.GetCommandList().get(),
			opInitializers[OP_CONV].GetDispatchable().get(),
			initBindingTable.GetDispatchableBindingTable().get());

		// Bind resources for execution
		for (auto i = 0u; i < c_numConvLayers; ++i)
		{
			const auto descriptorOffset = convDescriptorsIdx + i * convOpDescriptorCount;
			N_RETURN(m_convBindings[i].Create(m_mlDevice, m_convOps[i], descriptorPool,
				m_convOps[i].GetDescriptorCount(), descriptorOffset), false);

			// See table at the beginning of the function for the mapping of ops to resources.
			const auto& inputResource = (i == 0) ? m_modelInput : ((i == 1 || i == 4 || i == 6) ?
				m_modelIntermediateResult[0] : m_modelIntermediateResult[1]);
			const auto& outputResource = (i == 1 || i == 4 || i == 6) ?
				m_modelIntermediateResult[1] : m_modelIntermediateResult[0];

			m_convBindings[i].AppendInput(inputResource);
#if ML_MANAGED_WEIGHTS
			m_convBindings[i].AppendInput(nullptr);
			m_convBindings[i].AppendInput(nullptr);
#else
			m_convBindings[i].AppendInput(m_modelConvFilterWeights[i]);
			if (i == 6) m_convBindings[i].AppendInput(nullptr); // last layer has no bias;
			else m_convBindings[i].AppendInput(m_modelConvBiasWeights[i]);
#endif
			m_convBindings[i].AppendOutput(outputResource);
			m_convBindings[i].GetDispatchableBindingTable();
			N_RETURN(bindTempResourceIfNeeded(m_convOps[i].GetTemporaryResourceSize(), m_convBindings[i],
				m_modelConvTemporaryResources[i], (L"ConvTemporary" + to_wstring(i)).c_str()), false);

			if (m_modelConvPersistentResources[i].GetResource())
				m_convBindings[i].BindPersistent(m_modelConvPersistentResources[i]);
		}
	}

	// Addition layer
	{
		Binding initBindingTable;

		// Bind resources for initialization.
		assert(opInitializers[OP_ADD].GetPersistentResourceSize() == 0);
		N_RETURN(initBindingTable.Create(m_mlDevice, opInitializers[OP_ADD], descriptorPool,
			opInitializers[OP_ADD].GetDescriptorCount(), additionDescriptorsIdx), false);

		// If the operator requires a persistent resource, it must be bound as output for the initializer.
		if (m_modelAddPersistentResource.GetResource())
			initBindingTable.AppendOutput(m_modelAddPersistentResource);
		N_RETURN(bindTempResourceIfNeeded(opInitializers[OP_ADD].Operator::GetTemporaryResourceSize(),
			initBindingTable, m_modelInitTemporaryResources[OP_ADD], L"AddInitTemporary"), false);

		// Run initialization
		commandRecorder->RecordDispatch(commandList.GetCommandList().get(),
			opInitializers[OP_ADD].GetDispatchable().get(),
			initBindingTable.GetDispatchableBindingTable().get());

		// Bind resources for execution
		{
			N_RETURN(m_addResidualBinding.Create(m_mlDevice, m_addResidualOp, descriptorPool,
				m_addResidualOp.GetDescriptorCount(), additionDescriptorsIdx), false);

			// m_modelOutput will already hold the result of the first upsample operation. We add the result of
			// the last convolution (the residual) to it in-place to get the final result.
			m_addResidualBinding.AppendInput(m_modelIntermediateResult[1]);
			m_addResidualBinding.AppendInput(m_modelOutput);
			m_addResidualBinding.AppendOutput(m_modelOutput);
			N_RETURN(bindTempResourceIfNeeded(m_addResidualOp.GetTemporaryResourceSize(), m_addResidualBinding,
				m_modelAddTemporaryResource, L"AddTemporary"), false);

			if (m_modelAddPersistentResource.GetResource())
				m_addResidualBinding.BindPersistent(m_modelAddPersistentResource);
		}
	}

	return true;
}
