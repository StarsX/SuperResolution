//--------------------------------------------------------------------------------------
// By XU, Tianchen
//--------------------------------------------------------------------------------------

#pragma once

#include "Core/XUSG.h"
#include "LoadWeights.h"

class SuperResolution
{
public:
	SuperResolution(const XUSG::Device::sptr& device, const XUSG::ML::Device::sptr& mlDevice);
	virtual ~SuperResolution();

	bool Init(XUSG::CommandList* pCommandList, const XUSG::ML::CommandRecorder* pCommandRecorder,
		uint32_t vendorId, std::vector<XUSG::Resource::uptr>& uploaders, const wchar_t* fileName,
		bool isFP16Supported = false);

	void ImageToTensors(XUSG::CommandList* pCommandList);
	void Process(XUSG::CommandList* pCommandList, const XUSG::ML::CommandRecorder* pCommandRecorder);
	void Render(XUSG::CommandList* pCommandList, XUSG::RenderTarget& renderTarget);

	uint32_t GetOutWidth() const;
	uint32_t GetOutHeight() const;

protected:
	enum OpType : uint8_t
	{
		OP_UP_SAMPLE,
		OP_CONV,
		OP_ADD,

		NUM_OP
	};

	enum CbvSrvUavPoolIndex : uint8_t
	{
		GRAPHICS_POOL,
		ML_POOL
	};

	struct ImageLayout
	{
		uint32_t Height;
		uint32_t Width;
		bool UseNhwc;
	};

	bool createResources(XUSG::CommandList* pCommandList, const XUSG::ML::CommandRecorder* pCommandRecorder,
		uint32_t vendorId, std::vector<XUSG::Resource::uptr>& uploaders, bool isFP16Supported);
	bool createWeightTensors(XUSG::CommandList* pCommandList, XUSG::ML::WeightMapType& weights,
		const char* convLayerName, const char* scaleLayerName, const char* shiftLayerName,
		const uint32_t filterSizes[4], std::vector<XUSG::Resource::uptr>& uploaders,
		XUSG::RawBuffer::uptr& filterWeightBuffer, XUSG::RawBuffer::uptr& biasWeightBuffer);
	bool createWeightResource(const uint32_t tensorSizes[4], XUSG::RawBuffer::uptr& resourceOut);
	bool createPipelineLayouts();
	bool createPipelines();
	bool createDescriptorTables();
	bool initResources(XUSG::CommandList* pCommandList, const XUSG::ML::CommandRecorder* pCommandRecorder);
	
	XUSG::Device::sptr		m_device;
	XUSG::ML::Device::sptr	m_mlDevice;

	XUSG::ShaderPool::uptr				m_shaderPool;
	XUSG::PipelineLayoutCache::uptr		m_pipelineLayoutCache;
	XUSG::Graphics::PipelineCache::uptr	m_graphicsPipelineCache;
	XUSG::Compute::PipelineCache::uptr	m_computePipelineCache;
	XUSG::DescriptorTableCache::uptr	m_descriptorTableCache;

	// Model layer sizes and indices
	static const size_t	c_numUpsampleLayers = 2;
	static const size_t	c_numConvLayers = 7;
	static const size_t	c_numIntermediateBuffers = 2;

	// Resources
	XUSG::Texture::sptr		m_inputImage;
	XUSG::TypedBuffer::uptr	m_modelInput;
	XUSG::TypedBuffer::uptr	m_modelOutput;

	XUSG::RawBuffer::uptr	m_modelIntermediateResult[c_numIntermediateBuffers];
	XUSG::RawBuffer::uptr	m_modelConvFilterWeights[c_numConvLayers];
	XUSG::RawBuffer::uptr	m_modelConvBiasWeights[c_numConvLayers];

	XUSG::RawBuffer::uptr	m_modelUpsamplePersistentResources[c_numUpsampleLayers];
	XUSG::RawBuffer::uptr	m_modelConvPersistentResources[c_numConvLayers];
	XUSG::RawBuffer::uptr	m_modelAddPersistentResource;

	XUSG::RawBuffer::uptr	m_modelInitTemporaryResources[NUM_OP];
	XUSG::RawBuffer::uptr	m_modelUpsampleTemporaryResources[c_numUpsampleLayers];
	XUSG::RawBuffer::uptr	m_modelConvTemporaryResources[c_numConvLayers];
	XUSG::RawBuffer::uptr	m_modelAddTemporaryResource;

	XUSG::ML::TensorLayout m_tensorLayout;
	XUSG::ML::TensorDataType m_tensorDataType;

	// Operators
	XUSG::ML::Operator::sptr m_upsampleOps[c_numUpsampleLayers];
	XUSG::ML::Operator::sptr m_convOps[c_numConvLayers];
	XUSG::ML::Operator::sptr m_addResidualOp;

	// Bindings
	XUSG::ML::Binding::uptr	m_upsampleBindings[c_numUpsampleLayers];
	XUSG::ML::Binding::uptr	m_convBindings[c_numConvLayers];
	XUSG::ML::Binding::uptr	m_addResidualBinding;

	// XUSG resources
	XUSG::PipelineLayout	m_pipelineLayouts[2];
	XUSG::Pipeline			m_pipelines[2];

	XUSG::DescriptorTable	m_uavSrvTable;
	XUSG::DescriptorTable	m_srvTable;

	ImageLayout m_imageLayoutIn;
	ImageLayout m_imageLayoutOut;
};
