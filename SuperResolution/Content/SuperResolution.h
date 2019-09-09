//--------------------------------------------------------------------------------------
// By XU, Tianchen
//--------------------------------------------------------------------------------------

#pragma once

#include "Core/XUSG.h"
#include "LoadWeights.h"

class SuperResolution
{
public:
	SuperResolution(const XUSG::Device& device, const XUSG::ML::Device& mlDevice);
	virtual ~SuperResolution();

	bool Init(XUSG::CommandList& commandList, const XUSG::ML::CommandRecorder& commandRecorder,
		uint32_t vendorId, std::vector<XUSG::Resource>& uploaders, const wchar_t* fileName);

	void ImageToTensors(const XUSG::CommandList& commandList);
	void Process(XUSG::CommandList& commandList, const XUSG::ML::CommandRecorder& commandRecorder);
	void Render(XUSG::CommandList& commandList, XUSG::RenderTarget& renderTarget);

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

	struct ImageLayout
	{
		uint32_t Height;
		uint32_t Width;
		bool UseNhwc;
	};

	bool createResources(XUSG::CommandList& commandList, const XUSG::ML::CommandRecorder& commandRecorder,
		uint32_t vendorId, std::vector<XUSG::Resource>& uploaders);
	bool createWeightTensors(const XUSG::CommandList& commandList, XUSG::ML::WeightMapType& weights,
		const char* convLayerName, const char* scaleLayerName, const char* shiftLayerName,
		const uint32_t filterSizes[4], std::vector<XUSG::Resource>& uploaders,
		XUSG::RawBuffer& filterWeightBuffer, XUSG::RawBuffer* pBiasWeightBuffer);
	bool createWeightResource(const uint32_t tensorSizes[4], XUSG::RawBuffer& resourceOut);
	bool createPipelineLayouts();
	bool createPipelines();
	bool createDescriptorTables();
	bool initResources(XUSG::CommandList& commandList, const XUSG::ML::CommandRecorder& commandRecorder);
	
	XUSG::Device		m_device;
	XUSG::ML::Device	m_mlDevice;

	XUSG::ShaderPool				m_shaderPool;
	XUSG::PipelineLayoutCache		m_pipelineLayoutCache;
	XUSG::Graphics::PipelineCache	m_graphicsPipelineCache;
	XUSG::Compute::PipelineCache	m_computePipelineCache;
	XUSG::DescriptorTableCache		m_descriptorTableCache;
	XUSG::DescriptorTableCache		m_mlDescriptorTableCache;

	// Model layer sizes and indices
	static const size_t	c_numUpsampleLayers = 2;
	static const size_t	c_numConvLayers = 7;
	static const size_t	c_numIntermediateBuffers = 2;

	// Resources
	std::shared_ptr<XUSG::ResourceBase> m_inputImage;
	XUSG::TypedBuffer	m_modelInput;
	XUSG::TypedBuffer	m_modelOutput;

	XUSG::RawBuffer	m_modelIntermediateResult[c_numIntermediateBuffers];
	XUSG::RawBuffer	m_modelConvFilterWeights[c_numConvLayers];
	XUSG::RawBuffer	m_modelConvBiasWeights[c_numConvLayers];

	XUSG::RawBuffer	m_modelUpsamplePersistentResources[c_numUpsampleLayers];
	XUSG::RawBuffer	m_modelConvPersistentResources[c_numConvLayers];
	XUSG::RawBuffer	m_modelAddPersistentResource;

	XUSG::RawBuffer	m_modelInitTemporaryResources[NUM_OP];
	XUSG::RawBuffer	m_modelUpsampleTemporaryResources[c_numUpsampleLayers];
	XUSG::RawBuffer	m_modelConvTemporaryResources[c_numConvLayers];
	XUSG::RawBuffer	m_modelAddTemporaryResource;

	XUSG::ML::TensorLayout m_tensorLayout;
	XUSG::ML::TensorDataType m_tensorDataType;

	// Operators
	XUSG::ML::Operator	m_upsampleOps[c_numUpsampleLayers];
	XUSG::ML::Operator	m_convOps[c_numConvLayers];
	XUSG::ML::Operator	m_addResidualOp;

	// Bindings
	XUSG::ML::Binding	m_upsampleBindings[c_numUpsampleLayers];
	XUSG::ML::Binding	m_convBindings[c_numConvLayers];
	XUSG::ML::Binding	m_addResidualBinding;

	// XUSG resources
	XUSG::PipelineLayout	m_pipelineLayouts[2];
	XUSG::Pipeline			m_pipelines[2];

	XUSG::DescriptorTable	m_uavSrvTable;
	XUSG::DescriptorTable	m_srvTable;

	ImageLayout m_imageLayoutIn;
	ImageLayout m_imageLayoutOut;
};
