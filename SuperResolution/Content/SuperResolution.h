//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "Core/XUSG.h"
#include "LoadWeights.h"

class SuperResolution
{
public:
	SuperResolution();
	virtual ~SuperResolution();

	bool Init(XUSG::CommandList* pCommandList, const XUSG::ML::CommandRecorder* pCommandRecorder,
		const XUSG::DescriptorTableLib::sptr& descriptorTableLib, uint32_t vendorId,
		std::vector<XUSG::Resource::uptr>& uploaders, const char* fileName, bool isFP16Supported = false);

	void ImageToTensors(XUSG::CommandList* pCommandList);
	void Process(XUSG::CommandList* pCommandList, const XUSG::ML::CommandRecorder* pCommandRecorder);
	void Render(XUSG::CommandList* pCommandList, XUSG::RenderTarget* pRenderTarget);

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

	bool createResources(XUSG::CommandList* pCommandList, const XUSG::ML::CommandRecorder* pCommandRecorder,
		uint32_t vendorId, std::vector<XUSG::Resource::uptr>& uploaders, bool isFP16Supported);
	bool createWeightTensors(XUSG::CommandList* pCommandList, XUSG::ML::WeightMapType& weights,
		const char* convLayerName, const char* scaleLayerName, const char* shiftLayerName,
		const uint32_t filterSizes[4], std::vector<XUSG::Resource::uptr>& uploaders,
		XUSG::Buffer::uptr& filterWeightBuffer, XUSG::Buffer::uptr& biasWeightBuffer);
	bool createWeightResource(const XUSG::Device* pDevice, const uint32_t tensorSizes[4], XUSG::Buffer::uptr& resourceOut);
	bool createPipelineLayouts();
	bool createPipelines();
	bool createDescriptorTables();
	bool initResources(XUSG::CommandList* pCommandList, const XUSG::ML::CommandRecorder* pCommandRecorder);

	XUSG::ShaderLib::uptr				m_shaderLib;
	XUSG::PipelineLayoutLib::uptr		m_pipelineLayoutLib;
	XUSG::Graphics::PipelineLib::uptr	m_graphicsPipelineLib;
	XUSG::Compute::PipelineLib::uptr	m_computePipelineLib;
	XUSG::DescriptorTableLib::sptr		m_descriptorTableLib;

	// Model layer sizes and indices
	static const size_t	c_numUpsampleLayers = 2;
	static const size_t	c_numConvLayers = 7;
	static const size_t	c_numIntermediateBuffers = 2;

	// Resources
	XUSG::Texture::uptr		m_inputImage;
	XUSG::TypedBuffer::uptr	m_modelInput;
	XUSG::TypedBuffer::uptr	m_modelOutput;

	XUSG::Buffer::uptr	m_modelIntermediateResult[c_numIntermediateBuffers];
	XUSG::Buffer::uptr	m_modelConvFilterWeights[c_numConvLayers];
	XUSG::Buffer::uptr	m_modelConvBiasWeights[c_numConvLayers];

	XUSG::Buffer::uptr	m_modelUpsamplePersistentResources[c_numUpsampleLayers];
	XUSG::Buffer::uptr	m_modelConvPersistentResources[c_numConvLayers];
	XUSG::Buffer::uptr	m_modelAddPersistentResource;

	XUSG::Buffer::uptr	m_modelInitTemporaryResources[NUM_OP];
	XUSG::Buffer::uptr	m_modelUpsampleTemporaryResources[c_numUpsampleLayers];
	XUSG::Buffer::uptr	m_modelConvTemporaryResources[c_numConvLayers];
	XUSG::Buffer::uptr	m_modelAddTemporaryResource;

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
