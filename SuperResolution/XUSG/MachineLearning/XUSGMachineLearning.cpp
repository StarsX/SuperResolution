//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DXFrameworkHelper.h"
#include "XUSGMachineLearning.h"

using namespace std;
using namespace XUSG;
using namespace XUSG::ML;

//--------------------------------------------------------------------------------------
// Tensor
//--------------------------------------------------------------------------------------

Tensor::Tensor() :
	m_bufferTensorDesc(),
	m_tensorDesc()
{
}

Tensor::~Tensor()
{
}

uint64_t Tensor::Create(TensorDataType dataType, uint32_t dimensionCount, const uint32_t* pSizes,
	const uint32_t* pStrides, TensorFlag flags)
{
	m_bufferTensorDesc.DataType = static_cast<decltype(m_bufferTensorDesc.DataType)>(dataType);
	m_bufferTensorDesc.Flags = static_cast<DML_TENSOR_FLAGS>(flags);
	m_bufferTensorDesc.DimensionCount = dimensionCount;
	m_bufferTensorDesc.Sizes = pSizes;
	m_bufferTensorDesc.Strides = pStrides;
	m_bufferTensorDesc.TotalTensorSizeInBytes = calcBufferTensorSize(dataType, dimensionCount, pSizes, pStrides);

	// Create DirectML operator(s). Operators represent abstract functions such as "multiply", "reduce", "convolution", or even
	// compound operations such as recurrent neural nets. This example creates an instance of the Identity operator,
	// which applies the function f(x) = x for all elements in a tensor.
	m_tensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
	m_tensorDesc.Desc = &m_bufferTensorDesc;

	return m_bufferTensorDesc.TotalTensorSizeInBytes;
}

const TensorDesc& Tensor::GetTensorDesc() const
{
	return m_tensorDesc;
}

uint64_t Tensor::GetTensorBufferSize() const
{
	return m_bufferTensorDesc.TotalTensorSizeInBytes;
}

uint64_t Tensor::calcBufferTensorSize(TensorDataType dataType, uint32_t dimensionCount,
	const uint32_t* pSizes, const uint32_t* pStrides)
{
	auto elementSizeInBytes = 0u;
	switch (dataType)
	{
	case TensorDataType::FLOAT32:
	case TensorDataType::UINT32:
	case TensorDataType::INT32:
		elementSizeInBytes = 4;
		break;

	case TensorDataType::FLOAT16:
	case TensorDataType::UINT16:
	case TensorDataType::INT16:
		elementSizeInBytes = 2;
		break;

	case TensorDataType::UINT8:
	case TensorDataType::INT8:
		elementSizeInBytes = 1;
		break;

	default:
		return 0; // Invalid data type
	}

	uint64_t minimumImpliedSizeInBytes = 0;
	if (!pStrides)
	{
		minimumImpliedSizeInBytes = *pSizes;
		for (auto i = 1u; i < dimensionCount; ++i)
		{
			minimumImpliedSizeInBytes *= pSizes[i];
		}
		minimumImpliedSizeInBytes *= elementSizeInBytes;
	}
	else
	{
		auto indexOfLastElement = 0u;
		for (auto i = 0u; i < dimensionCount; ++i)
		{
			indexOfLastElement += (pSizes[i] - 1) * pStrides[i];
		}

		minimumImpliedSizeInBytes = (indexOfLastElement + 1) * elementSizeInBytes;
	}

	// Round up to the nearest 4 bytes.
	minimumImpliedSizeInBytes = (minimumImpliedSizeInBytes + 3) & ~3ui64;

	return minimumImpliedSizeInBytes;
}

//--------------------------------------------------------------------------------------
// Operator
//--------------------------------------------------------------------------------------

Operator::Operator() :
	m_dispatchable(nullptr)
{
}

Operator::~Operator()
{
}

bool Operator::Create(const ML::Device& device, const OperatorDesc& desc, ExecutionFlag flags)
{
	com_ptr<IDMLOperator> dmlOperator;
	V_RETURN(device->CreateOperator(&desc, IID_PPV_ARGS(&dmlOperator)), cerr, false);

	// Compile the operator into an object that can be dispatched to the GPU. In this step, DirectML performs operator
	// fusion and just-in-time (JIT) compilation of shader bytecode, then compiles it into a Direct3D 12 pipeline state object (PSO).
	// The resulting compiled operator is a baked, optimized form of an operator suitable for execution on the GPU.
	V_RETURN(device->CompileOperator(dmlOperator.get(), static_cast<DML_EXECUTION_FLAGS>(flags),
		IID_PPV_ARGS(&m_dispatchable)), cerr, false);

	return true;
}

const Dispatchable& Operator::GetDispatchable() const
{
	return m_dispatchable;
}

uint32_t Operator::GetDescriptorCount() const
{
	return m_dispatchable->GetBindingProperties().RequiredDescriptorCount;
}

uint64_t Operator::GetTemporaryResourceSize() const
{
	return m_dispatchable->GetBindingProperties().TemporaryResourceSize;
}

uint64_t Operator::GetPersistentResourceSize() const
{
	return m_dispatchable->GetBindingProperties().PersistentResourceSize;
}

//--------------------------------------------------------------------------------------
// Operator initializer
//--------------------------------------------------------------------------------------

OperatorInitializer::OperatorInitializer() :
	Operator()
{
}

OperatorInitializer::~OperatorInitializer()
{
}

bool OperatorInitializer::Create(const ML::Device& device, const Operator* pOperators, uint32_t numOperators)
{
	vector<com_ptr<IDMLCompiledOperator>> compiledOperators(numOperators);
	vector<IDMLCompiledOperator*> dmlCompiledOperators(numOperators);
	for (auto i = 0u; i < numOperators; ++i)
	{
		pOperators[i].GetDispatchable()->QueryInterface(IID_PPV_ARGS(&compiledOperators[i]));
		dmlCompiledOperators[i] = compiledOperators[i].get();
	}

	com_ptr<IDMLOperatorInitializer> dmlOperatorInitializer;
	V_RETURN(device->CreateOperatorInitializer(numOperators, dmlCompiledOperators.data(),
		IID_PPV_ARGS(&dmlOperatorInitializer)), cerr, false);
	m_dispatchable = dmlOperatorInitializer;

	// Query the operator for the required size (in descriptors) of its binding table.
	// You need to initialize an operator exactly once before it can be executed, and
	// the two stages require different numbers of descriptors for binding. For simplicity,
	// we create a single descriptor heap that's large enough to satisfy them both.
	const auto initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
	m_descriptorCount = initializeBindingProperties.RequiredDescriptorCount;
	m_temporaryResourceSize = initializeBindingProperties.TemporaryResourceSize;
	for (auto i = 0u; i < numOperators; ++i)
	{
		m_descriptorCount = (max)(pOperators[i].GetDescriptorCount(), m_descriptorCount);
		m_temporaryResourceSize = (max)(pOperators[i].GetTemporaryResourceSize(), m_temporaryResourceSize);
	}

	return true;
}

uint32_t OperatorInitializer::GetDescriptorCount() const
{
	return m_descriptorCount;
}

uint64_t OperatorInitializer::GetTemporaryResourceSize() const
{
	return m_temporaryResourceSize;
}

//--------------------------------------------------------------------------------------
// Binding
//--------------------------------------------------------------------------------------

Binding::Binding() :
	m_bindingTable(nullptr),
	m_descriptorStride(128),
	m_isDispatchable(false)
{
}

Binding::~Binding()
{
}

bool Binding::Create(const ML::Device& device, const Operator& dispatchable, const DescriptorPool& descriptorPool,
	uint32_t descriptorCount, int32_t descriptorOffset)
{
	com_ptr<ID3D12Device> parent;
	device->GetParentDevice(IID_PPV_ARGS(&parent));
	m_descriptorStride = parent->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	Descriptor descriptor(descriptorPool->GetCPUDescriptorHandleForHeapStart());
	DescriptorTable::element_type descriptorHandle(descriptorPool->GetGPUDescriptorHandleForHeapStart());
	descriptor.Offset(descriptorOffset, m_descriptorStride);
	descriptorHandle.Offset(descriptorOffset, m_descriptorStride);

	DML_BINDING_TABLE_DESC dmlBindingTableDesc = {};
	dmlBindingTableDesc.Dispatchable = dispatchable.GetDispatchable().get();
	dmlBindingTableDesc.CPUDescriptorHandle = descriptor;
	dmlBindingTableDesc.GPUDescriptorHandle = descriptorHandle;
	dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

	V_RETURN(device->CreateBindingTable(&dmlBindingTableDesc, IID_PPV_ARGS(&m_bindingTable)), cerr, false);
	m_isDispatchable = false;

	return true;
}

bool Binding::Reset(const Operator& dispatchable, const DescriptorPool& descriptorPool,
	uint32_t descriptorCount, int32_t descriptorOffset)
{
	Descriptor descriptor(descriptorPool->GetCPUDescriptorHandleForHeapStart());
	DescriptorTable::element_type descriptorHandle(descriptorPool->GetGPUDescriptorHandleForHeapStart());
	descriptor.Offset(descriptorOffset, m_descriptorStride);
	descriptorHandle.Offset(descriptorOffset, m_descriptorStride);

	DML_BINDING_TABLE_DESC dmlBindingTableDesc = {};
	dmlBindingTableDesc.Dispatchable = dispatchable.GetDispatchable().get();
	dmlBindingTableDesc.CPUDescriptorHandle = descriptor;
	dmlBindingTableDesc.GPUDescriptorHandle = descriptorHandle;
	dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

	V_RETURN(m_bindingTable->Reset(&dmlBindingTableDesc), cerr, false);
	m_isDispatchable = false;

	return true;
}

void Binding::BindInput(uint32_t i, const ResourceBase& buffer, uint64_t size, uint64_t offset)
{
	BindInput(i, buffer.GetResource(), size, offset);
}

void Binding::BindInput(uint32_t i, const Resource& resource, uint64_t size, uint64_t offset)
{
	BindInputBuffer(i, resource, size, offset);
	BindInput(i, resource ? static_cast<size_t>(i) : -1);
}

void Binding::BindInput(uint32_t i, size_t bindingIndex, uint32_t bindingCount)
{
	if (i >= m_inputBindings.size())
		m_inputBindings.resize(i + 1);

	assert(bindingIndex == -1 || bindingIndex < m_inputBufferBindings.size());

	if (bindingCount < 1 || bindingIndex == -1)
	{
		m_inputBindings[i].Type = DML_BINDING_TYPE_NONE;
		m_inputBindings[i].Desc = nullptr;
	}
	else if (bindingCount > 1)
	{
		m_inputArrayBindings.push_back(ArrayBinding());
		m_inputArrayBindings.back().BindingCount = bindingCount;
		m_inputArrayBindings.back().Bindings = (const BufferBinding*)bindingIndex;

		m_inputBindings[i].Type = DML_BINDING_TYPE_BUFFER_ARRAY;
		m_inputBindings[i].Desc = (const void*)(m_inputArrayBindings.size() - 1);
	}
	else
	{
		m_inputBindings[i].Type = DML_BINDING_TYPE_BUFFER;
		m_inputBindings[i].Desc = (const void*)bindingIndex;
	}
}

void Binding::BindOutput(uint32_t i, const ResourceBase& buffer, uint64_t size, uint64_t offset)
{
	BindOutput(i, buffer.GetResource(), size, offset);
}

void Binding::BindOutput(uint32_t i, const Resource& resource, uint64_t size, uint64_t offset)
{
	BindOutputBuffer(i, resource, size, offset);
	BindOutput(i, resource ? static_cast<size_t>(i) : -1);
}

void Binding::BindOutput(uint32_t i, size_t bindingIndex, uint32_t bindingCount)
{
	if (i >= m_outputBindings.size())
		m_outputBindings.resize(i + 1);

	assert(bindingIndex == -1 || bindingIndex < m_outputBufferBindings.size());

	const auto isEmpty = bindingCount < 1 || bindingIndex == -1;
	if (isEmpty)
	{
		m_outputBindings[i].Type = DML_BINDING_TYPE_NONE;
		m_outputBindings[i].Desc = nullptr;
	}
	else if (bindingCount > 1)
	{
		m_outputArrayBindings.push_back(ArrayBinding());
		m_outputArrayBindings.back().BindingCount = bindingCount;
		m_outputArrayBindings.back().Bindings = (const BufferBinding*)bindingIndex;

		m_outputBindings[i].Type = DML_BINDING_TYPE_BUFFER_ARRAY;
		m_outputBindings[i].Desc = (const void*)(m_outputArrayBindings.size() - 1);
	}
	else
	{
		m_outputBindings[i].Type = DML_BINDING_TYPE_BUFFER;
		m_outputBindings[i].Desc = (const void*)bindingIndex;
	}
}

void Binding::AppendInput(const ResourceBase& buffer, uint64_t size, uint64_t offset)
{
	AppendInput(buffer.GetResource(), size,  offset);
}

void Binding::AppendInput(const Resource& resource, uint64_t size, uint64_t offset)
{
	BindInput(static_cast<uint32_t>(m_inputBindings.size()), resource, size, offset);
}

void Binding::AppendInput(size_t bindingIndex, uint32_t bindingCount)
{
	BindInput(static_cast<uint32_t>(m_inputBindings.size()), bindingIndex, bindingCount);
}

void Binding::AppendOutput(const ResourceBase& buffer, uint64_t size, uint64_t offset)
{
	AppendOutput(buffer.GetResource(), size, offset);
}

void Binding::AppendOutput(const Resource& resource, uint64_t size, uint64_t offset)
{
	BindOutput(static_cast<uint32_t>(m_outputBindings.size()), resource, size, offset);
}

void Binding::AppendOutput(size_t bindingIndex, uint32_t bindingCount)
{
	BindOutput(static_cast<uint32_t>(m_outputBindings.size()), bindingIndex, bindingCount);
}

void Binding::BindInputBuffer(uint32_t i, const ResourceBase& buffer, uint64_t size, uint64_t offset)
{
	BindInputBuffer(i, buffer.GetResource(), size, offset);
}

void Binding::BindInputBuffer(uint32_t i, const Resource& resource, uint64_t size, uint64_t offset)
{
	if (i >= m_inputBufferBindings.size())
		m_inputBufferBindings.resize(i + 1);

	size = !resource || size > 0 ? size : resource->GetDesc().Width;
	m_inputBufferBindings[i].Buffer = resource.get();
	m_inputBufferBindings[i].Offset = offset;
	m_inputBufferBindings[i].SizeInBytes = size;
}

void Binding::BindOutputBuffer(uint32_t i, const ResourceBase& buffer, uint64_t size, uint64_t offset)
{
	BindOutputBuffer(i, buffer.GetResource(), size, offset);
}

void Binding::BindOutputBuffer(uint32_t i, const Resource& resource, uint64_t size, uint64_t offset)
{
	if (i >= m_outputBufferBindings.size())
		m_outputBufferBindings.resize(i + 1);

	size = !resource || size > 0 ? size : resource->GetDesc().Width;
	m_outputBufferBindings[i].Buffer = resource.get();
	m_outputBufferBindings[i].Offset = offset;
	m_outputBufferBindings[i].SizeInBytes = size;
}

void Binding::BindTemporary(const ResourceBase& buffer, uint64_t size, uint64_t offset)
{
	BindTemporary(buffer.GetResource(), size, offset);
}

void Binding::BindTemporary(const Resource& resource, uint64_t size, uint64_t offset)
{
	size = !resource || size > 0 ? size : resource->GetDesc().Width;
	const DML_BUFFER_BINDING bufferBinding = { resource.get(), offset, size };
	const DML_BINDING_DESC bindingDesc =
	{
		resource ? DML_BINDING_TYPE_BUFFER : DML_BINDING_TYPE_NONE,
		resource ? &bufferBinding : nullptr
	};

	m_bindingTable->BindTemporaryResource(&bindingDesc);
}

void Binding::BindPersistent(const ResourceBase& buffer, uint64_t size, uint64_t offset)
{
	BindPersistent(buffer.GetResource(), size, offset);
}

void Binding::BindPersistent(const Resource& resource, uint64_t size, uint64_t offset)
{
	size = !resource || size > 0 ? size : resource->GetDesc().Width;
	const DML_BUFFER_BINDING bufferBinding = { resource.get(), offset, size };
	const DML_BINDING_DESC bindingDesc =
	{
		resource ? DML_BINDING_TYPE_BUFFER : DML_BINDING_TYPE_NONE,
		resource ? &bufferBinding : nullptr
	};

	m_bindingTable->BindPersistentResource(&bindingDesc);
}

const BindingTable& Binding::GetBindingTable() const
{
	return m_bindingTable;
}

const BindingTable& Binding::GetDispatchableBindingTable()
{
	if (!m_isDispatchable)
	{
		for (auto& arrayBinding : m_inputArrayBindings)
		{
			const auto i = (size_t)arrayBinding.Bindings;
			arrayBinding.Bindings = &m_inputBufferBindings[i];
		}

		for (auto& arrayBinding : m_outputArrayBindings)
		{
			const auto i = (size_t)arrayBinding.Bindings;
			arrayBinding.Bindings = &m_outputBufferBindings[i];
		}

		for (auto& bindingDesc : m_inputBindings)
		{
			const auto i = (size_t)bindingDesc.Desc;
			if (bindingDesc.Type == DML_BINDING_TYPE_BUFFER_ARRAY)
				bindingDesc.Desc = &m_inputArrayBindings[i];
			else if (bindingDesc.Type == DML_BINDING_TYPE_BUFFER)
				bindingDesc.Desc = &m_inputBufferBindings[i];
		}

		for (auto& bindingDesc : m_outputBindings)
		{
			const auto i = (size_t)bindingDesc.Desc;
			if (bindingDesc.Type == DML_BINDING_TYPE_BUFFER_ARRAY)
				bindingDesc.Desc = &m_outputArrayBindings[i];
			else if (bindingDesc.Type == DML_BINDING_TYPE_BUFFER)
				bindingDesc.Desc = &m_outputBufferBindings[i];
		}

		m_bindingTable->BindInputs(static_cast<uint32_t>(m_inputBindings.size()), m_inputBindings.data());
		m_bindingTable->BindOutputs(static_cast<uint32_t>(m_outputBindings.size()), m_outputBindings.data());
		m_isDispatchable = true;
	}

	return m_bindingTable;
}
