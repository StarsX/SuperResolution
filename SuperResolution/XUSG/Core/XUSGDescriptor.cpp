//--------------------------------------------------------------------------------------
// By Stars XU Tianchen
//--------------------------------------------------------------------------------------

#include "XUSGDescriptor.h"
#include "XUSGSampler.inl"

using namespace std;
using namespace XUSG;

Util::DescriptorTable::DescriptorTable()
{
	m_key.resize(0);
}

Util::DescriptorTable::~DescriptorTable()
{
}

void Util::DescriptorTable::SetDescriptors(uint32_t start, uint32_t num, const Descriptor* srcDescriptors)
{
	const auto size = sizeof(Descriptor) * (start + num);
	if (size > m_key.size())
		m_key.resize(size);

	const auto descriptors = reinterpret_cast<Descriptor*>(&m_key[0]);
	memcpy(&descriptors[start], srcDescriptors, sizeof(Descriptor) * num);
}

void Util::DescriptorTable::SetSamplers(uint32_t start, uint32_t num,
	const SamplerPreset* presets, DescriptorTableCache& descriptorTableCache)
{
	const auto size = sizeof(Sampler*) * (start + num);
	if (size > m_key.size())
		m_key.resize(size);

	const auto descriptors = reinterpret_cast<const Sampler**>(&m_key[0]);

	for (auto i = 0u; i < num; ++i)
		descriptors[start + i] = descriptorTableCache.GetSampler(presets[i]).get();
}

DescriptorTable Util::DescriptorTable::CreateCbvSrvUavTable(DescriptorTableCache& descriptorTableCache, uint8_t index)
{
	return descriptorTableCache.createCbvSrvUavTable(m_key, index);
}

DescriptorTable Util::DescriptorTable::GetCbvSrvUavTable(DescriptorTableCache& descriptorTableCache, uint8_t index)
{
	return descriptorTableCache.getCbvSrvUavTable(m_key, index);
}

DescriptorTable Util::DescriptorTable::CreateSamplerTable(DescriptorTableCache& descriptorTableCache, uint8_t index)
{
	return descriptorTableCache.createSamplerTable(m_key, index);
}

DescriptorTable Util::DescriptorTable::GetSamplerTable(DescriptorTableCache& descriptorTableCache, uint8_t index)
{
	return descriptorTableCache.getSamplerTable(m_key, index);
}

RenderTargetTable Util::DescriptorTable::CreateRtvTable(DescriptorTableCache& descriptorTableCache, uint8_t index)
{
	return descriptorTableCache.createRtvTable(m_key, index);
}

RenderTargetTable Util::DescriptorTable::GetRtvTable(DescriptorTableCache& descriptorTableCache, uint8_t index)
{
	return descriptorTableCache.getRtvTable(m_key, index);
}

const string& Util::DescriptorTable::GetKey() const
{
	return m_key;
}

//--------------------------------------------------------------------------------------

DescriptorTableCache::DescriptorTableCache() :
	m_device(nullptr),
	m_cbvSrvUavTables(0),
	m_samplerTables(0),
	m_rtvTables(0),
	m_descriptorKeyPtrs(),
	m_descriptorPools(),
	m_descriptorStrides(),
	m_descriptorCounts(),
	m_samplerPresets()
{
	// Sampler presets
	m_pfnSamplers[SamplerPreset::POINT_WRAP] = SamplerPointWrap;
	m_pfnSamplers[SamplerPreset::POINT_CLAMP] = SamplerPointClamp;
	m_pfnSamplers[SamplerPreset::POINT_BORDER] = SamplerPointBorder;
	m_pfnSamplers[SamplerPreset::POINT_LESS_EQUAL] = SamplerPointLessEqual;

	m_pfnSamplers[SamplerPreset::LINEAR_WRAP] = SamplerLinearWrap;
	m_pfnSamplers[SamplerPreset::LINEAR_CLAMP] = SamplerLinearClamp;
	m_pfnSamplers[SamplerPreset::LINEAR_BORDER] = SamplerLinearBorder;
	m_pfnSamplers[SamplerPreset::LINEAR_LESS_EQUAL] = SamplerLinearLessEqual;

	m_pfnSamplers[SamplerPreset::ANISOTROPIC_WRAP] = SamplerAnisotropicWrap;
	m_pfnSamplers[SamplerPreset::ANISOTROPIC_CLAMP] = SamplerAnisotropicClamp;
	m_pfnSamplers[SamplerPreset::ANISOTROPIC_BORDER] = SamplerAnisotropicBorder;
	m_pfnSamplers[SamplerPreset::ANISOTROPIC_LESS_EQUAL] = SamplerAnisotropicLessEqual;
}

DescriptorTableCache::DescriptorTableCache(const Device& device, const wchar_t* name) :
	DescriptorTableCache()
{
	SetDevice(device);
	SetName(name);
}

DescriptorTableCache::~DescriptorTableCache()
{
}

void DescriptorTableCache::SetDevice(const Device& device)
{
	m_device = device;

	m_descriptorStrides[CBV_SRV_UAV_POOL] = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	m_descriptorStrides[SAMPLER_POOL] = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
	m_descriptorStrides[RTV_POOL] = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
}

void DescriptorTableCache::SetName(const wchar_t* name)
{
	if (name) m_name = name;
}

bool DescriptorTableCache::AllocateDescriptorPool(DescriptorPoolType type, uint32_t numDescriptors, uint8_t index)
{
	checkDescriptorPoolTypeStorage(type, index);

	return allocateDescriptorPool(type, numDescriptors, index);
}

DescriptorTable DescriptorTableCache::CreateCbvSrvUavTable(const Util::DescriptorTable& util, uint8_t index)
{
	return createCbvSrvUavTable(util.GetKey(), index);
}

DescriptorTable DescriptorTableCache::GetCbvSrvUavTable(const Util::DescriptorTable& util, uint8_t index)
{
	return getCbvSrvUavTable(util.GetKey(), index);
}

DescriptorTable DescriptorTableCache::CreateSamplerTable(const Util::DescriptorTable& util, uint8_t index)
{
	return createSamplerTable(util.GetKey(), index);
}

DescriptorTable DescriptorTableCache::GetSamplerTable(const Util::DescriptorTable& util, uint8_t index)
{
	return getSamplerTable(util.GetKey(), index);
}

RenderTargetTable DescriptorTableCache::CreateRtvTable(const Util::DescriptorTable& util, uint8_t index)
{
	return createRtvTable(util.GetKey(), index);
}

RenderTargetTable DescriptorTableCache::GetRtvTable(const Util::DescriptorTable& util, uint8_t index)
{
	return getRtvTable(util.GetKey(), index);
}

const DescriptorPool& DescriptorTableCache::GetDescriptorPool(DescriptorPoolType type, uint8_t index) const
{
	return m_descriptorPools[type][index];
}

const shared_ptr<Sampler>& DescriptorTableCache::GetSampler(SamplerPreset preset)
{
	if (m_samplerPresets[preset] == nullptr)
		m_samplerPresets[preset] = make_shared<Sampler>(m_pfnSamplers[preset]());

	return m_samplerPresets[preset];
}

uint32_t DescriptorTableCache::GetDescriptorStride(DescriptorPoolType type) const
{
	return m_descriptorStrides[type];
}

void DescriptorTableCache::checkDescriptorPoolTypeStorage(DescriptorPoolType type, uint8_t index)
{
	if (index >= m_descriptorPools[type].size())
		m_descriptorPools[type].resize(index + 1);

	if (index >= m_descriptorCounts[type].size())
		m_descriptorCounts[type].resize(index + 1);

	if (index >= m_descriptorKeyPtrs[type].size())
		m_descriptorKeyPtrs[type].resize(index + 1);
}

bool DescriptorTableCache::allocateDescriptorPool(DescriptorPoolType type, uint32_t numDescriptors, uint8_t index)
{
	static const D3D12_DESCRIPTOR_HEAP_TYPE heapTypes[NUM_DESCRIPTOR_POOL] =
	{
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
		D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
		D3D12_DESCRIPTOR_HEAP_TYPE_RTV
	};

	static const wchar_t* poolNames[] =
	{
		L".CbvSrvUavPool",
		L".SamplerPool",
		L".RtvPool"
	};

	D3D12_DESCRIPTOR_HEAP_DESC desc = {};
	desc.NumDescriptors = numDescriptors;
	desc.Type = heapTypes[type];
	if (type != D3D12_DESCRIPTOR_HEAP_TYPE_RTV) desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	V_RETURN(m_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_descriptorPools[type][index])), cerr, false);
	if (!m_name.empty()) m_descriptorPools[type][index]->SetName((m_name + poolNames[type]).c_str());

	m_descriptorCounts[type][index] = 0;

	return true;
}

bool DescriptorTableCache::reallocateCbvSrvUavPool(const string& key, uint8_t index)
{
	assert(key.size() > 0);
	const auto numDescriptors = static_cast<uint32_t>(key.size() / sizeof(Descriptor));

	checkDescriptorPoolTypeStorage(CBV_SRV_UAV_POOL, index);

	// Allocate a new pool if neccessary
	const auto& descriptorPool = m_descriptorPools[CBV_SRV_UAV_POOL][index];
	const auto descriptorCount = m_descriptorCounts[CBV_SRV_UAV_POOL][index] + numDescriptors;
	if (!descriptorPool || descriptorPool->GetDesc().NumDescriptors < descriptorCount)
	{
		N_RETURN(allocateDescriptorPool(CBV_SRV_UAV_POOL, descriptorCount, index), false);

		// Recreate descriptor tables
		for (const auto& pKey : m_descriptorKeyPtrs[CBV_SRV_UAV_POOL][index])
		{
			const auto table = createCbvSrvUavTable(*pKey, index);
			*m_cbvSrvUavTables[*pKey] = *table;
		}
	}

	return true;
}

bool DescriptorTableCache::reallocateSamplerPool(const string& key, uint8_t index)
{
	assert(key.size() > 0);
	const auto numDescriptors = static_cast<uint32_t>(key.size() / sizeof(Sampler*));

	checkDescriptorPoolTypeStorage(SAMPLER_POOL, index);

	// Allocate a new pool if neccessary
	const auto& descriptorPool = m_descriptorPools[SAMPLER_POOL][index];
	const auto descriptorCount = m_descriptorCounts[SAMPLER_POOL][index] + numDescriptors;
	if (!descriptorPool || descriptorPool->GetDesc().NumDescriptors < descriptorCount)
	{
		N_RETURN(allocateDescriptorPool(SAMPLER_POOL, descriptorCount, index), false);

		// Recreate descriptor tables
		for (const auto& pKey : m_descriptorKeyPtrs[SAMPLER_POOL][index])
		{
			const auto table = createSamplerTable(*pKey, index);
			*m_samplerTables[*pKey] = *table;
		}
	}

	return true;
}

bool DescriptorTableCache::reallocateRtvPool(const string& key, uint8_t index)
{
	assert(key.size() > 0);
	const auto numDescriptors = static_cast<uint32_t>(key.size() / sizeof(Descriptor));

	checkDescriptorPoolTypeStorage(RTV_POOL, index);

	// Allocate a new pool if neccessary
	const auto& descriptorPool = m_descriptorPools[RTV_POOL][index];
	const auto descriptorCount = m_descriptorCounts[RTV_POOL][index] + numDescriptors;
	if (!descriptorPool || descriptorPool->GetDesc().NumDescriptors < descriptorCount)
	{
		N_RETURN(allocateDescriptorPool(RTV_POOL, descriptorCount, index), false);

		// Recreate descriptor tables
		for (const auto& pKey : m_descriptorKeyPtrs[RTV_POOL][index])
		{
			const auto table = createRtvTable(*pKey, index);
			*m_rtvTables[*pKey] = *table;
		}
	}

	return true;
}

DescriptorTable DescriptorTableCache::createCbvSrvUavTable(const string& key, uint8_t index)
{
	if (key.size() > 0)
	{
		const auto numDescriptors = static_cast<uint32_t>(key.size() / sizeof(Descriptor));
		const auto descriptors = reinterpret_cast<const Descriptor*>(&key[0]);

		// Compute start addresses for CPU and GPU handles
		const auto& descriptorPool = m_descriptorPools[CBV_SRV_UAV_POOL][index];
		const auto& descriptorStride = m_descriptorStrides[CBV_SRV_UAV_POOL];
		auto& descriptorCount = m_descriptorCounts[CBV_SRV_UAV_POOL][index];
		Descriptor descriptor(descriptorPool->GetCPUDescriptorHandleForHeapStart(), descriptorCount, descriptorStride);
		DescriptorTable table = make_shared<DescriptorView>(descriptorPool->GetGPUDescriptorHandleForHeapStart(),
			descriptorCount, descriptorStride);

		// Create a descriptor table
		for (auto i = 0u; i < numDescriptors; ++i)
		{
			// Copy a descriptor
			m_device->CopyDescriptorsSimple(1, descriptor, descriptors[i], D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			descriptor.Offset(descriptorStride);
			++descriptorCount;
		}

		return table;
	}

	return nullptr;
}

DescriptorTable DescriptorTableCache::getCbvSrvUavTable(const string& key, uint8_t index)
{
	if (key.size() > 0)
	{
		const auto tableIter = m_cbvSrvUavTables.find(key);

		// Create one, if it does not exist
		if (tableIter == m_cbvSrvUavTables.end() && reallocateCbvSrvUavPool(key, index))
		{
			const auto table = createCbvSrvUavTable(key, index);
			m_cbvSrvUavTables[key] = table;
			m_descriptorKeyPtrs[CBV_SRV_UAV_POOL][index].push_back(&m_cbvSrvUavTables.find(key)->first);

			return table;
		}

		return tableIter->second;
	}

	return nullptr;
}

DescriptorTable DescriptorTableCache::createSamplerTable(const string& key, uint8_t index)
{
	if (key.size() > 0)
	{
		const auto numDescriptors = static_cast<uint32_t>(key.size() / sizeof(Sampler*));
		const auto descriptors = reinterpret_cast<const Sampler* const*>(&key[0]);

		// Compute start addresses for CPU and GPU handles
		const auto& descriptorPool = m_descriptorPools[SAMPLER_POOL][index];
		const auto& descriptorStride = m_descriptorStrides[SAMPLER_POOL];
		auto& descriptorCount = m_descriptorCounts[SAMPLER_POOL][index];
		Descriptor descriptor(descriptorPool->GetCPUDescriptorHandleForHeapStart(), descriptorCount, descriptorStride);
		DescriptorTable table = make_shared<DescriptorView>(descriptorPool->GetGPUDescriptorHandleForHeapStart(),
			descriptorCount, descriptorStride);

		// Create a descriptor table
		for (auto i = 0u; i < numDescriptors; ++i)
		{
			// Copy a descriptor
			m_device->CreateSampler(descriptors[i], descriptor);
			descriptor.Offset(descriptorStride);
			++descriptorCount;
		}

		return table;
	}

	return nullptr;
}

DescriptorTable DescriptorTableCache::getSamplerTable(const string& key, uint8_t index)
{
	if (key.size() > 0)
	{
		const auto tableIter = m_samplerTables.find(key);

		// Create one, if it does not exist
		if (tableIter == m_samplerTables.end() && reallocateSamplerPool(key, index))
		{
			const auto table = createSamplerTable(key, index);
			m_samplerTables[key] = table;
			m_descriptorKeyPtrs[SAMPLER_POOL][index].push_back(&m_samplerTables.find(key)->first);

			return table;
		}

		return tableIter->second;
	}

	return nullptr;
}

RenderTargetTable DescriptorTableCache::createRtvTable(const string& key, uint8_t index)
{
	if (key.size() > 0)
	{
		const auto numDescriptors = static_cast<uint32_t>(key.size() / sizeof(Descriptor));
		const auto descriptors = reinterpret_cast<const Descriptor*>(&key[0]);

		// Compute start addresses for CPU and GPU handles
		const auto& descriptorPool = m_descriptorPools[RTV_POOL][index];
		const auto& descriptorStride = m_descriptorStrides[RTV_POOL];
		auto& descriptorCount = m_descriptorCounts[RTV_POOL][index];
		Descriptor descriptor(descriptorPool->GetCPUDescriptorHandleForHeapStart(), descriptorCount, descriptorStride);
		RenderTargetTable table = make_shared<Descriptor>();
		*table = descriptor;

		// Create a descriptor table
		for (auto i = 0u; i < numDescriptors; ++i)
		{
			// Copy a descriptor
			m_device->CopyDescriptorsSimple(1, descriptor, descriptors[i], D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
			descriptor.Offset(descriptorStride);
			++descriptorCount;
		}

		return table;
	}

	return nullptr;
}

RenderTargetTable DescriptorTableCache::getRtvTable(const string& key, uint8_t index)
{
	if (key.size() > 0)
	{
		const auto tableIter = m_rtvTables.find(key);

		// Create one, if it does not exist
		if (tableIter == m_rtvTables.end() && reallocateRtvPool(key, index))
		{
			const auto table = createRtvTable(key, index);
			m_rtvTables[key] = table;
			m_descriptorKeyPtrs[RTV_POOL][index].push_back(&m_rtvTables.find(key)->first);

			return table;
		}

		return tableIter->second;
	}

	return nullptr;
}
