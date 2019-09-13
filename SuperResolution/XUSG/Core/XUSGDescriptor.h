//--------------------------------------------------------------------------------------
// By Stars XU Tianchen
//--------------------------------------------------------------------------------------

#pragma once

#include "XUSGType.h"

namespace XUSG
{
	enum DescriptorPoolType : uint8_t
	{
		CBV_SRV_UAV_POOL,
		SAMPLER_POOL,
		RTV_POOL,

		NUM_DESCRIPTOR_POOL
	};

	enum SamplerPreset : uint8_t
	{
		POINT_WRAP,
		POINT_CLAMP,
		POINT_BORDER,
		POINT_LESS_EQUAL,

		LINEAR_WRAP,
		LINEAR_CLAMP,
		LINEAR_BORDER,
		LINEAR_LESS_EQUAL,

		ANISOTROPIC_WRAP,
		ANISOTROPIC_CLAMP,
		ANISOTROPIC_BORDER,
		ANISOTROPIC_LESS_EQUAL,

		NUM_SAMPLER_PRESET
	};

	class DescriptorTableCache;

	namespace Util
	{
		class DescriptorTable
		{
		public:
			DescriptorTable();
			virtual ~DescriptorTable();

			void SetDescriptors(uint32_t start, uint32_t num, const Descriptor* srcDescriptors);
			void SetSamplers(uint32_t start, uint32_t num, const SamplerPreset* presets,
				DescriptorTableCache& descriptorTableCache);

			XUSG::DescriptorTable CreateCbvSrvUavTable(DescriptorTableCache& descriptorTableCache, uint8_t index = 0);
			XUSG::DescriptorTable GetCbvSrvUavTable(DescriptorTableCache& descriptorTableCache, uint8_t index = 0);

			XUSG::DescriptorTable CreateSamplerTable(DescriptorTableCache& descriptorTableCache, uint8_t index = 0);
			XUSG::DescriptorTable GetSamplerTable(DescriptorTableCache& descriptorTableCache, uint8_t index = 0);

			RenderTargetTable CreateRtvTable(DescriptorTableCache& descriptorTableCache, uint8_t index = 0);
			RenderTargetTable GetRtvTable(DescriptorTableCache& descriptorTableCache, uint8_t index = 0);

			const std::string& GetKey() const;

		protected:
			std::string m_key;
		};
	}

	class DescriptorTableCache
	{
	public:
		DescriptorTableCache();
		DescriptorTableCache(const Device& device, const wchar_t* name = nullptr);
		virtual ~DescriptorTableCache();

		void SetDevice(const Device& device);
		void SetName(const wchar_t* name);

		bool AllocateDescriptorPool(DescriptorPoolType type, uint32_t numDescriptors, uint8_t index = 0);

		DescriptorTable CreateCbvSrvUavTable(const Util::DescriptorTable& util, uint8_t index = 0);
		DescriptorTable GetCbvSrvUavTable(const Util::DescriptorTable& util, uint8_t index = 0);

		DescriptorTable CreateSamplerTable(const Util::DescriptorTable& util, uint8_t index = 0);
		DescriptorTable GetSamplerTable(const Util::DescriptorTable& util, uint8_t index = 0);

		RenderTargetTable CreateRtvTable(const Util::DescriptorTable& util, uint8_t index = 0);
		RenderTargetTable GetRtvTable(const Util::DescriptorTable& util, uint8_t index = 0);

		const DescriptorPool& GetDescriptorPool(DescriptorPoolType type, uint8_t index = 0) const;

		const std::shared_ptr<Sampler>& GetSampler(SamplerPreset preset);

		uint32_t GetDescriptorStride(DescriptorPoolType type) const;

	protected:
		friend class Util::DescriptorTable;

		void checkDescriptorPoolTypeStorage(DescriptorPoolType type, uint8_t index);

		bool allocateDescriptorPool(DescriptorPoolType type, uint32_t numDescriptors, uint8_t index);
		bool reallocateCbvSrvUavPool(const std::string& key, uint8_t index);
		bool reallocateSamplerPool(const std::string& key, uint8_t index);
		bool reallocateRtvPool(const std::string& key, uint8_t index);

		DescriptorTable createCbvSrvUavTable(const std::string& key, uint8_t index);
		DescriptorTable getCbvSrvUavTable(const std::string& key, uint8_t index);

		DescriptorTable createSamplerTable(const std::string& key, uint8_t index);
		DescriptorTable getSamplerTable(const std::string& key, uint8_t index);

		RenderTargetTable createRtvTable(const std::string& key, uint8_t index);
		RenderTargetTable getRtvTable(const std::string& key, uint8_t index);

		Device m_device;

		std::unordered_map<std::string, DescriptorTable> m_cbvSrvUavTables;
		std::unordered_map<std::string, DescriptorTable> m_samplerTables;
		std::unordered_map<std::string, RenderTargetTable> m_rtvTables;

		std::vector<std::vector<const std::string*>> m_descriptorKeyPtrs[NUM_DESCRIPTOR_POOL];

		std::vector<DescriptorPool>	m_descriptorPools[NUM_DESCRIPTOR_POOL];
		std::vector<uint32_t>		m_descriptorCounts[NUM_DESCRIPTOR_POOL];
		uint32_t m_descriptorStrides[NUM_DESCRIPTOR_POOL];

		std::shared_ptr<Sampler> m_samplerPresets[NUM_SAMPLER_PRESET];
		std::function<Sampler()> m_pfnSamplers[NUM_SAMPLER_PRESET];

		std::wstring	m_name;
	};
}
