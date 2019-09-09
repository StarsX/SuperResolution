//--------------------------------------------------------------------------------------
// By Stars XU Tianchen
//--------------------------------------------------------------------------------------

#pragma once

#include "MachineLearning/XUSGMachineLearning.h"
#include <map>

namespace XUSG
{
	namespace ML
	{
		enum class TensorLayout
		{
			DEFAULT,
			NHWC
		};

		using WeightsType = std::vector<float>;
		using WeightMapType = std::map<std::string, WeightsType>;

		class Util
		{
		public:
			Util(const Device& device, TensorDataType tensorDataType = TensorDataType::FLOAT32,
				TensorLayout tensorLayout = TensorLayout::DEFAULT);
			virtual ~Util();

			bool CreateUpsampleLayer(const uint32_t inputSizes[4], uint64_t& inputBufferRequiredSize,
				uint64_t& outputBufferRequiredSize, uint32_t outputSizes[4], Operator& opOut,
				uint32_t scaleSizeX = 2, uint32_t scaleSizeY = 2,
				InterpolationType interpolationType = InterpolationType::NEAREST_NEIGHBOR);
			bool CreateConvolutionLayer(const uint32_t inputSizes[4], const uint32_t* filterSizes,
				bool useBiasAndActivation, uint64_t& inputBufferRequiredSize, uint64_t& outputBufferRequiredSize,
				uint32_t outputSizes[4], Operator& opOut);
			bool CreateAdditionLayer(const uint32_t inputSizes[4], Operator& opOut);
			
			void CreateWeightTensors(WeightMapType& weights, const char* convLayerName, const char* scaleLayerName,
				const char* shiftLayerName, const uint32_t filterSizes[4], std::vector<uint8_t>& filterWeightsOut,
				std::vector<uint8_t>& biasWeightsOut);

			static void GetStrides(const uint32_t sizes[4], TensorLayout layout, uint32_t stridesOut[4]);

		protected:
			Device			m_device;

			TensorDataType	m_tensorDataType;
			TensorLayout	m_tensorLayout;
		};
	}
}
