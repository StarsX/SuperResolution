//--------------------------------------------------------------------------------------
// By Stars XU Tianchen
//--------------------------------------------------------------------------------------

RWBuffer<half>		rwTensor;
Texture2D<float3>	txInput;

cbuffer cb
{
	uint g_height;
	uint g_width;
	bool g_nhwc;
};

[numthreads(8, 8, 1)]
//void main(uint2 blockID : SV_GroupID, uint2 threadID : SV_GroupThreadID)
void main(uint2 DTid : SV_DispatchThreadID)
{
	//const uint x = blockID.x * 8 + threadID.x;
	//const uint y = blockID.y * 8 + threadID.y;
	const uint x = DTid.x;
	const uint y = DTid.y;

	if (x < g_width && y < g_height)
	{
		const uint index = g_width * y + x;
		const float3 val = txInput[uint2(x, y)];

		if (g_nhwc)
		{
			rwTensor[index * 3] = val.x;
			rwTensor[index * 3 + 1] = val.y;
			rwTensor[index * 3 + 2] = val.z;
		}
		else
		{
			const uint planeSize = g_width * g_width;

			// RGB plane order since model was trained on this
			rwTensor[index] = val.x;
			rwTensor[index + planeSize] = val.y;
			rwTensor[index + planeSize * 2] = val.z;
		}
	}
}
