//--------------------------------------------------------------------------------------
// By XU, Tianchen
//--------------------------------------------------------------------------------------

cbuffer cb
{
	uint g_height;
	uint g_width;
	bool g_nhwc;
};

Buffer<half> g_roTensor;

float4 main(float4 Pos : SV_POSITION) : SV_TARGET
{
	float3 color;
	const uint2 pos = Pos.xy;
	const uint index = pos.y * g_width + pos.x;

	if (g_nhwc)
	{
		color.x = g_roTensor[index * 3];
		color.y = g_roTensor[index * 3 + 1];
		color.z = g_roTensor[index * 3 + 2];
	}
	else
	{
		const uint blockSize = g_height * g_width;
		color.x = g_roTensor[index];
		color.y = g_roTensor[index + blockSize];
		color.z = g_roTensor[index + 2 * blockSize];
	}

	return float4(color, 1.0);
}
