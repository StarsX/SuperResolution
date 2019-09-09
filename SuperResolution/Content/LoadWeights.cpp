//--------------------------------------------------------------------------------------
// LoadWeights.cpp
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. Copyright (C) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

#include "LoadWeights.h"
#include <fstream>

using namespace std;
using namespace XUSG::ML;

namespace
{
	const int c_bufferLength = 256;
}

// Loads weight values from a file.
bool LoadWeights(const string& fpath, WeightMapType& weightMap)
{
	ifstream input(fpath, ifstream::binary);
	if (!(input) || !(input.good()) || !(input.is_open()))
	{
		cerr << "Unable to open weight file: " << fpath << endl;

		return false;
	}

	int32_t count;
	try
	{
		input.read(reinterpret_cast<char*>(&count), 4);
	}
	catch (const ifstream::failure&)
	{
		cerr << "Invalid weight map file: " << fpath << endl;

		return false;
	}

	if (count < 0)
	{
		cerr << "Invalid weight map file: " << fpath << endl;

		return false;
	}

	cout << "Number of weight tensors: " + to_string(count) << endl;

	uint32_t name_len;
	uint32_t w_len;
	char name_buf[c_bufferLength];

	try
	{
		while (count--)
		{
			input.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
			if (name_len > c_bufferLength - 1)
			{
				cerr << "name_len exceeds c_bufferLength: " << name_len
					<< " vs " << c_bufferLength - 1 << endl;
				return false;
			}
			input.read(name_buf, name_len);
			name_buf[name_len] = '\0';
			string name(name_buf);

			input.read(reinterpret_cast<char*>(&w_len), sizeof(uint32_t));
			weightMap[name] = WeightsType(w_len);
			input.read(reinterpret_cast<char*>(weightMap[name].data()), sizeof(float) * w_len);

			cout << "Loaded tensor: " + name + " -> " + to_string(w_len) << endl;
		}

		input.close();
	}
	catch (const ifstream::failure&)
	{
		cerr << "Invalid tensor data" << endl;
		return false;
	}
	catch (const out_of_range&)
	{
		cerr << "Invalid tensor format" << endl;

		return false;
	}

	return true;
}
