//--------------------------------------------------------------------------------------
// LoadWeights.h
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. Copyright (C) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

#pragma once

#include "MachineLearning/XUSGMachineLearning.h"

bool LoadWeights(const std::string& fpath, XUSG::ML::WeightMapType& weightMap);
