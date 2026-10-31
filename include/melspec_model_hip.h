#pragma once
// Training placeholder. Auto conservatively stays on CPU until calibrated.
#ifndef MELSPEC_HIP_MODEL_CALIBRATED
#define MELSPEC_HIP_MODEL_CALIBRATED 0
#endif
#include "melspec_model_cuda.h"
#ifndef MELSPEC_MODEL_HIP_CONST
#define MELSPEC_MODEL_HIP_CONST 1.0
#endif
#ifndef MELSPEC_MODEL_HIP_WORK
#define MELSPEC_MODEL_HIP_WORK 0.0
#endif
#ifndef MELSPEC_MODEL_HIP_INPUT
#define MELSPEC_MODEL_HIP_INPUT 0.0
#endif
#ifndef MELSPEC_MODEL_HIP_OUTPUT
#define MELSPEC_MODEL_HIP_OUTPUT 0.0
#endif
#ifndef MELSPEC_MODEL_HIP_CACHE
#define MELSPEC_MODEL_HIP_CACHE 0.0
#endif
