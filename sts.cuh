
#ifndef STS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>
#include <iostream>

typedef std::int8_t int8;
typedef std::uint8_t uint8;

typedef std::int16_t int16;
typedef std::uint16_t uint16;

typedef std::int32_t int32;
typedef std::uint32_t uint32;

typedef std::uint64_t uint64;
typedef std::int64_t int64;

enum class FunctionType {
	PANDORA_71_8,
	PANDORA_72_8,
	BAD_SILENT,
	BAD_WATCHER,
	BAD_IRONCLAD,
	BAD_MAP,
	FAST_QNODES, 
	CUSTOM, 
	SHARD,
	ZYZZ
};

const std::int8_t searchLength = 5;

struct TestInfo {
	unsigned int blocks;
	unsigned int threads;
	std::uint64_t start;
	std::uint64_t end;

	FunctionType fnc;
	void* data;
};

cudaError_t testPandoraSeedsWithCuda(TestInfo info, FunctionType fnc, uint64* results);

#endif // !STS_CUH
