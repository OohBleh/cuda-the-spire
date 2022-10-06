
#ifndef STS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>
#include <iostream>

#include <array>

typedef std::int8_t int8;
typedef std::uint8_t uint8;

typedef std::int16_t int16;
typedef std::uint16_t uint16;

typedef std::int32_t int32;
typedef std::uint32_t uint32;

typedef std::uint64_t uint64;
typedef std::int64_t int64;

static const uint64 ONE_BILLION = 1'000'000'000ULL;
static const uint64 ONE_TRILLION = 1'000'000'000'000ULL;

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
	ZYZZ,
	SILENT_TAS, 
	IRONCLAD_TAS, 
	BOTTLENECK
};

const std::int8_t searchLength = 5;

struct TestInfo {
public:
	unsigned int blocks;
	unsigned int threads;
	unsigned int width;
	unsigned int batchSizeInBillions;
	unsigned int verbosity;

	std::uint64_t start;
	std::uint64_t end;

	FunctionType fnc;
	void* data;

	TestInfo(std::array<unsigned int, 6> parameters) {
		blocks = parameters[0];
		threads = parameters[1];
		width = parameters[2];
		batchSizeInBillions = parameters[3];
		start = parameters[4] * static_cast<std::uint64_t>(batchSizeInBillions) * ONE_BILLION;
		end = start + static_cast<std::uint64_t>(batchSizeInBillions) * ONE_BILLION;
		verbosity = parameters[5];
	}

	void incrementInterval() {
		start += static_cast<std::uint64_t>(batchSizeInBillions) * ONE_BILLION;
		end += static_cast<std::uint64_t>(batchSizeInBillions) * ONE_BILLION;
	}
};

cudaError_t testSeedsWithCuda(TestInfo info, FunctionType fnc, uint64* results);


// ************************************************************** BEGIN Utility Function(s)

__forceinline__ __device__ bool writeResults(
	const unsigned int totalIdx,
	const unsigned int width,
	const uint64 seed,
	uint16& ctr,
	uint64* results
) {
	results[width * totalIdx + ctr] = seed;
	ctr++;
	return ctr == width;
}
// ************************************************************** END Utility Function(s)

#endif // !STS_CUH
