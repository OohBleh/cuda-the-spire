
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

constexpr uint64 ONE_BILLION = 1'000'000'000ULL;
constexpr uint64 ONE_TRILLION = 1'000'000'000'000ULL;

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
	void* data = nullptr;

	TestInfo(std::array<unsigned int, 6> parameters, FunctionType fun) :
		blocks(parameters[0]), threads(parameters[1]), width(parameters[2]),
		batchSizeInBillions(parameters[3]), verbosity(parameters[5]),
		start(parameters[4] * static_cast<std::uint64_t>(batchSizeInBillions)* ONE_BILLION),
		end(start + static_cast<std::uint64_t>(batchSizeInBillions) * ONE_BILLION),
		fnc(fun)
	{}

	void incrementInterval() {
		start += static_cast<std::uint64_t>(batchSizeInBillions) * ONE_BILLION;
		end += static_cast<std::uint64_t>(batchSizeInBillions) * ONE_BILLION;
	}
};

cudaError_t testSeedsWithCuda(TestInfo info, uint64* results);


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

template<typename F>
__device__ void kernel(TestInfo info, uint64* results, F filter) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	const unsigned int width = info.width;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);
	uint16 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads) {

		if (filter(seed)) {
			continue;
		}
		if (writeResults(totalIdx, width, seed, ctr, results)) {
			return;
		}
	}
}



// ************************************************************** END Utility Function(s)

#endif // !STS_CUH
