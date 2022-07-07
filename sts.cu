#include <stdio.h>
#include "map.cu"
#include "neow.cu"
#include "pbox.cu"
#include "card-rewards.cu"
#include "qnodes.cu"
#include "custom.cu"
#include "shard.cu"
#include "tas.cu"




// ************************************************************** BEGIN PBox Kernel(s)

template<uint8 n, uint8 limit>
__global__ void pandoraSeedKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testPandoraSeed<n, limit>(seed)) {
			results[totalIdx] = seed;
			return;
		}
	}
}

template<uint8 n, uint8 limit>
__global__ void pandoraSeedKernelFast(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testPandoraSeedFast<n, limit>(seed)) {
			results[totalIdx] = seed;
			return;
		}
	}
}

__global__ void zyzzKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (zyzzTest(seed)) {
			results[totalIdx] = seed;
			return;
		}
	}
}

// ************************************************************** END PBox Kernel(s)

// ************************************************************** BEGIN Unwinnable Kernels

template<uint8 nCardRewards>
__forceinline__ __device__ bool testNoPotionsFast(const uint64 seed) {

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	uint8 thresh = 40;

	for (uint8 i = 0; i < nCardRewards; i++) {

		if (random64Fast<100>(seed0, seed1) < thresh) {
			return false;
		}
		thresh += 10;
	}

	return true;
}

/*

combats with the left column filled in will be considered "bad"

				Blue Slaver	0 -- 4/32
4,5				Gremlin Gang	4/32 -- 6/32
				Looter	6/32 -- 10/32
10,11,12,13		Large Slime	10/32 -- 14/32
14,15			Lots of Slimes	14/32 -- 16/32
16,17,18		Exordium Thugs	16/32 -- 19/32
19,20,21		Exordium Wildlife	19/32 -- 22/32
				Red Slaver	22/32 -- 24/32
24,25,26,27		3 Louse	24/32 -- 28/32
				2 Fungi Beasts	28/32 -- 1

bit vector of "bad" combats as a uint32 is 255851568
the fights that cannot follow 2 louses are 251658240
the fights that cannot follow 2 louses are 64512
*/

template<uint8 nQNodes>
__forceinline__ __device__ bool testCombatQNodes(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	for (uint8 i = 0; i < nQNodes; i++) {
		float p = randomFloatFast(seed0, seed1);
		if (p > 0.1f) {
			return false;
		}
	}

	uint8 c2 = ((uint8)4 * randomFloatFast(seed0, seed1));
	while (c2 == 0) {
		c2 = (uint8)(4 * randomFloatFast(seed0, seed1));
	}

	uint8 c3 = ((uint8)4 * randomFloatFast(seed0, seed1));
	while ((c3 == 0) || (c3 == c2)) {
		c3 = (uint8)(4 * randomFloatFast(seed0, seed1));
	}

	uint8 c4 = ((uint8)32 * randomFloatFast(seed0, seed1));
	while (
		((c3 == 2) && ((1 << c4) & 251658240) != 0)
		|| ((c3 == 3) && ((1 << c4) & 64512) != 0)
		) {
		c4 = ((uint8)32 * randomFloatFast(seed0, seed1));
	}
	if (((1 << c4) & 255851568) == 0) {
		return false;
	}

	uint8 c5 = ((uint8)32 * randomFloatFast(seed0, seed1));
	while (c5 == c4) {
		c5 = ((uint8)32 * randomFloatFast(seed0, seed1));
	}
	if (((1 << c5) & 255851568) == 0) {
		return false;
	}
	return true;
}

template<uint8 nCardRewards>
__global__ void badSilentKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testBadNeow1<nCardRewards>(seed)) {
			if (testBadSilentCardsFast<nCardRewards>(seed)) {
				results[totalIdx] = seed;
				return;
			}
		}
	}
}

template<uint8 nCardRewards>
__global__ void badWatcherKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testBadNeow2(seed)) {
			if (testBadWatcherCardsFast<nCardRewards>(seed)) {
				results[totalIdx] = seed;
				return;
			}
		}
	}
}

template<uint8 nCardRewards>
__global__ void badIroncladKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (!testBadNeow1<nCardRewards>(seed)) {
			continue;
		}
		if (!testBadIroncladCardsFast<nCardRewards>(seed)) {
			continue;
		}
		if (!testCombatQNodes<2>(seed)) {
			continue;
		}
		results[totalIdx] = seed;
		return;
	}
}


// ************************************************************** END Unwinnable Kernels

// ************************************************************** BEGIN Map Kernels

__global__ void badMapKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		
		if (!testBadNeow2(seed)) {
continue;
		}
		if (!testBadWatcherCardsFast<2>(seed)) {
			continue;
		}

		if (testSeedForSinglePath<3>(seed)) {
			results[totalIdx] = seed;
			return;
		}
	}
}

// ************************************************************** END Map Kernels

// ************************************************************** BEGIN ?-Node Kernel(s)

__global__ void fastQNodeKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (onlyTreasures(seed)) {
			if (neowsLament(seed)) {
				results[totalIdx] = seed;
				return;
			}
		}
	}
}

// ************************************************************** END ?-Node Kernel(s)

// ************************************************************** BEGIN Custom Mode Kernel(s)

__global__ void SealedRBKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testSneck0andSpecializedRB(seed)) {
			if (testSealedRB(seed)) {
				results[totalIdx] = seed;
				return;
			}
		}
	}
}

__global__ void SealedBKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testSneck0andSpecializedB(seed)) {
			if (testSealedB(seed)) {
				results[totalIdx] = seed;
				return;
			}
		}
	}
}

// ************************************************************** END Custom Mode Kernel(s)

// ************************************************************** BEGIN Shard Kernel(s)


__global__ void shardKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	const unsigned int width = info.width;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);
	uint8 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (
			shardNeowFirst(seed)
			&& hyperbeamFirstShop(seed)
			&& startsPBox(seed)
		) {
			/*
			//results[totalIdx] = seed;
			results[width * totalIdx + ctr] = seed;
			ctr++;
			if (ctr == width) {
				return;
			}
			//return;
			*/
			if (writeResults(totalIdx, width, seed, ctr, results)) {
				return;
			}
		}
	}
}

// ************************************************************** END Shard Kernel(s)
// 
// ************************************************************** Begin TAS Kernel(s)

__global__ void tasKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	const unsigned int width = info.width;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);
	uint8 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (
			juzuNeowFirstEvent<5, 9, 0>(seed)
			&& finaleFirstShop(seed)
			&& startsPBox(seed)
		) {
			/*
			//results[totalIdx] = seed;
			results[width * totalIdx + ctr] = seed;
			ctr++;
			if (ctr == width) {
				return;
			}
			//return;
			*/
			if (writeResults(totalIdx, width, seed, ctr, results)) {
				return;
			}
		}
	}
}

__global__ void tasKernel2(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	const unsigned int width = info.width;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);
	uint8 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (
			juzuNeowFirstEvent<1, 13, 1>(seed)
			&& shrineShop(seed)
			&& startsPBox(seed)
		) {
			if (writeResults(totalIdx, width, seed, ctr, results)) {
				return;
			}
		}
	}
}

// ************************************************************** END Shard Kernel(s)


cudaError_t testPandoraSeedsWithCuda(TestInfo info, FunctionType fnc, uint64* results)
{
	const unsigned int totalThreads = info.blocks * info.threads;
	const unsigned int width = info.width;
	uint64* dev_results = nullptr;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_results, width * totalThreads * sizeof(uint64));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//cudaStatus = cudaMemcpy(dev_results, results, totalThreads * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//    fprintf(stderr, "cudaMemcpy failed!");
	//    goto Error;
	//}

	switch (fnc) {
	case FunctionType::PANDORA_71_8:
		//pandoraSeedKernelFast<71, 8> <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::PANDORA_72_8:
		//pandoraSeedKernelFast<72, 8> << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::BAD_SILENT:
		badSilentKernel<3> << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::BAD_WATCHER:
		badWatcherKernel<3> << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::BAD_IRONCLAD:
		badIroncladKernel<5> << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::BAD_MAP:
		badMapKernel << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::FAST_QNODES:
		fastQNodeKernel << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::CUSTOM:
		SealedRBKernel << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::SHARD:
		shardKernel << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::ZYZZ:
		zyzzKernel << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::SILENT_TAS:
		tasKernel << <info.blocks, info.threads >> > (info, dev_results);
		break;

	case FunctionType::IRONCLAD_TAS:
		tasKernel2 << <info.blocks, info.threads >> > (info, dev_results);
		break;

	default:
		break;
	}

	// Launch a kernel on the GPU with one thread for each element.


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(results, dev_results, width * totalThreads * sizeof(uint64), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	//cudaFree(dev_threads);
	cudaFree(dev_results);

	return cudaStatus;
}
