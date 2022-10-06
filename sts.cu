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

/*TODO: fix this! */
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

/*TODO: fix this! */
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

/*TODO: fix this! */
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
__global__ void unwinnableKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	const unsigned int width = info.width;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);
	uint16 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads) {

		if (!testBadNeow2(seed)) {
			continue;
		}
		if (!testBadWatcherCardsFast<3>(seed)) {
			continue;
		}

		if (!floor6Bottleneck(seed)) {
			continue;
		}
		if (writeResults(totalIdx, width, seed, ctr, results)) {
			return;
		}
	}
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
	const unsigned int width = info.width;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);
	uint16 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads) {

		if (!testBadNeow2(seed)) {
			continue;
		}
		if (!testBadWatcherCardsFast<2>(seed)) {
			continue;
		}

		if (!floor6Bottleneck(seed)) {
			continue;
		}
		if (writeResults(totalIdx, width, seed, ctr, results)) {
			return;
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
	const unsigned int width = info.width;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);
	uint16 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads) {

		if (!testBadNeow2(seed)) {
			continue;
		}
		if (!testBadWatcherCardsFast<2>(seed)) {
			continue;
		}

		/*if (testSeedForSinglePath<3>(seed)) {
			results[totalIdx] = seed;
			return;
		}*/

		if (!floor6Bottleneck(seed)) {
			continue;
		}
		if (writeResults(totalIdx, width, seed, ctr, results)) {
			return;
		}
	}
}

__global__ void bottleneckKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	const unsigned int width = info.width;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);
	uint16 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (!floor6Bottleneck(seed)) {
			continue;
		}
		if (writeResults(totalIdx, width, seed, ctr, results)) {
			return;
		}
	}
}

// ************************************************************** END Map Kernels

// ************************************************************** BEGIN ?-Node Kernel(s)

__global__ void fastQNodeKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	const unsigned int width = info.width;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);
	uint16 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (
			onlyShopsTreasures<9>(seed)
			&& neowsLament(seed)
			) {
			if (writeResults(totalIdx, width, seed, ctr, results)) {
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
	uint16 ctr = 0;

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
	uint16 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (
			juzuNeowFirstEvent<5, false, 0>(seed)
			//&& finaleFirstShop(seed)
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
	uint16 ctr = 0;

	for (int i = 0; i < width; i++) {
		results[width * totalIdx + i] = false;
	}

	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (
			juzuNeowFirstEvent<1, 13, 1>(seed)
			&& shrineShop<false>(seed)
			&& startsPBox(seed)
		) {
			if (writeResults(totalIdx, width, seed, ctr, results)) {
				return;
			}
		}
	}
}

// ************************************************************** END Shard Kernel(s)


cudaError_t testSeedsWithCuda(TestInfo info, FunctionType fnc, uint64* results)
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
		pandoraSeedKernelFast<71, 8> <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::PANDORA_72_8:
		pandoraSeedKernelFast<72, 8> << <info.blocks, info.threads >> > (info, dev_results);
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

	
	case FunctionType::BOTTLENECK:
		bottleneckKernel << <info.blocks, info.threads >> > (info, dev_results);
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
