#include <iostream>
using std::cout;
using std::endl;

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
	return kernel<SeedType::HashedRunSeed>(info, results, [](uint64 seed) {return !testPandoraSeedFast<n, limit>(seed); });
}

__global__ void zyzzKernel(TestInfo info, uint64* results) {
	return kernel<SeedType::RunSeed>(info, results, [](uint64 seed) {return zyzzTest(seed); });
}

// ************************************************************** END PBox Kernel(s)

// ************************************************************** BEGIN Unwinnable Kernels

template<uint8 nCardRewards>
__global__ void badSilentKernel(TestInfo info, uint64* results) {
	
	auto filter = [=](uint64 seed) {
		if (!getsBadNeowOptions1<nCardRewards>(seed)) {
			return true;
		}
		if (!testBadSilentCardsFast<nCardRewards>(seed)) {
			return true;
		}
		return false;
	};
	return kernel<SeedType::RunSeed>(info, results, filter);
}

template<uint8 nCardRewards>
__global__ void badWatcherKernel(TestInfo info, uint64* results) {
	
	auto filter = [=](uint64 seed) {
		if (!getsBadNeowOptions2(seed)) {
			return true;
		}
		if (!getsBadWatcherCards<nCardRewards, SeedType::RunSeed>(seed)) {
			return true;
		}
		if (!floor6Bottleneck(seed)) {
			return true;
		}
		return false;
	};

	return kernel<SeedType::RunSeed>(info, results, filter);
}

template<uint8 nCardRewards>
__global__ void badIroncladKernel(TestInfo info, uint64* results) {
	auto filter = [=](uint64 seed) {
		if (!getsBadNeowOptions1<nCardRewards>(seed)) {
			return true;
		}
		if (!testBadIroncladCardsFast<nCardRewards>(seed)) {
			return true;
		}
		if (!testCombatQNodes<nCardRewards>(seed)) {
			return true;
		}
		return false;
	};
	return kernel<SeedType::RunSeed>(info, results, filter);
}

// ************************************************************** END Unwinnable Kernels

// ************************************************************** BEGIN Map Kernels

__global__ void bottleneckKernel(TestInfo info, uint64* results) {
	return kernel<SeedType::RunSeed>(info, results, [](uint64 seed) {return !floor6Bottleneck(seed); });
}

// ************************************************************** END Map Kernels

// ************************************************************** BEGIN ?-Node Kernel(s)

/*TODO: use kernel function*/
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
			if (writeResults<SeedType::RunSeed>(totalIdx, width, seed, ctr, results)) {
				return;
			}
		}
	}
}

// ************************************************************** END ?-Node Kernel(s)

// ************************************************************** BEGIN Custom Mode Kernel(s)

/*TODO: use kernel function*/
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

/*TODO: use kernel function*/
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

/*TODO: use kernel function*/
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
			if (writeResults<SeedType::RunSeed>(totalIdx, width, seed, ctr, results)) {
				return;
			}
		}
	}
}

// ************************************************************** END Shard Kernel(s)
// 
// ************************************************************** Begin TAS Kernel(s)

/*TODO: use kernel function*/
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
			if (writeResults<SeedType::RunSeed>(totalIdx, width, seed, ctr, results)) {
				return;
			}
		}
	}
}

/*TODO: use kernel function*/
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
			if (writeResults<SeedType::RunSeed>(totalIdx, width, seed, ctr, results)) {
				return;
			}
		}
	}
}

// ************************************************************** END Shard Kernel(s)


cudaError_t testSeedsWithCuda(TestInfo info, uint64* results)
{
	const unsigned int totalThreads = info.blocks * info.threads;
	const unsigned int width = info.width;
	uint64* dev_results = nullptr;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_results, width * totalThreads * sizeof(uint64));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		goto Error;
	}

	switch (info.fnc) {
	case FunctionType::PANDORA_71_8:
		pandoraSeedKernel<71, 6> <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::PANDORA_72_8:
		pandoraSeedKernel<72, 8> <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::BAD_SILENT:
		badSilentKernel<3> <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::BAD_WATCHER:
		badWatcherKernel<2> <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::BAD_IRONCLAD:
		badIroncladKernel<3> <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::FAST_QNODES:
		fastQNodeKernel <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::CUSTOM:
		SealedRBKernel <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::SHARD:
		shardKernel <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::ZYZZ:
		zyzzKernel <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::SILENT_TAS:
		tasKernel <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	case FunctionType::IRONCLAD_TAS:
		tasKernel2 <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	
	case FunctionType::BOTTLENECK:
		bottleneckKernel <<<info.blocks, info.threads >>> (info, dev_results);
		break;

	default:
		break;
	}

	// Launch a kernel on the GPU with one thread for each element.


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		cout << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!" << endl;
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(results, dev_results, width * totalThreads * sizeof(uint64), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy failed!" << endl;
		goto Error;
	}

Error:
	//cudaFree(dev_threads);
	cudaFree(dev_results);

	return cudaStatus;
}
