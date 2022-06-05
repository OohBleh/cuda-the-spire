#include <stdio.h>
#include "map.cu"
#include "neow.cu"
#include "pbox.cu"
#include "card-rewards.cu"
#include "qnodes.cu"
#include "custom.cu"

// ************************************************************** BEGIN PBox Kernel(s)

template<uint8 n, uint8 limit>
__global__ void pandoraSeedKernelFast(TestInfo info, bool* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testPandoraSeedFast<n, limit>(seed)) {
			results[totalIdx] = true;
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

// ************************************************************** END Fast Map Functions

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

// ************************************************************** BEGIN Custom Mode Functions

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

// ************************************************************** END Custom Mode Functions

// ************************************************************** BEGIN Shuffle Functions

static constexpr uint64 JAVA_MULTIPLIER = 0x5DEECE66DULL;
static constexpr uint64 JAVA_ADDEND = 0xBULL;
static constexpr uint64 JAVA_MASK = (1ULL << 48) - 1;

__forceinline__ __device__ bool shardFirst(const uint64 seed) {

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	// shuffle commons, uncommons, rares
	randomLong(seed0, seed1);
	randomLong(seed0, seed1);
	randomLong(seed0, seed1);
	
	uint64 shuffleSeed = randomLong(seed0, seed1);
	shuffleSeed = (shuffleSeed ^ JAVA_MULTIPLIER) & JAVA_MASK;
	shuffleSeed = (shuffleSeed * JAVA_MULTIPLIER + JAVA_ADDEND) & JAVA_MASK;
	
	uint64 tempSeed = shuffleSeed;
	//int32 r = static_cast<int32>(shuffleSeed >> (48 - 31));
	int32 r = static_cast<int32>(tempSeed >> 17);
	//static constexpr int bound = 16;
	//r = static_cast<int32>(((bound * static_cast<uint64>(r)) >> 31));
	r = static_cast<int32>(((16 * static_cast<uint64>(r)) >> 31));
	if (r == 8) {
		return true;
	}

	r = static_cast<int32>(tempSeed >> 17);
	int bound = 17;
	int m = bound - 1;
	if ((bound & m) == 0)  // i.e., bound is a power of 2
		r = static_cast<int32>(((bound * static_cast<uint64>(r)) >> 31));
	else {
		for (int32_t u = r; u - (r = u % bound) + m < 0; ) {
			shuffleSeed = (shuffleSeed * JAVA_MULTIPLIER + JAVA_ADDEND) & JAVA_MASK;
			u = static_cast<int32>(shuffleSeed >> (48 - 31));
		}
	}
	return r == 8;

}



__forceinline__ __device__ bool startsPBox(const uint64 seed) {
	
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	randomLong(seed0, seed1);
	randomLong(seed0, seed1);
	randomLong(seed0, seed1);
	randomLong(seed0, seed1);

	uint64 shuffleSeed = randomLong(seed0, seed1);

	
	// scramble seed
	shuffleSeed = (shuffleSeed ^ JAVA_MULTIPLIER) & JAVA_MASK;

	int size = 22;
	int K = 5;
	for (int i = size; i > 1; i--) {



		shuffleSeed = (shuffleSeed * JAVA_MULTIPLIER + JAVA_ADDEND) & JAVA_MASK;
		int r = static_cast<int32>(shuffleSeed >> (48 - 31));
		int bound = i;
		int m = bound - 1;
		if ((bound & m) == 0)  // i.e., bound is a power of 2
			r = static_cast<int32>(((bound * static_cast<uint64>(r)) >> 31));
		else {
			for (int32_t u = r; u - (r = u % bound) + m < 0; ) {
				shuffleSeed = (shuffleSeed * JAVA_MULTIPLIER + JAVA_ADDEND) & JAVA_MASK;
				u = static_cast<int32>(shuffleSeed >> (48 - 31));
			}
		}

		if (5 == r) {
			return false;
		}

		else if (K == r) {
			K = i - 1;
		}
		else if (K == i - 1) {
			K = r;
		}
	}

	return K < 3;
}


__forceinline__ __device__ bool hyperbeamFirstShop(const uint64 seed) {

	constexpr uint8 D_NUM_A = 18;
	constexpr uint8 D_NUM_B = 36;
	constexpr uint8 D_NUM_C = 17;

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	//first card
	int8 roll = random64Fast<100>(seed0, seed1);
	bool is_common = (roll >= 35);

	uint16 card1 = (is_common) ? random64Fast<D_NUM_A>(seed0, seed1) : (random64Fast<D_NUM_B>(seed0, seed1) + D_NUM_A);

	int8 adj = (is_common) ? (4) : (5);

	//second card
	roll = random64Fast<100>(seed0, seed1) + adj;
	is_common = (roll >= 40);

	uint16 card2;
	do {
		card2 = (is_common) ? random64Fast<D_NUM_A>(seed0, seed1) : (random64Fast<D_NUM_B>(seed0, seed1) + D_NUM_A);
	} while (card2 == card1);

	adj = (is_common) ? (adj - 1) : adj;

	//third card
	roll = random64Fast<100>(seed0, seed1) + adj;
	is_common = (roll >= 40);

	uint16 card3;
	do {
		card3 = (is_common) ? random64Fast<D_NUM_A>(seed0, seed1) : (random64Fast<D_NUM_B>(seed0, seed1) + D_NUM_A);
	} while ((card3 == card1) || (card3 == card2));

	random64Fast<2>(seed0, seed1);
	random64Fast<2>(seed0, seed1);
	random64Fast<2>(seed0, seed1);

	adj = (is_common) ? (adj - 1) : adj;
	
	// first shop attack
	// rarity
	roll = random64Fast<100>(seed0, seed1) + adj;

	if (roll < 9) {
		// random rare attack
		card1 = random64Fast<5>(seed0, seed1);
		/*
		if (card1 == 2) {
			return true;
		}
		*/
	}
	else if (roll >= 46) {
		// random common attack
		card1 = 5 + random64Fast<10>(seed0, seed1);
	}
	else {
		// random uncommon attack
		card1 = 15 + random64Fast<8>(seed0, seed1);
	}

	// second shop attack
	do {
		roll = random64Fast<100>(seed0, seed1) + adj;
		if (roll < 9) {
			// random rare attack
			card2 = random64Fast<5>(seed0, seed1);
			/*
			if (card2 == 2) {
				return true;
			}
			*/
		}
		else if (roll >= 46) {
			// random common attack
			card2 = 5 + random64Fast<10>(seed0, seed1);
		}
		else {
			// random uncommon attack
			card2 = 15 + random64Fast<8>(seed0, seed1);
		}
	} while (card2 == card1);
	
	if (card1 != 2 && card2 != 2) {
		return false;
	};

	// hyperbeam is card 1 (0) or card 2 (1) in the shop
	uint8 hyperbeamPosition = (card1 == 2) ? 0 : 1;


	// first shop skill
	// rarity
	roll = random64Fast<100>(seed0, seed1) + adj;

	if (roll < 9) {
		// random rare skill
		card1 = random64Fast<6>(seed0, seed1);
		/*
		if (card1 == 2) {
			return true;
		}
		*/
	}
	else if (roll >= 46) {
		// random common skill
		card1 = 6 + random64Fast<8>(seed0, seed1);
	}
	else {
		// random uncommon skill
		card1 = 14 + random64Fast<19>(seed0, seed1);
	}

	// second shop skill
	do {
		roll = random64Fast<100>(seed0, seed1) + adj;
		if (roll < 9) {
			// random rare skill
			card2 = random64Fast<6>(seed0, seed1);
			/*
			if (card2 == 2) {
				return true;
			}
			*/
		}
		else if (roll >= 46) {
			// random common skill
			card2 = 6 + random64Fast<8>(seed0, seed1);
		}
		else {
			// random uncommon skill
			card2 = 14 + random64Fast<19>(seed0, seed1);
		}
	} while (card2 == card1);

	// power card (rarity & card)
	random64Fast<2>(seed0, seed1);
	random64Fast<2>(seed0, seed1);

	// colorless cards (uncommon and rare)
	random64Fast<2>(seed0, seed1);
	random64Fast<2>(seed0, seed1);
	
	// do some stuff to see if Blasphemy is generated at Act I boss reward
	// dud rarity roll
	random64Fast<2>(seed0, seed1);
	
	// rare Shard card roll
	if (random64Fast<77>(seed0, seed1) != 49) {
		//return false;
	}

	// is Liars Game first, second, or third event? & 
	// is Hyperbeam discounted?
	seed0 = murmurHash3(seed);
	seed1 = murmurHash3(seed0);
	
	// are the first two ?-nodes events?
	if (
		randomFloatFast(seed0, seed1) < 0.15f || 
		randomFloatFast(seed0, seed1) < 0.30f
	) {
		return false;
	}
	
	// 3rd, 4th, and 5th RNG calls coincide with 1st, 2nd, and 3rd event room
	static constexpr bool SERPENT_MISSED[9][8][7] = { true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,false,false,false,false,false,false,false,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,false,false,false,false,false,false,false,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,false,false,false,false,false,false,false,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,false,false,false,false,false,false,false,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,false,false,false,false,false,false,false,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,false,false,false,false,false,false,false,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,false,false,false,false,false,false,false,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,true,true,true,true,false,true,true,false,false,false,false,false,false,false,true,true,true,true,true,false,true,true,true,true,true,true,false,true };
	if (SERPENT_MISSED[random64Fast<9>(seed0, seed1)][random64Fast<8>(seed0, seed1)][random64Fast<7>(seed0, seed1)]) {
		return false;
	}

	random64Fast<2>(seed0, seed1);
	random64Fast<2>(seed0, seed1);


	// 7th RNG call coincides with position of discounted card
	return random64Fast<5>(seed0, seed1) == hyperbeamPosition;
}



__global__ void shardKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (
			neowsLament(seed) 
			&& shardFirst(seed) 
			&& hyperbeamFirstShop(seed)
			&& startsPBox(seed)
		) {
			results[totalIdx] = seed;
			return;
		}
	}
}



// ************************************************************** END Shuffle Functions



cudaError_t testPandoraSeedsWithCuda(TestInfo info, FunctionType fnc, uint64* results)
{
	const unsigned int totalThreads = info.blocks * info.threads;
	uint64* dev_results = nullptr;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_results, totalThreads * sizeof(uint64));
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
	cudaStatus = cudaMemcpy(results, dev_results, totalThreads * sizeof(uint64), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	//cudaFree(dev_threads);
	cudaFree(dev_results);

	return cudaStatus;
}
