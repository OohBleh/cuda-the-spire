
#include "sts.cuh"
#include <stdio.h>

__forceinline__ __device__ uint64 murmurHash3(uint64 x) {
	x ^= x >> 33;
	x *= static_cast<uint64>(-49064778989728563LL);
	x ^= x >> 33;
	x *= static_cast<uint64>(-4265267296055464877LL);
	x ^= x >> 33;
	return x;
}

template<uint16 n>
__forceinline__ __device__ uint64 random64(uint64& seed0, uint64& seed1) {
	bool didOverflow;
	uint64 value;
	do {
		uint64 s1 = seed0;
		uint64 s0 = seed1;
		seed0 = s0;
		s1 ^= s1 << 23;
		seed1 = s1 ^ s0 ^ s1 >> 17 ^ s0 >> 26;
		auto nextLong = seed1 + s0;

		uint64 bits = nextLong >> 1;
		value = bits % n;

		didOverflow =
			static_cast<int64>(bits - value + n - 1) < 0LL;

	} while (didOverflow);

	return value;
}

template<uint16 n>
__forceinline__ __device__ uint16 random64Fast(uint64& seed0, uint64& seed1) {
	uint16 value;
	uint64 s1 = seed0;
	uint64 s0 = seed1;
	seed0 = s0;
	s1 ^= s1 << 23;
	seed1 = s1 ^ s0 ^ s1 >> 17 ^ s0 >> 26;
	uint64 bits = (seed0 + seed1) >> 1;
	value = bits % n;
	return value;
}

__forceinline__ __device__ float randomFloatFast(uint64& seed0, uint64& seed1) {
	
	static constexpr double NORM_FLOAT = 5.9604644775390625E-8;

	uint64 s1 = seed0;
	uint64 s0 = seed1;
	seed0 = s0;
	s1 ^= s1 << 23;
	seed1 = s1 ^ s0 ^ s1 >> 17 ^ s0 >> 26;
	
	uint64 x = (seed0 + seed1) >> 40;
	double d = static_cast<double>(x) * NORM_FLOAT;
	return static_cast<float>(d);
}



// ************************************************************** BEGIN Accurate Pandora Functions

// the random call should be 1 more than on the cpu version
template<uint8 n, uint8 limit>
__forceinline__ __device__ bool testPandoraSeed(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	uint8 num = random64<n>(seed0, seed1, n);

	for (int8 i = 1; i < 7; i++) {
		if (random64<n>(seed0, seed1, n) != num) {
			return false;
		}
	}
	return true;
}

template<uint8 n, uint8 limit>
__global__ void pandoraSeedKernel(TestInfo info, bool* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testPandoraSeed<n, limit>(seed)) {
			results[totalIdx] = true;
			return;
		}
	}
}

// ************************************************************** END Accurate Pandora Functions

// ************************************************************** BEGIN Fast Pandora Functions


template<uint8 n, uint8 limit>
__forceinline__ __device__ bool testPandoraSeedFast(const uint64 seed) {
	uint64 seed0 = seed;
	uint64 seed1 = murmurHash3(seed0);

	const uint8 num = random64Fast<n>(seed0, seed1);

	for (uint8 i = 1; i < limit; i++) {
		if (random64Fast<n>(seed0, seed1) != num) {
			return false;
		}
	}

	return true;
}

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

// ************************************************************** END Fast Pandora Functions

// ************************************************************** BEGIN Fast Pandora Functions








template<uint8 nCardRewards>
__forceinline__ __device__ bool testBadNeow1(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	/*
		[ ] THREE_CARDS = 0,
		[ ] ONE_RANDOM_RARE_CARD,
		[x] REMOVE_CARD,
		[?] UPGRADE_CARD,
		[ ] TRANSFORM_CARD,
		[ ] RANDOM_COLORLESS,
	*/
	uint8 k = random64Fast<6>(seed0, seed1);
	if (k != 2) {
		return false;
	}

	/*
		[ ] THREE_SMALL_POTIONS,
		[x] RANDOM_COMMON_RELIC,
		[x] TEN_PERCENT_HP_BONUS,
		[ ] THREE_ENEMY_KILL,
		[x] HUNDRED_GOLD,
	*/
	k = random64Fast<5>(seed0, seed1);
	//if ((k != 1) && (k != 4)) {
	//if (k != 4) {
	if ((k == 0) || (k == 3)) {
		return false;
	}
	k = random64Fast<4>(seed0, seed1);
	/*										0 HP_LOSS	1 NO_GOLD	2 CURSE		3 DAMAGE
			0 [ ] RANDOM_COLORLESS_2,			0			0			0			0
			1 [ ] REMOVE_TWO,					1			1			=====		1
			2 [ ] ONE_RARE_RELIC,				2			2			1			2
			3 [ ] THREE_RARE_CARDS,				3			3			2			3
			4 [x] TWO_FIFTY_GOLD,				[4]			=====		[3]			[4]
			5 [ ] TRANSFORM_TWO_CARDS,			5			4			4			5
			6 [x] TWENTY_PERCENT_HP_BONUS,		=====		[5]			[5]			[6]
	*/

	uint8 l = (k == 3) ? random64Fast<7>(seed0, seed1) : random64Fast<6>(seed0, seed1);
	l = (k == 1) ? (l - 1) : l;
	l = (k == 2) ? (l + 1) : l;
	//l = (k == 1) ? 0 : l;
	return (l == 4) || (l == 6);

	return true;
}


template<uint8 nCardRewards>
__forceinline__ __device__ bool testBadNeow2(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	/*
		[ ] 0 THREE_CARDS = 0,
		[ ] 1 ONE_RANDOM_RARE_CARD,
		[ ] 2 REMOVE_CARD,
		[ ] 3 UPGRADE_CARD,
		[x] 4 TRANSFORM_CARD,
		[ ] 5 RANDOM_COLORLESS,
	*/
	
	if (random64Fast<6>(seed0, seed1) != 4) {
		return false;
	}

	/*
		[ ] THREE_SMALL_POTIONS,
		[x] RANDOM_COMMON_RELIC,
		[x] TEN_PERCENT_HP_BONUS,
		[ ] THREE_ENEMY_KILL,
		[x] HUNDRED_GOLD,
	*/
	uint8 k = random64Fast<5>(seed0, seed1);
	if ((k == 0) || (k == 3)) {
		return false;
	}
	k = random64Fast<4>(seed0, seed1);
	/*									0 HP_LOSS	1 NO_GOLD	2 CURSE		3 DAMAGE
		[ ] RANDOM_COLORLESS_2,				0			0			0			0
		[ ] REMOVE_TWO,						1			1			=====		1
		[ ] ONE_RARE_RELIC,					2			2			1			2
		[ ] THREE_RARE_CARDS,				3			3			2			3
		[x] TWO_FIFTY_GOLD,					[4]			=====		[3]			[4]
		[ ] TRANSFORM_TWO_CARDS,			5			4			4			5
		[x] TWENTY_PERCENT_HP_BONUS,		=====		[5]			[5]			[6]
	*/

	uint8 l = (k == 3) ? random64Fast<7>(seed0, seed1) : random64Fast<6>(seed0, seed1);
	l = (k == 1) ? (l - 1) : l;
	l = (k == 2) ? (l + 1) : l;
	//l = (k == 1) ? 0 : l;
	return (l == 4) || (l == 6);

	return true;
}

template<uint8 nCardRewards>
__forceinline__ __device__ bool testBadSilentCardsFast(const uint64 seed) {

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	constexpr uint64 BAD_COMMON = 188480;
	constexpr uint64 BAD_UNCOMMON = 1912611214;
	constexpr uint64 BAD_RARE = 67120;

	constexpr uint16 NUM_A = 19;
	constexpr uint16 NUM_B = 33;
	constexpr uint16 NUM_AB = 52;
	//constexpr uint16 NUM_C = 19;
	constexpr uint16 NUM_GCD = 627;

	uint8 adj = 5;

	for (uint8 i = 1; i <= nCardRewards; i++) {

		bool foundrare = false;

		//first card
		uint8 roll = random64Fast<100>(seed0, seed1) + adj;

		bool is_rare = (roll < 3);
		bool is_common = (roll >= 40);
		bool is_uncommon = (!is_common) && (!is_rare);
		uint16 card_roll = random64Fast<NUM_GCD>(seed0, seed1);
		uint8 card_roll_modA = card_roll % NUM_A;
		uint8 card_roll_modB = card_roll % NUM_B;

		bool is_reject = (is_common) && (((1 << card_roll_modA) & BAD_COMMON) == 0);
		is_reject |= is_uncommon && (((1 << card_roll_modB) & BAD_UNCOMMON) == 0);
		is_reject |= is_rare && (((1 << card_roll_modA) & BAD_RARE) == 0);

		if (is_reject) {
			return false;
		}

		uint8 card1 = (is_rare) ? (card_roll_modA + NUM_AB) : card_roll_modA;
		card1 = (is_uncommon) ? (card_roll_modB + NUM_A) : card1;

		foundrare = is_rare;
		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;

		//second card
		roll = random64Fast<100>(seed0, seed1) + adj;
		uint8 card2;

		is_rare = (roll < 3);
		is_common = (roll >= 40);
		is_uncommon = (!is_common) && (!is_rare);

		do {
			card_roll = random64Fast<NUM_GCD>(seed0, seed1);
			card_roll_modA = card_roll % NUM_A;
			card_roll_modB = card_roll % NUM_B;

			is_reject = is_common && (((1 << card_roll_modA) & BAD_COMMON) == 0);
			is_reject |= is_uncommon && (((1 << card_roll_modB) & BAD_UNCOMMON) == 0);
			is_reject |= is_rare && (((1 << card_roll_modA) & BAD_RARE) == 0);

			if (is_reject) {
				return false;
			}

			card2 = (is_rare) ? (card_roll_modA + NUM_AB) : card_roll_modA;
			card2 = (is_uncommon) ? (card_roll_modB + NUM_A) : card2;


		} while (card2 == card1);

		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;
		foundrare |= is_rare;

		//third card
		roll = random64Fast<100>(seed0, seed1) + adj;
		uint8 card3;

		is_rare = (roll < 3);
		is_common = (roll >= 40);
		is_uncommon = (!is_common) && (!is_rare);

		do {
			card_roll = random64Fast<NUM_GCD>(seed0, seed1);
			card_roll_modA = card_roll % NUM_A;
			card_roll_modB = card_roll % NUM_B;

			is_reject = is_common && (((1 << card_roll_modA) & BAD_COMMON) == 0);
			is_reject |= is_uncommon && (((1 << card_roll_modB) & BAD_UNCOMMON) == 0);
			is_reject |= is_rare && (((1 << card_roll_modA) & BAD_RARE) == 0);

			if (is_reject) {
				return false;
			}

			card3 = (is_rare) ? (card_roll_modA + NUM_AB) : card_roll_modA;
			card3 = (is_uncommon) ? (card_roll_modB + NUM_A) : card3;
		} while ((card3 == card1) || (card3 == card2));

		foundrare |= is_rare;
		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;


		if (foundrare) {
			random64Fast<2>(seed0, seed1);
			random64Fast<2>(seed0, seed1);
		}
		else {
			random64Fast<2>(seed0, seed1);
			random64Fast<2>(seed0, seed1);
			random64Fast<2>(seed0, seed1);
		}
	}

	return true;
}

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
__global__ void badIroncladKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testBadNeow1<nCardRewards>(seed)) {
			if (testCombatQNodes<3>(seed)) {
				results[totalIdx] = seed;
				return;
			}
		}
	}
}

// ************************************************************** END Fast Pandora Functions

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

	case FunctionType::BAD_IRONCLAD:
		badIroncladKernel<3> << <info.blocks, info.threads >> > (info, dev_results);
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
