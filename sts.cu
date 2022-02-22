
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
__forceinline__ __device__ bool testBadNeow(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);
	uint8 k = random64Fast<6>(seed0, seed1);
	if (k != 2) {
		return false;
	}
	k = random64Fast<5>(seed0, seed1);
	if ((k != 1) && (k != 4)) {
	//if (k != 4) {
	//if ((k == 0) || (k == 3)){
		return false;
	}
	k = random64Fast<4>(seed0, seed1);

	
	uint8 l = (k == 3) ? random64Fast<7>(seed0, seed1) : random64Fast<6>(seed0, seed1);
	l = (k == 2) ? (l + 1) : l;
	//l = (k == 1) ? 0 : l;
	return (l == 4) && (k != 1);

	//uint8 l = random64Fast<42>(seed0, seed1);
	//l = (k == 2) ? (l + 1) : l;
	//l = (k == 3) ? (l % 7) : (l % 6);
	//return (l == 4) || (k == 1);


	switch (k) {
	case 0:
		return random64Fast<6>(seed0, seed1) == 4;
		/*
		if (random64Fast<6>(seed0, seed1) != 4) {
			return false;
		}
		*/
	case 1:
		return false;
	case 2:
		return random64Fast<6>(seed0, seed1) == 3;
		/*
		if (random64Fast<6>(seed0, seed1) != 3) {
			return false;
		}
		*/
	case 3:
		return random64Fast<7>(seed0, seed1) == 4;
		/*
		if (random64Fast<7>(seed0, seed1) != 4) {
			return false;
		}
		*/

	}

	return true;
}











template<uint8 nCardRewards>
__forceinline__ __device__ bool testBadCardsFast(const uint64 seed) {

	/*
	uint64 seed0 = seed;
	uint64 seed1 = murmurHash3(seed0);
	*/

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	uint8 adj = 5;

	for (uint8 i = 1; i <= nCardRewards; i++) {

		bool foundrare = false;

		//first card
		uint8 roll = random64Fast<100>(seed0, seed1) + adj;

		bool is_rare = (roll < 3);
		bool is_common = (roll >= 40);
		bool is_uncommon = (!is_common) && (!is_rare);
		uint16 card_roll = random64Fast<627>(seed0, seed1);
		uint8 card_roll_mod19 = card_roll % 19;
		uint8 card_roll_mod33 = card_roll % 33;

		bool is_reject = (is_common) && (((1 << card_roll_mod19) & 188480) == 0);
		is_reject |= is_uncommon && (((1 << card_roll_mod33) & 1912611214) == 0);
		is_reject |= is_rare && (((1 << card_roll_mod19) & 67120) == 0);

		if (is_reject) {
			return false;
		}

		uint8 card1 = (is_rare) ? (card_roll_mod19 + 52) : card_roll_mod19;
		card1 = (is_uncommon) ? (card_roll_mod33 + 19) : card1;

		foundrare = is_rare;
		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;

		/*
		uint8 card1;

		//rare
		if (roll < 3) {
			card1 = random64Fast<19>(seed0, seed1);
			if ((1 << card1) & 67120) {
				adj = 5;
				foundrare = true;
				card1 += 52;
			}
			else {
				return false;
			}
		}

		//common
		else if (roll >= 40) {
			card1 = random64Fast<19>(seed0, seed1);
			if ((1 << card1) & 188480) {
				adj -= 1;
			}
			else {
				return false;
			}
		}

		//uncommon
		else {
			card1 = random64Fast<33>(seed0, seed1);
			if ((1 << card1) & 1912611214) {
				card1 += 19;
			}
			else {
				return false;
			}
		}
		*/

		//second card

		roll = random64Fast<100>(seed0, seed1) + adj;
		uint8 card2;

		is_rare = (roll < 3);
		is_common = (roll >= 40);
		is_uncommon = (!is_common) && (!is_rare);

		do {
			card_roll = random64Fast<627>(seed0, seed1);
			card_roll_mod19 = card_roll % 19;
			card_roll_mod33 = card_roll % 33;

			is_reject = is_common && (((1 << card_roll_mod19) & 188480) == 0);
			is_reject |= is_uncommon && (((1 << card_roll_mod33) & 1912611214) == 0);
			is_reject |= is_rare && (((1 << card_roll_mod19) & 67120) == 0);

			if (is_reject) {
				return false;
			}

			//card2 = card_roll_mod19 + ((uint8)is_rare) * 52;
			//card2 += ((uint8)is_uncommon) * (card_roll_mod33 + 19 - card2);



			card2 = (is_rare) ? (card_roll_mod19 + 52) : card_roll_mod19;
			card2 = (is_uncommon) ? (card_roll_mod33 + 19) : card2;


		} while (card2 == card1);

		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;
		foundrare |= is_rare;

		/*
		//rare
		if (roll < 3) {

			while (card2 == card1) {

				card2 = random64Fast<19>(seed0, seed1);

				if ((1 << card2) & 67120) {
					card2 += 52;
				}
				else {
					return false;
				}
			}

			adj = 5;
			foundrare = true;
		}

		//common
		else if (roll >= 40) {

			while (card2 == card1) {

				card2 = random64Fast<19>(seed0, seed1);

				if ((1 << card2) & 188480) {}
				else {
					return false;
				}
			}

			adj -= 1;
		}

		//uncommon
		else {

			while (card2 == card1) {

				card2 = random64Fast<33>(seed0, seed1);

				if ((1 << card2) & 1912611214) {
					card2 += 19;
				}
				else {
					return false;
				}
			}
		}

		*/

		//third card
		roll = random64Fast<100>(seed0, seed1) + adj;
		uint8 card3;

		is_rare = (roll < 3);
		is_common = (roll >= 40);
		is_uncommon = (!is_common) && (!is_rare);

		do {
			card_roll = random64Fast<627>(seed0, seed1);
			card_roll_mod19 = card_roll % 19;
			card_roll_mod33 = card_roll % 33;

			is_reject = is_common && (((1 << card_roll_mod19) & 188480) == 0);
			is_reject |= is_uncommon && (((1 << card_roll_mod33) & 1912611214) == 0);
			is_reject |= is_rare && (((1 << card_roll_mod19) & 67120) == 0);

			if (is_reject) {
				return false;
			}

			card3 = (is_rare) ? (card_roll_mod19 + 52) : card_roll_mod19;
			card3 = (is_uncommon) ? (card_roll_mod33 + 19) : card3;
		} while ((card3 == card1) || (card3 == card2));

		foundrare |= is_rare;
		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;


		/*
		if (i == 1) {
			if ((roll != 24) || seed != 1) {
			return false;
			}
		}
		*/

		/*
		uint8 card3 = card1;
		//rare
		if (roll < 3) {

			while ((card3 == card1) || (card3 == card2)) {

				card3 = random64Fast<19>(seed0, seed1);

				if ((1 << card3) & 67120) {
					card3 += 52;
				}
				else {
					return false;
				}
			}

			adj = 5;
			foundrare = true;
		}

		//common
		else if (roll >= 40) {

			while ((card3 == card1) || (card3 == card2)) {

				card3 = random64Fast<19>(seed0, seed1);

				if ((1 << card3) & 188480) {}
				else {
					return false;
				}
			}

			adj -= 1;
		}

		//uncommon
		else {

			while ((card3 == card1) || (card3 == card2)) {

				card3 = random64Fast<33>(seed0, seed1);

				if ((1 << card3) & 1912611214) {
					card3 += 19;
				}
				else {
					return false;
				}
			}
		}
		*/

		if (foundrare) {
			random64Fast<2>(seed0, seed1);
			random64Fast<2>(seed0, seed1);
		}
		else {
			random64Fast<2>(seed0, seed1);
			random64Fast<2>(seed0, seed1);
			random64Fast<2>(seed0, seed1);
		}

		/*
		if (i == 1) {
			if ((card1 != 5) || (card2 != 1) || (card3 != 46)) {
				return false;
			}
		}
		if (i == 2) {
			if ((card1 != 40) || (card2 != 45) || (card3 != 50)) {
			//if ((card1 != 27) || (card2 != 45) || (card3 != 50)) {
					return false;
			}
		}

		if (i == 3) {
			if ((card1 != 20) || (card2 != 23) || (card3 != 25)) {
				return false;
			}
		}
		if (i == 4) {
			if ((card1 != 33) || (card2 != 4) || (card3 != 5)) {
				return false;
			}
		}
		if (i == 5) {
			if ((card1 != 48) || (card2 != 34) || (card3 != 9)) {
				return false;
			}

			else {
				return true;
			}
		}
		*/

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


template<uint8 nCardRewards>
__global__ void oohblehSeedKernel(TestInfo info, uint64* results) {
	const unsigned int totalIdx = blockIdx.x * info.threads + threadIdx.x;
	uint64 seed = info.start + static_cast<uint64>(totalIdx);

	results[totalIdx] = false;
	for (; seed < info.end; seed += info.blocks * info.threads)
	{
		if (testBadNeow<nCardRewards>(seed)) {
			//if (true || testNoPotionsFast<nCardRewards>(seed)) {
				if (testBadCardsFast<nCardRewards>(seed)) {
					results[totalIdx] = seed;
					return;
				}
			//}
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

	case FunctionType::OOH_BLEH:
		oohblehSeedKernel<3> << <info.blocks, info.threads >> > (info, dev_results);
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
