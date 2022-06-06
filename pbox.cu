#include "rng.cuh"

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

/*
zyzz wants a seed that:
	(1) gets 6 copies of Vault
	(2) gets 1 copy of Alpha
	(3) swaps into Pandora's Box
	(4) starts with Stone Calendar
	(5) starts with Mercury Hourglass

here, we filter for (1) & (2)
*/

__forceinline__ __device__ bool zyzzTest(const uint64 seed) {
	
	// vault & alpha
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	uint8 num;
	uint8 nVault = 0;
	uint8 nAlpha = 0;

	for (uint8 i = 0; i < 7; i++) {
		num = random8Fast<71>(seed0, seed1);
		if (num == 63) {
			nVault++;
		}
		else if (num == 64) {
			nAlpha = 1;
		}

		if (nVault + nAlpha < i) {
			//return false;
		}
	}
	
	if (nVault + nAlpha <= 6) {
		//return false;
	}

	// hourglass & stone calendar
	seed0 = murmurHash3(seed);
	seed1 = murmurHash3(seed0);

	// common relics
	randomLong(seed0, seed1);

	// uncommon relics: MH is 27 of 30
	uint64 javaSeed = randomLong(seed0, seed1);
	if (!isFirstAfterShuffle<30, 27>(javaSeed)) {
		return false;
	}

	// rare relics: SC is 5 of 27
	javaSeed = randomLong(seed0, seed1);
	if (!isFirstAfterShuffle<27, 5>(javaSeed)) {
		return false;
	}

	// shop relics
	randomLong(seed0, seed1);

	/*
	boss relics:
		PBox is 5 of 21
		Holy Water is 19 of 21

		test for Holy Water in 1st position
		then test for PBox position
	*/
	javaSeed = randomLong(seed0, seed1);
	return isFirstAfterShuffle<21, 5>(javaSeed);
	
	// is shufflePosition bugged?
	/*
	uint64 javaSeedClone = javaSeed;
	uint8 pboxPos = shufflePosition<21, 5>(javaSeedClone);
	
	//return (isFirstAfterShuffle<21, 19>(javaSeed)) ? (pboxPos == 1) : (pboxPos == 0);
	return pboxPos == 0;
	*/
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
// ************************************************************** END Fast Pandora Functions
