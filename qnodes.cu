#include "rng.cuh"

template<uint8 nQNodes>
__forceinline__ __device__ bool onlyTreasures(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	static constexpr bool BAD_ROLLS[9][100] = { true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,false,false,true,true,true };


	for (uint8 i = 0; i < nQNodes; i++) {
		//float p = randomFloatFast(seed0, seed1);
		uint8 roll = static_cast<uint8>(100 * randomFloatFast(seed0, seed1));
		if (BAD_ROLLS[i][roll]) {
			return false;
		}
	}

	return true;
}

template<uint8 nQNodes>
__forceinline__ __device__ bool onlyShopsTreasures(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	uint8 combatThreshold = 10;
	uint8 shopThreshold = 3;
	uint8 treasureThreshold = 2;

	for (uint8 i = 0; i < nQNodes; i++) {
		//float p = randomFloatFast(seed0, seed1);
		uint8 roll = static_cast<uint8>(100 * randomFloatFast(seed0, seed1));
		if (roll < combatThreshold) {
			return false;
		}
		roll -= combatThreshold;

		if (roll < shopThreshold) {
			combatThreshold += 10;
			shopThreshold = 3;
			treasureThreshold += 2;
			continue;
		}

		roll -= shopThreshold;
		if (roll < treasureThreshold) {
			combatThreshold += 10;
			shopThreshold += 3;
			treasureThreshold = 2;
			continue;
		}
		return false;
	}

	return true;
}

__forceinline__ __device__ bool onlyShopsTreasures2(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	static constexpr uint8 NEXT_STATE[73][2] = { 1,2,5,4,3,6,9,8,7,10,11,8,7,12,15,14,13,16,17,14,13,18,19,14,13,20,23,22,21,24,25,22,21,26,27,22,21,28,29,22,21,30,33,32,31,34,35,32,31,36,37,32,31,38,39,32,31,40,41,32,31,42,45,44,43,46,47,44,43,48,49,44,43,50,51,44,43,52,53,44,43,54,55,44,43,56,59,58,57,60,61,58,57,62,63,58,57,64,65,58,57,66,67,58,57,68,69,58,57,70,71,58,57,72,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	static constexpr uint8 CUTOFFS[73][3] = { 10,13,15,20,23,27,20,26,28,30,33,37,30,36,38,30,33,39,30,39,41,40,43,47,40,46,48,40,43,49,40,49,51,40,43,51,40,52,54,50,53,57,50,56,58,50,53,59,50,59,61,50,53,61,50,62,64,50,53,63,50,65,67,60,63,67,60,66,68,60,63,69,60,69,71,60,63,71,60,72,74,60,63,73,60,75,77,60,63,75,60,78,80,70,73,77,70,76,78,70,73,79,70,79,81,70,73,81,70,82,84,70,73,83,70,85,87,70,73,85,70,88,90,70,73,87,70,91,93,80,83,87,80,86,88,80,83,89,80,89,91,80,83,91,80,92,94,80,83,93,80,95,97,80,83,95,80,98,100,80,83,97,80,101,103,80,83,99,80,104,106,90,93,97,90,96,98,90,93,99,90,99,101,90,93,101,90,102,104,90,93,103,90,105,107,90,93,105,90,108,110,90,93,107,90,111,113,90,93,109,90,114,116,90,93,111,90,117,119 };
	uint8 currState = 0;

	for (uint8 i = 0; i < 9; i++) {

		uint8 roll = static_cast<uint8>(100 * randomFloatFast(seed0, seed1));
		if (roll < CUTOFFS[currState][0] || roll >= CUTOFFS[currState][2]) {
			return false;
		}

		currState = (roll < CUTOFFS[currState][1]) ? NEXT_STATE[currState][0] : NEXT_STATE[currState][1];
	}

	return true;
}