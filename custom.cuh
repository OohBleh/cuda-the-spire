#pragma once
#include "rng.cuh"

__forceinline__ __device__ bool testSneck0andSpecializedRB(const uint64 seed) {

	/* getting 5 Skims from Specialized
	constexpr uint8 NUM_BR = 143;
	constexpr uint8 SKIM = 42;
	//constexpr uint8 CAI = 140;

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	if (random8Fast<NUM_BR>(seed0, seed1) != SKIM) {
		return false;
	}

	uint8 num0 = 0;
	*/

	// getting The Bomb from Specialized
	constexpr uint8 NUM_WHITE_RED_BLUE_CARDS = 35 + 71 + 72;
	constexpr uint8 BOMB_POSITION_RBW = 33 + 71 + 72;

	uint64 seed0 = seed; // murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	if (random8Fast<NUM_WHITE_RED_BLUE_CARDS>(seed0, seed1) != BOMB_POSITION_RBW) {
		return false;
	}


	/* 
		checks that 7 Snecko'd card costs sum to <= 2 
		this allows us to loop Skims, Fusion, Consume, Bomb
	*/
	uint64 seed2 = murmurHash3(inverseHash(seed) + 1);
	uint64 seed3 = murmurHash3(seed2);

	uint8 sneckoSum = 0;
	for (uint8 i = 0; i < 7; i++) {

		/*
		if (random8Fast<4>(seed2, seed3) == 0) {
			num0++;
		}
		*/

		sneckoSum += random8Fast<4>(seed2, seed3);
		if (sneckoSum > 2) {
			return false;
		}
	}

	return true;

	/*
	bool foundTheBomb = false;
	for (uint8 i = 0; i < 5; i++){
		if (randomFloatFast(seed0, seed1) >= 0.5f){
			random8Fast<2>(seed0, seed1);
			continue;
		}
		if (random8Fast<15>(seed0, seed1) == 11) {
			foundTheBomb = true;
			break;
		}
	}

	return foundTheBomb;
	*/
}

__forceinline__ __device__ bool testSealedRB(const uint64 seed) {

	constexpr uint8 NUM_UNCOMMONS_BLUE_RED_WHITE = 92;  // 20 COLORLESS + 36 RED + 36 BLUE
	constexpr bool isGood[NUM_UNCOMMONS_BLUE_RED_WHITE] = { 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
	};
	uint64 seed0 = seed; // murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);
	uint8 ctrs[5] = {};
	static constexpr uint8 goodInts[5] = {
		11 + 20,		// burning pact
		43 + 20,		// fusion or consume
		23 + 20,		// entrench
		42 + 20,		// consume or fusion
		67 + 20			// skim
	};

	static constexpr uint8 IGNORE = 0; // 15;

	for (uint8 i = 0; i < IGNORE; i++) {
		random8Fast<2>(seed0, seed1);
		random8Fast<2>(seed0, seed1);
	}

	for (uint8 i = 0; i < 30 - IGNORE; i++) {

		if (random8Fast<100>(seed0, seed1) > 34) {
			random8Fast<2>(seed0, seed1);
			continue;
		}

		uint8 card = random8Fast<NUM_UNCOMMONS_BLUE_RED_WHITE>(seed0, seed1);
		if (isGood[card]) {
			if (card == goodInts[0]) {
				ctrs[0]++;
			}
			else if (card == goodInts[1]) {
				ctrs[1]++;
			}
			else if (card == goodInts[2]) {
				ctrs[2]++;
			}
			else if (card == goodInts[3]) {
				ctrs[3]++;
			}
			else if (card == goodInts[4]) {
				ctrs[4]++;
			}
		}

	}

	return (ctrs[0] > 0) && (ctrs[1] > 0) && (ctrs[2] > 0) && (ctrs[3] > 0) && (ctrs[4] > 1);
}




__forceinline__ __device__ bool testSneck0andSpecializedB(const uint64 seed) {

	constexpr uint8 NUM_B = 71;
	constexpr uint8 CAI = 68;

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	if (random8Fast<NUM_B>(seed0, seed1) != CAI) {
		return false;
	}

	uint64 seed2 = murmurHash3(seed + 1);
	uint64 seed3 = murmurHash3(seed2);

	for (uint8 i = 0; i < 7; i++) {
		if (random8Fast<4>(seed2, seed3)) {
			return false;
		}
	}

	return true;
}

__forceinline__ __device__ bool testSealedB(const uint64 seed) {

	constexpr uint8 B_NUM_B = 36;
	//static constexpr bool isGood[BR_NUM_B] = { false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, };
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);
	uint8 ctrs[2] = {};
	static constexpr int8 goodInts[2] = {
		3, 17
	};

	for (uint8 i = 0; i < 28; i++) {
		random8Fast<2>(seed0, seed1);
		random8Fast<2>(seed0, seed1);
	}

	for (uint8 i = 0; i < 2; i++) {
		if (random8Fast<100>(seed0, seed1) > 34) {
			return;
		}
		uint8 card = random8Fast<B_NUM_B>(seed0, seed1);
		//if (isGood[card]) {
		if (true) {
			if (card == goodInts[0]) {
				ctrs[0]++;
			}
			else if (card == goodInts[1]) {
				ctrs[1]++;
			}
		}

	}

	return (ctrs[0] > 0) && (ctrs[1] > 0);
}
