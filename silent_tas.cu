#include "rng.cuh"



__forceinline__ __device__ bool juzuNeowSerpent(const uint64 seed) {

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	// shuffle commons
	uint64 shuffleSeed = randomLong(seed0, seed1);
	javaScramble(shuffleSeed);


	int32 r = javaNext<31>(shuffleSeed);
	uint64 tempSeed = shuffleSeed;

	r = static_cast<int32>(tempSeed >> 17);
	static constexpr uint8 bound = 30;
	static constexpr uint8 m = bound - 1;

	for (int32 u = r; u - (r = u % bound) + m < 0; ) {
		u = javaNext<31>(shuffleSeed);
	}

	// Juzu is position 22 of 30
	if (r != 22) {
		return false;
	}

	// Neow is 3 (mod 5)
	if (random8Fast<5>(seed0, seed1) != 3) {
		return false;
	}

	// Ssserpent is 5 (mod 9)
	return random8Fast<9>(seed0, seed1) == 5;
}


__forceinline__ __device__ bool finaleFirstShop(const uint64 seed) {

	constexpr uint8 S_NUM_A = 19;
	constexpr uint8 S_NUM_B = 33;
	constexpr uint8 S_NUM_C = 19;

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	//first card
	int8 roll = random8Fast<100>(seed0, seed1);
	bool is_common = (roll >= 35);

	uint16 card1 = (is_common) ? random8Fast<S_NUM_A>(seed0, seed1) : (random8Fast<S_NUM_B>(seed0, seed1) + S_NUM_A);

	int8 adj = (is_common) ? (4) : (5);

	//second card
	roll = random8Fast<100>(seed0, seed1) + adj;
	is_common = (roll >= 40);

	uint16 card2;
	do {
		card2 = (is_common) ? random8Fast<S_NUM_A>(seed0, seed1) : (random8Fast<S_NUM_B>(seed0, seed1) + S_NUM_A);
	} while (card2 == card1);

	adj = (is_common) ? (adj - 1) : adj;

	//third card
	roll = random8Fast<100>(seed0, seed1) + adj;
	is_common = (roll >= 40);

	uint16 card3;
	do {
		card3 = (is_common) ? random8Fast<S_NUM_A>(seed0, seed1) : (random8Fast<S_NUM_B>(seed0, seed1) + S_NUM_A);
	} while ((card3 == card1) || (card3 == card2));

	random8Fast<2>(seed0, seed1);
	random8Fast<2>(seed0, seed1);
	random8Fast<2>(seed0, seed1);

	adj = (is_common) ? (adj - 1) : adj;

	// first shop attack
	// rarity
	roll = random8Fast<100>(seed0, seed1) + adj;

	if (roll < 9) {
		// random rare attack
		card1 = random8Fast<4>(seed0, seed1);
		/*
		if (card1 == 2) {
			return true;
		}
		*/
	}
	else if (roll >= 46) {
		// random common attack
		card1 = 4 + random8Fast<10>(seed0, seed1);
	}
	else {
		// random uncommon attack
		card1 = 14 + random8Fast<13>(seed0, seed1);
	}

	// second shop attack
	do {
		roll = random8Fast<100>(seed0, seed1) + adj;
		if (roll < 9) {
			// random rare attack
			card2 = random8Fast<4>(seed0, seed1);
			/*
			if (card2 == 2) {
				return true;
			}
			*/
		}
		else if (roll >= 46) {
			// random common attack
			card2 = 4 + random8Fast<10>(seed0, seed1);
		}
		else {
			// random uncommon attack
			card2 = 14 + random8Fast<13>(seed0, seed1);
		}
	} while (card2 == card1);

	if (card1 != 2 && card2 != 2) {
		return false;
	};

	// finale is card 1 (0) or card 2 (1) in the shop
	uint8 finalePosition = (card1 == 2) ? 0 : 1;
	
	// is Liars Game first event?
	// is finale discounted?
	seed0 = murmurHash3(seed);
	seed1 = murmurHash3(seed0);

	// is the first ?-nodes an event?	p1, p2
	if (
		randomFloatFast(seed0, seed1) < 0.15f ||
		randomFloatFast(seed0, seed1) < 0.25f
	) {
		return false;
	}

	// dud RNG rolls					k3 -- k7
	randomPreFloatFast(seed0, seed1);
	randomPreFloatFast(seed0, seed1);
	randomPreFloatFast(seed0, seed1);
	randomPreFloatFast(seed0, seed1);
	randomPreFloatFast(seed0, seed1);

	// 8th RNG call coincides with position of discounted card
	return random8Fast<5>(seed0, seed1) == finalePosition;
}

