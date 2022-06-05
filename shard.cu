#include "rng.cuh"

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
