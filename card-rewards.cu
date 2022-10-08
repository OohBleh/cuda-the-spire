#include "rng.cuh"

template<uint8 nCardRewards>
__forceinline__ __device__ bool testBadIroncladCardsFast(const uint64 seed) {

	constexpr uint64 BAD_COMMON = 24804;
	constexpr uint64 BAD_UNCOMMON = 30903371908;
	constexpr uint64 BAD_RARE = 36556;

	constexpr uint8 NUM_A = 20;
	constexpr uint8 NUM_B = 36;
	constexpr uint8 NUM_AB = 56;
	constexpr uint8 NUM_C = 16;
	constexpr uint16 NUM_GCD = 720;

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
		uint16 card_roll = random64Fast<NUM_GCD>(seed0, seed1);
		uint8 card_roll_modA = card_roll % NUM_A;
		uint8 card_roll_modB = card_roll % NUM_B;
		uint8 card_roll_modC = card_roll % NUM_C;

		bool is_reject = (is_common) && (((1 << card_roll_modA) & BAD_COMMON) == 0);
		is_reject |= is_uncommon && (((1 << card_roll_modB) & BAD_UNCOMMON) == 0);
		is_reject |= is_rare && (((1 << card_roll_modC) & BAD_RARE) == 0);

		if (is_reject) {
			return false;
		}

		uint8 card1 = (is_rare) ? (card_roll_modC + NUM_AB) : card_roll_modA;
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
			card_roll_modC = card_roll % NUM_C;

			is_reject = is_common && (((1 << card_roll_modA) & BAD_COMMON) == 0);
			is_reject |= is_uncommon && (((1 << card_roll_modB) & BAD_UNCOMMON) == 0);
			is_reject |= is_rare && (((1 << card_roll_modC) & BAD_RARE) == 0);

			if (is_reject) {
				return false;
			}

			card2 = (is_rare) ? (card_roll_modC + NUM_AB) : card_roll_modA;
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
			card_roll_modC = card_roll % NUM_C;

			is_reject = is_common && (((1 << card_roll_modA) & BAD_COMMON) == 0);
			is_reject |= is_uncommon && (((1 << card_roll_modB) & BAD_UNCOMMON) == 0);
			is_reject |= is_rare && (((1 << card_roll_modC) & BAD_RARE) == 0);

			if (is_reject) {
				return false;
			}

			card3 = (is_rare) ? (card_roll_modC + NUM_AB) : card_roll_modA;
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

template<uint8 nCardRewards, SeedType seedType>
__forceinline__ __device__ bool getsBadWatcherCards(const uint64 seed) {
	
	uint64 seed0, seed1;

	switch (seedType){
	case SeedType::RunSeed:
		seed0 = murmurHash3(seed);
		seed1 = murmurHash3(seed0);
		break;
	case SeedType::HashedRunSeed:
		seed0 = seed;
		seed1 = murmurHash3(seed0);
		break;
	default:
		break;
	}
	
	return getsBadWatcherCards<nCardRewards>(seed0, seed1);
}
	
template<uint8 nCardRewards>
__forceinline__ __device__ bool getsBadWatcherCards(uint64 seed0, uint64 seed1) {

	constexpr bool isBad[71] = { false,false,false,true,false,false,true,false,false,false,true,false,false,false,false,false,false,true,false,false,false,false,false,false,false,false,false,false,true,false,false,false,false,true,false,false,true,true,false,false,false,true,false,false,true,true,true,false,false,true,false,false,false,false,true,true,true,true,false,false,false,false,false,false,false,true,true,false,true,false,true, };
	
	constexpr uint8 nCommons = 19;
	constexpr uint8 nUncommons = 35;
	constexpr uint8 nRares = 17;

	constexpr uint8 nCommonsAndUncommons = 54;
	constexpr uint16 lcmOfUncommonsAndRares = 595;

	//int8 adj = 5;


	//first card
	int8 roll = random64Fast<100>(seed0, seed1);
	bool is_common = (roll >= 35);

	uint16 card1 = (is_common) ? random64Fast<nCommons>(seed0, seed1) : (random64Fast<nUncommons>(seed0, seed1) + nCommons);

	if (!isBad[card1]) {
		return false;
	}

	int8 adj = (is_common) ? (4) : (5);

	//second card
	roll = random64Fast<100>(seed0, seed1) + adj;
	is_common = (roll >= 40);

	uint16 card2;
	do {
		card2 = (is_common) ? random64Fast<nCommons>(seed0, seed1) : (random64Fast<nUncommons>(seed0, seed1) + nCommons);

		if (!isBad[card2]) {
			return false;
		}
	} while (card2 == card1);

	adj = (is_common) ? (adj - 1) : adj;

	//third card
	roll = random64Fast<100>(seed0, seed1) + adj;
	is_common = (roll >= 40);

	uint16 card3;
	do {
		card3 = (is_common) ? random64Fast<nCommons>(seed0, seed1) : (random64Fast<nUncommons>(seed0, seed1) + nCommons);

		if (!isBad[card3]) {
			return false;
		}
	} while ((card3 == card1) || (card3 == card2));

	adj = (is_common) ? (adj - 1) : adj;

	random64Fast<2>(seed0, seed1);
	random64Fast<2>(seed0, seed1);
	random64Fast<2>(seed0, seed1);

	for (uint8 i = 2; i <= nCardRewards; i++) {

		//first card
		roll = random64Fast<100>(seed0, seed1) + adj;
		bool is_rare = (roll < 3);
		is_common = (roll >= 40);
		bool is_uncommon = (!is_rare) && (!is_common);

		uint16 card1 = (is_common) ? random64Fast<nCommons>(seed0, seed1) : random64Fast<lcmOfUncommonsAndRares>(seed0, seed1);
		card1 = (is_rare) ? ((card1 % nRares) + nCommonsAndUncommons) : card1;
		card1 = (is_uncommon) ? ((card1 % nUncommons) + nCommons) : card1;

		if (!isBad[card1]) {
			return false;
		}

		bool foundrare = is_rare;
		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;

		//second card
		roll = random64Fast<100>(seed0, seed1) + adj;
		is_rare = (roll < 3);
		is_common = (roll >= 40);
		is_uncommon = (!is_rare) && (!is_common);

		card2;
		do {
			card2 = (is_common) ? random64Fast<nCommons>(seed0, seed1) : random64Fast<lcmOfUncommonsAndRares>(seed0, seed1);
			card2 = (is_rare) ? ((card2 % nRares) + nCommonsAndUncommons) : card2;
			card2 = (is_uncommon) ? ((card2 % nUncommons) + nCommons) : card2;

			if (!isBad[card2]) {
				return false;
			}
		} while (card2 == card1);

		foundrare |= (is_rare);
		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;

		//third card
		roll = random64Fast<100>(seed0, seed1) + adj;
		is_rare = (roll < 3);
		is_common = (roll >= 40);
		is_uncommon = (!is_rare) && (!is_common);

		card3;
		do {
			card3 = (is_common) ? random64Fast<nCommons>(seed0, seed1) : random64Fast<lcmOfUncommonsAndRares>(seed0, seed1);
			card3 = (is_rare) ? ((card3 % nRares) + nCommonsAndUncommons) : card3;
			card3 = (is_uncommon) ? ((card3 % nUncommons) + nCommons) : card3;

			if (!isBad[card3]) {
				return false;
			}
		} while ((card3 == card1) || (card3 == card2));

		foundrare |= (is_rare);
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
__forceinline__ __device__ bool testBadSilentCardsFast(const uint64 seed) {

	/*
		constexpr uint64 BAD_COMMON = 24804;
		constexpr uint64 BAD_UNCOMMON = 30903371908;
		constexpr uint64 BAD_RARE = 36556;
	*/

	constexpr bool S_BAD_CARDS[71] = { false,false,false,false,false,false,true,false,false,false,false,false,false,true,true,true,false,true,false,false,true,true,true,false,false,false,true,true,false,false,false,false,true,false,false,false,false,false,false,false,false,false,false,false,true,false,false,true,true,true,false,false,false,false,false,false,true,true,false,false,false,true,true,false,false,false,false,false,true,false,false };

	constexpr uint8 S_NUM_A = 19;
	constexpr uint8 S_NUM_B = 33;
	constexpr uint8 S_NUM_C = 19;

	constexpr uint8 S_SUM_AB = 52;
	constexpr uint16 S_LCM_BC = 627;
	//constexpr uint16 NUM_LCM = 11305;

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	int8 adj = 5;

	for (uint8 i = 1; i <= nCardRewards; i++) {

		//first card
		int8 roll = random64Fast<100>(seed0, seed1) + adj;
		bool is_rare = (roll < 3);
		bool is_common = (roll >= 40);
		bool is_uncommon = (!is_rare) && (!is_common);

		uint16 card1 = (is_common) ? random64Fast<S_NUM_A>(seed0, seed1) : random64Fast<S_LCM_BC>(seed0, seed1);
		card1 = (is_rare) ? ((card1 % S_NUM_C) + S_SUM_AB) : card1;
		card1 = (is_uncommon) ? ((card1 % S_NUM_B) + S_NUM_A) : card1;

		if (!S_BAD_CARDS[card1]) {
			return false;
		}

		bool foundrare = is_rare;
		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;

		//second card
		roll = random64Fast<100>(seed0, seed1) + adj;
		is_rare = (roll < 3);
		is_common = (roll >= 40);
		is_uncommon = (!is_rare) && (!is_common);

		uint16 card2;
		do {
			card2 = (is_common) ? random64Fast<S_NUM_A>(seed0, seed1) : random64Fast<S_LCM_BC>(seed0, seed1);
			card2 = (is_rare) ? ((card2 % S_NUM_C) + S_SUM_AB) : card2;
			card2 = (is_uncommon) ? ((card2 % S_NUM_B) + S_NUM_A) : card2;

			if (!S_BAD_CARDS[card2]) {
				return false;
			}
		} while (card2 == card1);

		foundrare |= (is_rare);
		adj = (is_common) ? (adj - 1) : adj;
		adj = (is_rare) ? 5 : adj;

		//third card
		roll = random64Fast<100>(seed0, seed1) + adj;
		is_rare = (roll < 3);
		is_common = (roll >= 40);
		is_uncommon = (!is_rare) && (!is_common);

		uint16 card3;
		do {
			card3 = (is_common) ? random64Fast<S_NUM_A>(seed0, seed1) : random64Fast<S_LCM_BC>(seed0, seed1);
			card3 = (is_rare) ? ((card3 % S_NUM_C) + S_SUM_AB) : card3;
			card3 = (is_uncommon) ? ((card3 % S_NUM_B) + S_NUM_A) : card3;

			if (!S_BAD_CARDS[card3]) {
				return false;
			}
		} while ((card3 == card1) || (card3 == card2));

		foundrare |= (is_rare);
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
