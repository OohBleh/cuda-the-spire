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

template<uint8 nCardRewards>
__forceinline__ __device__ bool testBadWatcherCardsFast(const uint64 seed) {

	//constexpr bool W_BAD_CARDS[71] = { false, false, false, true, false, false, false, false, true, false, false, false, false, false, false, false, true, true, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, true, false, false, true, true, false, false, false, true, true, false, true, true, false, false, false, true, true, false, false, true, true, true, true, true, true, true, false, false, true, false, false, false, true, false, true, false, true, };
	constexpr bool W_BAD_CARDS[71] = { false,false,false,true,false,false,true,false,false,false,true,false,false,false,false,false,false,true,false,false,false,true,false,false,false,false,false,false,true,false,false,false,false,true,false,false,true,true,false,false,false,true,false,false,true,true,false,false,false,true,false,false,false,false,true,true,true,true,false,false,false,false,false,false,false,false,true,false,true,false,false, };

	constexpr uint8 W_NUM_A = 19;
	constexpr uint8 W_NUM_B = 35;
	constexpr uint8 W_NUM_C = 17;

	constexpr uint8 W_SUM_AB = 54;
	constexpr uint16 W_LCM_BC = 595;

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	//int8 adj = 5;


	//first card
	int8 roll = random64Fast<100>(seed0, seed1);
	bool is_common = (roll >= 35);

	uint16 card1 = (is_common) ? random64Fast<W_NUM_A>(seed0, seed1) : (random64Fast<W_NUM_B>(seed0, seed1) + W_NUM_A);

	if (!W_BAD_CARDS[card1]) {
		return false;
	}

	int8 adj = (is_common) ? (4) : (5);

	//second card
	roll = random64Fast<100>(seed0, seed1) + adj;
	is_common = (roll >= 40);

	uint16 card2;
	do {
		card2 = (is_common) ? random64Fast<W_NUM_A>(seed0, seed1) : (random64Fast<W_NUM_B>(seed0, seed1) + W_NUM_A);

		if (!W_BAD_CARDS[card2]) {
			return false;
		}
	} while (card2 == card1);

	adj = (is_common) ? (adj - 1) : adj;

	//third card
	roll = random64Fast<100>(seed0, seed1) + adj;
	is_common = (roll >= 40);

	uint16 card3;
	do {
		card3 = (is_common) ? random64Fast<W_NUM_A>(seed0, seed1) : (random64Fast<W_NUM_B>(seed0, seed1) + W_NUM_A);

		if (!W_BAD_CARDS[card3]) {
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

		uint16 card1 = (is_common) ? random64Fast<W_NUM_A>(seed0, seed1) : random64Fast<W_LCM_BC>(seed0, seed1);
		card1 = (is_rare) ? ((card1 % W_NUM_C) + W_SUM_AB) : card1;
		card1 = (is_uncommon) ? ((card1 % W_NUM_B) + W_NUM_A) : card1;

		if (!W_BAD_CARDS[card1]) {
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
			card2 = (is_common) ? random64Fast<W_NUM_A>(seed0, seed1) : random64Fast<W_LCM_BC>(seed0, seed1);
			card2 = (is_rare) ? ((card2 % W_NUM_C) + W_SUM_AB) : card2;
			card2 = (is_uncommon) ? ((card2 % W_NUM_B) + W_NUM_A) : card2;

			if (!W_BAD_CARDS[card2]) {
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
			card3 = (is_common) ? random64Fast<W_NUM_A>(seed0, seed1) : random64Fast<W_LCM_BC>(seed0, seed1);
			card3 = (is_rare) ? ((card3 % W_NUM_C) + W_SUM_AB) : card3;
			card3 = (is_uncommon) ? ((card3 % W_NUM_B) + W_NUM_A) : card3;

			if (!W_BAD_CARDS[card3]) {
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