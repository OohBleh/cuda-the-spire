
#include "sts.cuh"
#include <stdio.h>

// ************************************************************** BEGIN RNG Functions

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

template<uint8 n>
__forceinline__ __device__ uint8 random8Fast(uint64& seed0, uint64& seed1) {
	uint8 value;
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

__forceinline__ __device__ uint64 randomPreFloatFast(uint64& seed0, uint64& seed1) {

	uint64 s1 = seed0;
	uint64 s0 = seed1;
	seed0 = s0;
	s1 ^= s1 << 23;
	seed1 = s1 ^ s0 ^ s1 >> 17 ^ s0 >> 26;

	uint64 x = (seed0 + seed1) >> 40;
	return x;
}

__forceinline__ __device__ uint64 randomLong(uint64& seed0, uint64& seed1) {

	uint64 s1 = seed0;
	uint64 s0 = seed1;
	seed0 = s0;
	s1 ^= s1 << 23;
	seed1 = s1 ^ s0 ^ s1 >> 17 ^ s0 >> 26;

	uint64 x = (seed0 + seed1);
	return x;
}

// ************************************************************** EMD RNG Functions

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

// ************************************************************** BEGIN Fast Cards/Neow Functions

__forceinline__ __device__ bool neowsLament(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	random64Fast<6>(seed0, seed1);

	/*
		[ ] THREE_SMALL_POTIONS,
		[ ] RANDOM_COMMON_RELIC,
		[ ] TEN_PERCENT_HP_BONUS,
		[x] THREE_ENEMY_KILL,
		[ ] HUNDRED_GOLD,
	*/

	return random64Fast<5>(seed0, seed1) == 3;
}

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

	return true;

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

__forceinline__ __device__ bool testBadNeow2(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	/*
		[ ] 0 THREE_CARDS = 0,
		[ ] 1 ONE_RANDOM_RARE_CARD,
		[x] 2 REMOVE_CARD,
		[ ] 3 UPGRADE_CARD,
		[ ] 4 TRANSFORM_CARD,
		[ ] 5 RANDOM_COLORLESS,
	*/

	if (random64Fast<6>(seed0, seed1) != 2) {
		return false;
	}

	/*
		[ ] THREE_SMALL_POTIONS,
		[x] RANDOM_COMMON_RELIC,
		[x] TEN_PERCENT_HP_BONUS,
		[ ] THREE_ENEMY_KILL,
		[x] HUNDRED_GOLD,
	*/

	static constexpr bool GOOD_NEOW2[5] = {true, false, false, true, false};
	//uint8 k = random64Fast<5>(seed0, seed1);
	//if ((k != 1) && (k != 4)) {
		//return false;
	//}

	if (GOOD_NEOW2[random64Fast<5>(seed0, seed1)]) {
		return false;
	}

	/*										[ ]			[ ]			[x]			[x]
										0 HP_LOSS	1 NO_GOLD	2 CURSE		3 DAMAGE
		[ ] RANDOM_COLORLESS_2,				0			0			0			0
		[ ] REMOVE_TWO,						1			1			=====		1
		[ ] ONE_RARE_RELIC,					2			2			1			2
		[ ] THREE_RARE_CARDS,				3			3			2			3
		[x] TWO_FIFTY_GOLD,					[4]			=====		[3]			[4]
		[ ] TRANSFORM_TWO_CARDS,			5			4			4			5
		[x] TWENTY_PERCENT_HP_BONUS,		=====		[5]			[5]			[6]

		k = 0 (l ~ Unif(6)): false, false, false, false, true, false,
			{false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, }
		k = 1 (l ~ Unif(6)): false, false, false, false, false, true, 
			{false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, }
		k = 2 (l ~ Unif(6)): false, false, false, true, false, true, 
			{false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, }
		k = 3 (l ~ Unif(7)): false, false, false, false, true, false, true, 
			{false, false, false, false, true, false, true, false, false, false, false, true, false, true, false, false, false, false, true, false, true, false, false, false, false, true, false, true, false, false, false, false, true, false, true, false, false, false, false, true, false, true, }
	*/
	
	/*
	uint8 k = random64Fast<4>(seed0, seed1);
	uint8 l = (k == 3) ? random64Fast<7>(seed0, seed1) : random64Fast<6>(seed0, seed1);
	l = (k == 1) ? (l - 1) : l;
	l = (k == 2) ? (l + 1) : l;
	return (l == 4) || (l == 6);
	*/

	
	static constexpr bool BAD_NEOW4[4][42] = {
		{false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, },
		{false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, true, },
		{false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, false, false, false, true, false, true, },
		{false, false, false, false, true, false, true, false, false, false, false, true, false, true, false, false, false, false, true, false, true, false, false, false, false, true, false, true, false, false, false, false, true, false, true, false, false, false, false, true, false, true, }
	};
	//uint8 k = random64Fast<4>(seed0, seed1);
	//uint8 l = random64Fast<42>(seed0, seed1);

	return BAD_NEOW4[
		random8Fast<4>(seed0, seed1)
	][
		random8Fast<42>(seed0, seed1)
	];
	
}

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


// ************************************************************** END Fast Cards/Neow Functions

// ************************************************************** BEGIN Fast Map Functions

struct MapNode {
	int8 parentCount;
	int8 parents[6];

	int8 edgeCount;
	int8 edges[3];

	__forceinline__ __device__ MapNode() {
		parentCount = 0;
		edgeCount = 0;
	}

	__forceinline__ __device__ void addParent(int8 parent) {
		parents[parentCount++] = parent;
	}

	__forceinline__ __device__ void addEdge(int8 edge) {
		int8 cur = 0;
		while (true) {
			if (cur == edgeCount) {
				edges[cur] = edge;
				++edgeCount;
				return;
			}

			if (edge == edges[cur]) {
				return;
			}

			if (edge < edges[cur]) {
				for (int8 x = edgeCount; x > cur; --x) {
					edges[x] = edges[x - 1];
				}
				edges[cur] = edge;
				++edgeCount;
				return;
			}
			++cur;
		}
	}

	__forceinline__ __device__ int8 getMaxEdge() const {
		return edges[edgeCount - 1];
	}


	__forceinline__ __device__ int8 getMinEdge() const {
		return edges[0];
	}

	__forceinline__ __device__ int8 getMaxXParent() const {
		int8 max = parents[0];
		for (int8 i = 1; i < parentCount; ++i) {
			if (parents[i] > max) {
				max = parents[i];
			}
		}
		return max;
	}


	__forceinline__ __device__ int8 getMinXParent() const {
		int8 min = parents[0];
		for (int8 i = 1; i < parentCount; ++i) {
			if (parents[i] < min) {
				min = parents[i];
			}
		}
		return min;
	}

};

struct Map {
	MapNode nodes[15][7];

	__forceinline__ __device__ MapNode& getNode(int8 x, int8 y) {
		return nodes[y][x];
	}
	__forceinline__ __device__ const MapNode& getNode(int8 x, int8 y) const {
		return nodes[y][x];
	}

	__forceinline__ __device__ Map() {

		for (int8 r = 0; r < 15; ++r) {
			for (int8 c = 0; c < 7; ++c) {
				nodes[r][c].edgeCount = 0;
				nodes[r][c].parentCount = 0;
			}
		}
	}
};

__forceinline__ __device__ bool getCommonAncestor(const Map& map, int8 x1, int8 x2, int8 y) {
	if (map.getNode(x1, y).parentCount == 0 || map.getNode(x2, y).parentCount == 0) {
		return false;
	}

	int8 l_node;
	int8 r_node;
	if (x1 < y) {
		l_node = x1;
		r_node = x2;
	}
	else {
		l_node = x2;
		r_node = x1;
	}

	int8 leftX = map.getNode(l_node, y).getMaxXParent();
	if (leftX == map.getNode(r_node, y).getMinXParent()) {
		return true;
	}
	return false;
}

__forceinline__ __device__ int8 choosePathParentLoopRandomizer(const Map& map, uint64& seed0, uint64& seed1, int8 curX, int8 curY, int8 newX) {
	const MapNode& newEdgeDest = map.getNode(newX, curY + 1);

	for (int8 i = 0; i < newEdgeDest.parentCount; i++) {
		int8 parentX = newEdgeDest.parents[i];
		if (curX == parentX) {
			continue;
		}
		if (!getCommonAncestor(map, parentX, curX, curY)) {
			continue;
		}
		
		/*
			if (newX > curX) {
				//newX = curX + rng.randRange8(-1, 0);
				newX = curX + random8Fast<2>(seed0, seed1) - 1;
				if (newX < 0) {
					newX = curX;
				}
			}
			else if (newX == curX) {
				//newX = curX + rng.randRange8(-1, 1);
				newX = curX + random8Fast<3>(seed0, seed1) - 1;
				if (newX > 6) {
					newX = curX - 1;
				}
				else if (newX < 0) {
					newX = curX + 1;
				}
			}
			else {
				//newX = curX + rng.randRange8(0, 1);
				newX = curX + random8Fast<2>(seed0, seed1);
				if (newX > 6) {
					newX = curX;
				}
			}
		*/

		static constexpr int8_t cPPLR[7][7][6] = { 1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,1,2,1,2,0,1,2,0,1,2,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3,2,3,2,3,1,2,3,1,2,3,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,2,3,4,2,3,4,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,3,4,5,3,4,5,3,4,3,4,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,4,5,6,4,5,6,4,5,4,5,4,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,6,5,5,6,5 };
		newX = cPPLR[curX][newX][random8Fast<6>(seed0, seed1)];

	}

	return newX;
}

__forceinline__ __device__ int choosePathAdjustNewX(const Map& map, int8 curX, int8 curY, int8 newEdgeX) {
	if (curX != 0) {
		auto right_node = map.getNode(curX - 1, curY);
		if (right_node.edgeCount > 0) {
			int8 left_edge_of_right_node = right_node.getMaxEdge();
			if (left_edge_of_right_node > newEdgeX) {
				newEdgeX = left_edge_of_right_node;
			}
		}
	}

	if (curX < 6) {
		auto right_node = map.getNode(curX + 1, curY);
		if (right_node.edgeCount > 0) {
			int8 left_edge_of_right_node = right_node.getMinEdge();
			if (left_edge_of_right_node < newEdgeX) {
				newEdgeX = left_edge_of_right_node;
			}
		}
	}
	return newEdgeX;
}

__device__ int8 chooseNewPath(Map& map, uint64& seed0, uint64& seed1, int8 curX, int8 curY) {
	MapNode& currentNode = map.getNode(curX, curY);
	/*
	int8 min;
	int8 max;
	if (curX == 0) {
		min = 0;
		max = 1;
	}
	else if (curX == 6) {
		min = -1;
		max = 0;
	}
	else {
		min = -1;
		max = 1;
	}
	int8 newEdgeX = curX + rng.randRange8(min, max);
	*/

	static constexpr int8_t NEXT[7][6] = { 0,1,0,1,0,1,0,1,2,0,1,2,1,2,3,1,2,3,2,3,4,2,3,4,3,4,5,3,4,5,4,5,6,4,5,6,5,6,5,6,5,6 };
	
	int8 newEdgeX = NEXT[curX][random8Fast<6>(seed0, seed1)];
	newEdgeX = choosePathParentLoopRandomizer(map, seed0, seed1, curX, curY, newEdgeX);
	newEdgeX = choosePathAdjustNewX(map, curX, curY, newEdgeX);

	return newEdgeX;
}

__device__ void createPathsIteration(Map& map, uint64& seed0, uint64& seed1, int8 startX) {
	int8 curX = startX;
	for (int8 curY = 0; curY < 15 - 1; ++curY) {
		int8 newX = chooseNewPath(map, seed0, seed1, curX, curY);
		map.getNode(curX, curY).addEdge(newX);
		map.getNode(newX, curY + 1).addParent(curX);
		curX = newX;
	}
}

__forceinline__ __device__ int8 chooseNewPathFirstTest(Map& map, uint64& seed0, uint64& seed1, int8 curX, int8 curY) {
	MapNode& currentNode = map.getNode(curX, curY);

	/*
	int8 min;
	int8 max;
	if (curX == 0) {
		min = 0;
		max = 1;
	}
	else if (curX == 6) {
		min = -1;
		max = 0;
	}
	else {
		min = -1;
		max = 1;
	}
	
	return curX + rng.randRange8(min, max);
	*/

	static constexpr int8_t NEXT[7][6] = { 0,1,0,1,0,1,0,1,2,0,1,2,1,2,3,1,2,3,2,3,4,2,3,4,3,4,5,3,4,5,4,5,6,4,5,6,5,6,5,6,5,6 };
	return NEXT[curX][random8Fast<6>(seed0, seed1)];

	
}

__device__ void createSinglePathTestFirstIteration(Map& map, uint64& seed0, uint64& seed1, int8 startX) {
	int8 curX = startX;
	for (int8 curY = 0; curY < 15 - 1; ++curY) {
		int8 newX = chooseNewPathFirstTest(map, seed0, seed1, curX, curY);
		auto& nextNode = map.getNode(newX, curY + 1);
		map.getNode(curX, curY).addEdge(newX);
		nextNode.addParent(curX);
		curX = newX;
	}
	map.getNode(curX, 14).addEdge(3);
}

__device__ bool createSinglePathTestIteration(Map& map, uint64& seed0, uint64& seed1, int8 startX) {
	int8 curX = startX;
	for (int8 curY = 0; curY < 15 - 1; ++curY) {
		int8 newX = chooseNewPath(map, seed0, seed1, curX, curY);

		auto& nextNode = map.getNode(newX, curY + 1);
		if (curY < searchLength && nextNode.parentCount == 0) {
			return false;
		}

		map.getNode(curX, curY).addEdge(newX);
		map.getNode(newX, curY + 1).addParent(curX);
		curX = newX;
	}
	map.getNode(curX, 14).addEdge(3);
	return true;
}

__device__ bool createSinglePathTestIterationFinal(Map& map, uint64& seed0, uint64& seed1, int8 startX) {
	int8 curX = startX;
	for (int8 curY = 0; curY < searchLength; ++curY) {
		int8 newX = chooseNewPath(map, seed0, seed1, curX, curY);

		auto& nextNode = map.getNode(newX, curY + 1);
		if (nextNode.parentCount == 0) {
			return false;
		}

		map.getNode(curX, curY).addEdge(newX);
		map.getNode(newX, curY + 1).addParent(curX);
		curX = newX;
	}
	return true;
}

template<uint8 nPaths>
__forceinline__ __device__ bool createPathsSinglePathTest(Map& map, uint64& seed0, uint64& seed1) {
	int8 firstStartX = random8Fast<7>(seed0, seed1);
	createSinglePathTestFirstIteration(map, seed0, seed1, firstStartX);

	for (int8 i = 1; i < nPaths - 1; ++i) {
		int8 startX = random8Fast<7>(seed0, seed1);

		while (startX == firstStartX && i == 1) {
			startX = random8Fast<7>(seed0, seed1);
		}

		bool res = createSinglePathTestIteration(map, seed0, seed1, startX);
		if (!res) {
			return false;
		}
	}

	int8 startX = random8Fast<7>(seed0, seed1);

	while (startX == firstStartX && nPaths == 1) {
		startX = random8Fast<7>(seed0, seed1);
	}

	//bool res = createSinglePathTestIteration(map, seed0, seed1, startX);
	bool res = createSinglePathTestIterationFinal(map, seed0, seed1, startX);
	
	if (!res) {
		return false;
	}
	return true;
}

__forceinline__ __device__ int8 getNewXFirstTest(uint64& seed0, uint64& seed1, int8 firstStartX, int8 curX, int8 curY, int8 correctNewX) {
	
	/*
		int8 min;
		int8 max;
		if (curX == 0) {
			min = 0;
			max = 1;
		}
		else if (curX == 6) {
			min = -1;
			max = 0;
		}
		else {
			min = -1;
			max = 1;
		}

		int8 newX = curX + rng.randRange8(min, max);
	*/
	
		static constexpr int8_t NEXT[7][6] = { 0,1,0,1,0,1,0,1,2,0,1,2,1,2,3,1,2,3,2,3,4,2,3,4,3,4,5,3,4,5,4,5,6,4,5,6,5,6,5,6,5,6 };
		int8 newX = NEXT[curX][random8Fast<6>(seed0, seed1)];

		//if (curY == 0) {
		/*
		if (curX != 0) {
			if (firstStartX == curX - 1) {
				int8 left_edge_of_right_node = correctNewX;
				if (left_edge_of_right_node > newX) {
					newX = left_edge_of_right_node;
				}
			}
		}
		if ((curX != 0) && (firstStartX == curX - 1) && (correctNewX > newX)) {
			newX = correctNewX;
		}
		
		/*
			if (curX < 6) {
				if (firstStartX == curX + 1) {
					int8 left_edge_of_right_node = correctNewX;
					if (left_edge_of_right_node < newX) {
						newX = left_edge_of_right_node;
					}
				}
			}
		//}

		if ((curX < 6) && (firstStartX == curX + 1) && (correctNewX < newX)) {
			newX = correctNewX;
		}

		return newX;
		*/

		static constexpr int8 COMP[7][7] = { 0,1,0,0,0,0,0,2,0,1,0,0,0,0,0,2,0,1,0,0,0,0,0,2,0,1,0,0,0,0,0,2,0,1,0,0,0,0,0,2,0,1,0,0,0,0,0,2,0 };
		static constexpr int8 NEWX[7][3][7] = { 0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,0,0,0,0,0,0,0,1,2,3,4,5,6,1,1,2,3,4,5,6,0,1,1,1,1,1,1,0,1,2,3,4,5,6,2,2,2,3,4,5,6,0,1,2,2,2,2,2,0,1,2,3,4,5,6,3,3,3,3,4,5,6,0,1,2,3,3,3,3,0,1,2,3,4,5,6,4,4,4,4,4,5,6,0,1,2,3,4,4,4,0,1,2,3,4,5,6,5,5,5,5,5,5,6,0,1,2,3,4,5,5,0,1,2,3,4,5,6,6,6,6,6,6,6,6,0,1,2,3,4,5,6 };

		return NEWX[
			correctNewX
		][
			COMP[firstStartX][curX]
		][
			newX
		];
}

__forceinline__ __device__ int8 passesFirstTest(uint64 seed) {
	//Random mapRng(seed + 1);
	
	uint64 seed0 = murmurHash3(seed + 1);
	//uint64 seed0 = murmurHash3(77 + 1);
	uint64 seed1 = murmurHash3(seed0);
	
	int8 firstStartX = random8Fast<7>(seed0, seed1);
	int8 firstNewX;

	int8 results[searchLength];
	//int8 Dlist[15];


	static constexpr bool MIDDLE[7] = { false, false, true, true, true, false, false };
	int8 curX = firstStartX;
	if (MIDDLE[curX]) {
		return false;
	}
	//Dlist[0] = 99;

	static constexpr int8_t NEXT[7][6] = { 0,1,0,1,0,1,0,1,2,0,1,2,1,2,3,1,2,3,2,3,4,2,3,4,3,4,5,3,4,5,4,5,6,4,5,6,5,6,5,6,5,6 };
	//static constexpr bool NOT_WALL[7] = { false, true, true, true, true, true, false };

	int8 curY = 0;
	for (curY = 0; curY < searchLength; ++curY) {
		/*
			int8 min;
			int8 max;
			if (curX == 0) {
				min = 0;
				max = 1;
			}
			else if (curX == 6) {
				min = -1;
				max = 0;
			}
			else {
				min = -1;
				max = 1;
			}
			int8 newX = curX + mapRng.randRange8(min, max);
		*/

		//int8 D = random8Fast<6>(seed0, seed1);
		//Dlist[curY + 1] = D;
		//int8 newX = NEXT[curX][D];

		int8 newX = NEXT[curX][random8Fast<6>(seed0, seed1)];

		if (MIDDLE[newX]) {
			return false;
		}


		results[curY] = newX;
		curX = newX;
	}

	/*
	static constexpr int8 trueResults[15] = {4,4,3,4,4,4,4,4,5,4,3,2,1,2};
	//static constexpr int8 trueD[15] = { 99, 0, 4, 0, 5, 1, 4, 1, 1, 5, 3, 0, 0, 0, 2 };

	for (int8 testY = 0; testY < 3 + 0*searchLength; ++testY) {
		if (results[testY] != trueResults[testY]) {
			return false;
		}

		//if (Dlist[testY] != trueD[testY]) {
			//return false;
		//}
	}
	return true;
	*/
	for (; curY < 14; ++curY) {
		random8Fast<2>(seed0, seed1);
	}

	int8 startX = random8Fast<7>(seed0, seed1);
	while (startX == firstStartX) {
		startX = random8Fast<7>(seed0, seed1);
	}

	curX = startX;
	int8 newX = getNewXFirstTest(seed0, seed1, firstStartX, curX, curY, results[curY]);
	if (newX != results[0]) {
		return false;
	}

	curX = newX;

	for (int8 curY = 1; curY < searchLength; ++curY) {

		newX = NEXT[curX][random8Fast<6>(seed0, seed1)];
		if (newX != results[curY]) {
			return false;
		}

		curX = newX;
	}

	return true;
}

template<uint8 nPaths>
__forceinline__ __device__ int8 testSeedForSinglePath(uint64 seed) {
	if (passesFirstTest(seed)) {
		//Random mapRng(seed + 1);
		uint64 seed0 = murmurHash3(seed + 1);
		uint64 seed1 = murmurHash3(seed0);
		Map map;
		return createPathsSinglePathTest<nPaths>(map, seed0, seed1);
		//return true;
	}
	else {
		return false;
	}
}

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

// ************************************************************** END Fast Map Functions

// ************************************************************** BEGIN ?-Node Functions

__forceinline__ __device__ bool onlyTreasures(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	static constexpr bool BAD_ROLLS[9][100] = { true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,false,false,true,true,true };
	

	for (uint8 i = 0; i < 7; i++) {
		//float p = randomFloatFast(seed0, seed1);
		uint8 roll = static_cast<uint8>(100 * randomFloatFast(seed0, seed1));
		if (BAD_ROLLS[i][roll]) {
			return false;
		}
	}

	return true;
}

__forceinline__ __device__ bool onlyShopsTreasures(const uint64 seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);

	uint8 combatThreshold = 10;
	uint8 shopThreshold = 3;
	uint8 treasureThreshold = 2;

	for (uint8 i = 0; i < 9; i++) {
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

// ************************************************************** END ?-Node Functions

// ************************************************************** BEGIN Custom Mode Functions

__forceinline__ __device__ bool testSneck0andSpecializedRB(const uint64 seed) {

	constexpr uint8 NUM_BR = 143;
	constexpr uint8 SKIM = 42;
	//constexpr uint8 CAI = 140;

	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);
	
	if (random8Fast<NUM_BR>(seed0, seed1) != SKIM) {
		return false;
	}

	uint8 num0 = 0;

	uint64 seed2 = murmurHash3(seed + 1);
	uint64 seed3 = murmurHash3(seed2);
	
	for (uint8 i = 0; i < 7; i++) {

		/*
		if (random8Fast<4>(seed2, seed3) == 0) {
			num0++;
		}
		*/
		if (random8Fast<4>(seed2, seed3)) {
			return false;
		}
	}

	return true;
}

__forceinline__ __device__ bool testSealedRB(const uint64 seed) {

	constexpr uint8 BR_NUM_B = 72;
	static constexpr bool isGood[BR_NUM_B] = { false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, };
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);
	uint8 ctr = 0;
	uint8 ctrs[6] = {};
	static constexpr int8 goodInts[6] = {
		11, 43, 23, 42
	};

	static constexpr uint8 IGNORE = 15;

	for (uint8 i = 0; i < IGNORE; i++) {
		random8Fast<2>(seed0, seed1);
		random8Fast<2>(seed0, seed1);
	}

	for (uint8 i = 0; i < 30 - IGNORE; i++) {

		if (random8Fast<100>(seed0, seed1) > 34) {
			random8Fast<2>(seed0, seed1);
			continue;
		}

		uint8 card = random8Fast<BR_NUM_B>(seed0, seed1);
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
		}

	}

	return (ctrs[0] > 0) && (ctrs[1] > 0) && (ctrs[2] > 0) && (ctrs[3] > 0);
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
	//int32 r = static_cast<int32>(shuffleSeed >> (48 - 31));
	int32 r = static_cast<int32>(shuffleSeed >> 17);
	//static constexpr int bound = 16;
	//r = static_cast<int32>(((bound * static_cast<uint64>(r)) >> 31));
	r = static_cast<int32>(((16 * static_cast<uint64>(r)) >> 31));
	
	return r == 8;
}



__forceinline__ __device__ bool startsPBox(uint64_t seed) {
	uint64 seed0 = murmurHash3(seed);
	uint64 seed1 = murmurHash3(seed0);
	// shuffle commons, uncommons, rares, shops
	randomLong(seed0, seed1);
	randomLong(seed0, seed1);
	randomLong(seed0, seed1);
	randomLong(seed0, seed1);

	uint64 shuffleSeed = randomLong(seed0, seed1);
	seed = (seed ^ JAVA_MULTIPLIER) & JAVA_MASK;

	int size = 22;
	int K = 5;
	for (int i = size; i > 1; i--) {

		seed = (seed * JAVA_MULTIPLIER + JAVA_ADDEND) & JAVA_MASK;
		int r = static_cast<std::int32_t>(seed >> (48 - 31));
		int bound = i;
		int m = bound - 1;
		if ((bound & m) == 0)
			r = static_cast<int32_t>(((bound * static_cast<std::uint64_t>(r)) >> 31));
		else {
			for (int32_t u = r; u - (r = u % bound) + m < 0; ) {
				seed = (seed * JAVA_MULTIPLIER + JAVA_ADDEND) & JAVA_MASK;
				u = static_cast<std::int32_t>(seed >> (48 - 31));
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
	// random64Fast<2>(seed0, seed1);
	
	// rare Shard card roll
	if (random64Fast<84>(seed0, seed1) != 56) {
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
