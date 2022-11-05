#pragma once
#include "rng.cuh"

// ************************************************************** BEGIN Neow Functions

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
__forceinline__ __device__ bool getsBadNeowOptions1(const uint64 seed) {
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

template<SeedType seedType>
__forceinline__ __device__ bool getsBadNeowOptions2(const uint64 seed) {
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

	static constexpr bool GOOD_NEOW2[5] = { true, false, false, true, false };
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
// ************************************************************** END Neow Functions
