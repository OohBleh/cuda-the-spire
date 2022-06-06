#include "sts.cuh"
#pragma once

// ************************************************************** BEGIN LIBGDX Functions

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

// ************************************************************** EMD LIBGDX Functions


// ************************************************************** BEGIN JAVA Functions

static constexpr uint64 JAVA_MULTIPLIER = 0x5DEECE66DULL;
static constexpr uint64 JAVA_ADDEND = 0xBULL;
static constexpr uint64 JAVA_MASK = (1ULL << 48) - 1;

__forceinline__ __device__ void javaScramble(uint64& javaSeed) {
	javaSeed = (javaSeed ^ JAVA_MULTIPLIER) & JAVA_MASK;
}

template<uint8 bits>
__forceinline__ __device__ int32 javaNext(uint64& javaSeed) {
	javaSeed = (javaSeed * JAVA_MULTIPLIER + JAVA_ADDEND) & JAVA_MASK;
	return static_cast<int32>(javaSeed >> (48 - bits));
}

__forceinline__ __device__ uint8 javaInt8(uint64& javaSeed, const uint8 bound) {
	int32 r = javaNext<31>(javaSeed);
	uint8 m = bound - 1;
	if ((bound & m) == 0)  // i.e., bound is a power of 2
		r = static_cast<int32>(((bound * static_cast<uint64>(r)) >> 31));
	else {
		for (int32 u = r; u - (r = u % bound) + m < 0;) {
			u = javaNext<31>(javaSeed);
		}
	}
	return static_cast<uint8>(r);
}

// ************************************************************** END JAVA Functions


// ************************************************************** END JAVA Functions

template<uint8 poolSize, uint8 target>
__forceinline__ __device__ bool javaInt8(uint64& javaSeed, const uint8 bound) {
	

	javaScramble(javaSeed);
	uint8 targetPos = target;
	
	for (uint8 i = poolSize; i > 1; i--) {
		uint8 k = javaInt8(javaSeed, i);
		if (k == targetPos) {
			return false;
		}
		if (i - 1 == targetPos) {
			targetPos = k;
		}
	}

	return targetPos == 0;
}

// ************************************************************** END JAVA Functions
