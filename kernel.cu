
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdint>
#include <string>
#include <chrono>
#include <iostream>
#include <ctime>
#include <fstream>
#include <memory>

#include "sts.cuh"

static const uint64 ONE_BILLION = 1000000000ULL;
static const uint64 ONE_TRILLION = 1000000000000ULL;


static std::string getString(std::int64_t seed) {
	constexpr auto chars = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";

	std::uint64_t uSeed = static_cast<std::uint64_t>(seed);
	std::string str;

	do {
		int rem = static_cast<int>(uSeed % 35);
		uSeed /= 35;
		str += chars[rem];
	} while (uSeed != 0);

	for (int i = 0; i < str.size() / 2; i++) {
		std::swap(str[i], str[str.size() - 1 - i]);
	}
	return str;
}

std::string getTime() {
	std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::string timeStr(std::ctime(&time));
	timeStr.erase(timeStr.end() - 1, timeStr.end());
	return timeStr;
}

int runPandorasSearch(const unsigned int blocks, const unsigned int threads, const std::uint64_t batchSizeBillion, const std::uint64_t startSeed, const char* filename) {
	uint64 searchCountTotal = static_cast<int64>(batchSizeBillion * 1000000000ULL);
	const unsigned int totalThreads = threads * blocks;
	const uint64 searchCountPerThread = searchCountTotal / totalThreads;

	std::ofstream outStream(filename, std::ios_base::app);

	TestInfo info{};
	info.blocks = blocks;
	info.threads = threads;
	info.start = startSeed;
	info.end = info.start + searchCountTotal;

	cudaError_t cudaStatus;
	std::string time;
	std::unique_ptr<uint64_t[]> results(new uint64_t[totalThreads]);

	uint64 foundThreads = 0;


	while (true) {
		time = getTime();
		//outStream << time << " " << info.start << " " << info.end << " " << std::endl;
		std::cout << time << " " << info.start << " " << info.end << " " << std::endl;

		cudaStatus = testPandoraSeedsWithCuda(info, FunctionType::OOH_BLEH, results.get());
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "testSeedsWithCuda failed!");
			return 1;
		}

		for (int i = 0; i < totalThreads; i++) {
			if (results[i]) {
				/*
				outStream << info.start + i << " " << info.end << " " << totalThreads << " seed = " << results[i] << '\n';
				std::cout << info.start + i << " " << info.end << " " << totalThreads << " seed = " << results[i] << '\n';
				*/
				++foundThreads; 
				
				outStream << results[i] << '\n';
				//std::cout << results[i] << "    " << foundThreads << '\n';
				
			}
		}

		std::cout << '\t' << foundThreads << '\n';
		
		info.start += searchCountTotal;
		info.end += searchCountTotal;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

int resultCountTotal = 0;
double benchmarkPandorasHelper(FunctionType fnc, uint64 startSeed, int blocks = 48, int threads = 512) {

	const uint64 iterSearchSize = 20 * ONE_BILLION;
	const int iters = 5;

	TestInfo info{};
	info.blocks = blocks;
	info.threads = threads;
	info.start = startSeed;
	info.end = info.start + iterSearchSize;

	const int totalThreadCount = info.blocks * info.threads;
	uint64_t* results = new uint64_t[totalThreadCount];

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < 10; ++i) {
		testPandoraSeedsWithCuda(info, fnc, results);

		for (int x = 0; x < totalThreadCount; ++x) {
			if (results[x]) {
				++resultCountTotal;
			}
		}

		info.start = info.end;
		info.end = info.start + iterSearchSize;
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto mydur = std::chrono::duration<double, std::milli>(end - start);

	return mydur.count();
}

void runBenchmark() {
	uint64 start = 2 * ONE_TRILLION;

	benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);
	benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);

	double baseSum = 0;
	double fastSum = 0;
	int runCount = 15;

	for (int i = 0; i < 5; ++i) {

		resultCountTotal = 0;
		baseSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);
		std::cout << resultCountTotal << "\n";

		resultCountTotal = 0;
		fastSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);
		std::cout << resultCountTotal << "\n";

		std::cout << "base " << baseSum << '\n';
		std::cout << "fast " << fastSum << std::endl;
	}

	start = 4 * ONE_TRILLION;
	for (int i = 0; i < 5; ++i) {
		resultCountTotal = 0;
		baseSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);
	}
	std::cout << resultCountTotal << "\n";

	for (int i = 0; i < 5; ++i) {
		resultCountTotal = 0;
		fastSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);
	}
	std::cout << resultCountTotal << "\n";

	start = 6 * ONE_TRILLION;
	for (int i = 0; i < 5; ++i) {
		resultCountTotal = 0;
		baseSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);
		std::cout << resultCountTotal << "\n";

		resultCountTotal = 0;
		fastSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);
		std::cout << resultCountTotal << "\n";

		std::cout << "base " << baseSum << '\n';
		std::cout << "fast " << fastSum << std::endl;
	}

	std::cout << "base avg ms: " << baseSum / runCount << '\n';
	std::cout << "fast avg ms: " << fastSum / runCount << '\n';

	std::cout << resultCountTotal << "\n";
}

void runBenchmark2() {
	uint64 start = 2 * ONE_TRILLION;

	benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);
	benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);

	double baseSum = 0;
	double fastSum = 0;
	int runCount = 15;

	int b = 68;
	int t = 512;

	for (int i = 0; i < 5; ++i) {
		baseSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start, b, t);
		fastSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);

		std::cout << b << " blocks " << t << " threads: " << baseSum << '\n';
		std::cout << "48 blocks  512 threads: " << fastSum << std::endl;
	}

	start = 4 * ONE_TRILLION;
	for (int i = 0; i < 5; ++i) {
		baseSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start, b, t);
	}

	for (int i = 0; i < 5; ++i) {
		fastSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);
	}

	start = 6 * ONE_TRILLION;
	for (int i = 0; i < 5; ++i) {
		baseSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start, b, t);
		fastSum += benchmarkPandorasHelper(FunctionType::PANDORA_71_8, start);

		std::cout << b << " blocks " << t << " threads: " << baseSum << '\n';
		std::cout << "48 blocks  512 threads: " << fastSum << std::endl;
	}

	std::cout << b << " blocks " << t << " threads: " << baseSum / runCount << '\n';
	std::cout << "48 blocks  512 threads: " << fastSum / runCount << std::endl;

	std::cout << resultCountTotal << "\n";
}



int main(int argc, const char* argv[])
{
	//runBenchmark2();
	/*
	TestInfo info;
	info.blocks = 1;
	info.threads = 1;
	//info.start = 320665035453320549ULL;
	info.start = 2522130182473665983ULL;
	info.end = info.start + 1;


	bool isTrue = false;
	testPandoraSeedsWithCuda(info, FunctionType::SDW_9_CARD_FAST, &isTrue);
	std::cout << isTrue << '\n';

	int blocks = 68;
	int threads = 512;
	std::uint64_t batchSizeBillion = 1;
	std::uint64_t start = 2522130180000000000;
	auto filename = "asdf.txt";
	return runPandorasSearch(blocks, threads, batchSizeBillion, start, filename);
	*/

	int blocks = 24;//std::stoi(argv[1]);
	int threads = 128;//std::stoi(argv[2]);
	std::uint64_t batchSizeBillion = 1;//std::stoull(argv[3]);
	std::uint64_t start = 0;//std::stoull(argv[4]);
	auto filename = "out.txt"; //argv[5];

	return runPandorasSearch(blocks, threads, batchSizeBillion, start, filename);

}