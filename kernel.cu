
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

#include <time.h>
#include <fstream>

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

inline std::string printWithCommas(uint64 seed) {
	std::string temp = std::to_string(seed);
	uint8 nLen = temp.size();
	temp.reserve((nLen - 1) / 3 + nLen);
	for (uint8 i = 1; i <= (nLen - 1) / 3; i++) {
		temp.insert(nLen - i * 3, 1, ',');
	}
	return temp;
}

static constexpr bool DELTA_PRINT = false;

int runPandorasSearch(
	const unsigned int blocks, 
	const unsigned int threads,
	const unsigned int width,
	const std::uint64_t batchSizeBillion,
	const std::uint64_t startSeed, 
	const char* filename, 
	const uint8 verbosity
) {
	//uint64 searchCountTotal = static_cast<int64>(batchSizeBillion * 1000000000ULL);
	uint64 searchCountTotal = static_cast<int64>(batchSizeBillion * ONE_BILLION);
	const unsigned int totalThreads = threads * blocks;
	const uint64 searchCountPerThread = searchCountTotal / totalThreads;

	std::ofstream outStream(filename, std::ios_base::app);

	TestInfo info{};
	info.blocks = blocks;
	info.threads = threads;
	info.width = width;
	info.start = startSeed;
	info.end = info.start + searchCountTotal;

	cudaError_t cudaStatus;
	std::string time;
	std::unique_ptr<uint64_t[]> results(new uint64_t[width * totalThreads]);

	uint64 foundThreads = 0;

	int nPrints = 0;

	while (true) {
		time = getTime();
		//outStream << time << " " << info.start << " " << info.end << " " << std::endl;
		std::cout << time << " " << info.start << " " << info.end << " " << std::endl;

		cudaStatus = testPandoraSeedsWithCuda(info, FunctionType::SILENT_TAS, results.get());
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "testSeedsWithCuda failed!");
			return 1;
		}

		std::uint64_t currSeed = 0;

		for (int i = 0; i < width * totalThreads; i++) {
			if (results[i]) {
				/*
				outStream << info.start + i << " " << info.end << " " << totalThreads << " seed = " << results[i] << '\n';
				std::cout << info.start + i << " " << info.end << " " << totalThreads << " seed = " << results[i] << '\n';
				*/
				++foundThreads; 
				
				if (DELTA_PRINT) {
					outStream << static_cast<std::int64_t>(results[i]) - static_cast<std::int64_t>(currSeed) << '\n';
					currSeed = results[i];
				}
				else {
					outStream << results[i] << '\n';
				}

				if (verbosity && nPrints < 20) {
					std::cout << getString(results[i]) << '\n';
					nPrints++;
				}
			}
		}

		std::cout << '\t' << printWithCommas(foundThreads) << '\n';
		
		info.start += searchCountTotal;
		info.end += searchCountTotal;

		if (verbosity > 1) {
			return 0;
		}
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

	std::ifstream f;
	f.open("OPTIONS.txt");
	std::string PARAM_NAMES[6] = {
		"blocks           ", 
		"threads          ", 
		"width            ", 
		"batchSizeBillions", 
		"startBatch       ", 
		"verbosity        "
	};
	unsigned int PARAMS[6];
	for (int i = 0; i < 6; i++) {
		f >> PARAMS[i];
		std::cout << PARAM_NAMES[i] << "\t" << PARAMS[i] << '\n';
	}
	f.close();
	std::cout << '\n';

	int blocks = PARAMS[0];//24;//std::stoi(argv[1]);
	int threads = PARAMS[1]; //128;//std::stoi(argv[2]);
	int width = PARAMS[2]; //8;
	std::uint64_t batchSizeBillion = PARAMS[3]; // 1;//std::stoull(argv[3]);
	std::uint64_t start = PARAMS[4] * batchSizeBillion * ONE_BILLION; // 0 * batchSizeBillion* ONE_BILLION;//std::stoull(argv[4]);
	//auto filename = "out.txt"; //argv[5];
	
	std::string fName = "results/out-" + std::to_string(time(NULL)) + ".txt";
	const char* filename = fName.c_str(); //argv[5];

	return runPandorasSearch(blocks, threads, width, batchSizeBillion, start, filename, PARAMS[5]);

}