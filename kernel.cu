
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdint>
#include <string>
using std::string;

#include <array>
using std::array;

#include <chrono>
#include <iostream>
using std::cout;
using std::endl;

#include <ctime>
#include <fstream>
#include <memory>

#include <time.h>
#include <fstream>

#include "sts.cuh"



static string getString(int64_t seed) {
	constexpr auto chars = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";

	uint64_t uSeed = static_cast<std::uint64_t>(seed);
	string str;

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

string getTime() {
	std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	string timeStr(std::ctime(&time));
	timeStr.erase(timeStr.end() - 1, timeStr.end());
	return timeStr;
}

inline string printWithCommas(uint64 seed) {
	string temp = std::to_string(seed);
	uint8 nLen = temp.size();
	temp.reserve((nLen - 1) / 3 + nLen);
	for (uint8 i = 1; i <= (nLen - 1) / 3; i++) {
		temp.insert(nLen - i * 3, 1, ',');
	}
	return temp;
}

static constexpr bool DELTA_PRINT = false;

int runPandorasSearch(
	TestInfo info,
	const char* filename
) {
	std::ofstream outStream(filename, std::ios_base::app);

	cudaError_t cudaStatus;
	string time;

	auto totalThreads = info.blocks * info.threads;

	std::unique_ptr<uint64_t[]> results(new uint64_t[info.width * totalThreads]);

	uint64 foundThreads = 0;

	int nPrints = 0;

	while (true) {
		time = getTime();
		//outStream << time << " " << info.start << " " << info.end << " " << std::endl;
		cout << time << " " << info.start << " " << info.end << " " << std::endl;

		cudaStatus = testPandoraSeedsWithCuda(info, FunctionType::BOTTLENECK, results.get());
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "testSeedsWithCuda failed!");
			return 1;
		}

		std::uint64_t currSeed = 0;

		for (int i = 0; i < info.width * totalThreads; i++) {
			if (results[i]) {
				++foundThreads; 
				
				if (DELTA_PRINT) {
					outStream << static_cast<std::int64_t>(results[i]) - static_cast<std::int64_t>(currSeed) << endl;
					currSeed = results[i];
				}
				else {
					outStream << results[i] << endl;
				}

				if (info.verbosity && nPrints < 20) {
					std::cout << getString(results[i]) << endl;
					nPrints++;
				}
			}
		}

		std::cout << '\t' << printWithCommas(foundThreads) << endl;
		
		info.incrementInterval();

		if (info.verbosity > 1) {
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

int main(int argc, const char* argv[])
{
	std::ifstream f;
	f.open("OPTIONS.txt");
	const string PARAM_NAMES[6] = {
		"blocks             ", 
		"threads            ", 
		"width              ", 
		"batchSizeInBillions", 
		"startBatch         ", 
		"verbosity          "
	};
	array<unsigned int, 6> PARAMS;
	for (int i = 0; i < 6; i++) {
		f >> PARAMS[i];
		std::cout << PARAM_NAMES[i] << "\t" << PARAMS[i] << endl;
	}
	f.close();

	cout << endl;

	TestInfo info = TestInfo(PARAMS);
	
	//int blocks = PARAMS[0];//24;//std::stoi(argv[1]);
	//int threads = PARAMS[1]; //128;//std::stoi(argv[2]);
	//int width = PARAMS[2]; //8;
	//std::uint64_t batchSizeBillion = PARAMS[3]; // 1;//std::stoull(argv[3]);
	//std::uint64_t start = PARAMS[4] * batchSizeBillion * ONE_BILLION; // 0 * batchSizeBillion* ONE_BILLION;//std::stoull(argv[4]);
	//auto filename = "out.txt"; //argv[5];
	
	string fName = "results/out-" + std::to_string(time(NULL)) + ".txt";
	const char* filename = fName.c_str(); //argv[5];

	return runPandorasSearch(info, filename);

}