
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

	uint64 uSeed = static_cast<uint64>(seed);
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

int runSeedSearch(TestInfo info, const char* filename
) {
	std::ofstream outStream(filename, std::ios_base::app);

	cudaError_t cudaStatus;
	string time;

	auto totalThreads = info.blocks * info.threads;

	std::unique_ptr<uint64[]> results(new uint64[info.width * totalThreads]);

	uint64 foundThreads = 0;

	int nPrints = 0;

	while (true) {
		time = getTime();
		cout << time << "\t" << printWithCommas(info.start) << "\t" << printWithCommas(info.end) << " " << endl;

		cudaStatus = testSeedsWithCuda(info, results.get());
		if (cudaStatus != cudaSuccess) {
			cout << "testSeedsWithCuda failed!";
			return 1;
		}

		uint64 currSeed = 0;

		for (int i = 0; i < info.width * totalThreads; i++) {
			if (results[i]) {
				++foundThreads; 
				
				outStream << results[i] << endl;

				if (info.verbosity == 1 || (info.verbosity == 2 && nPrints < 20)) {
					cout << getString(results[i]) << endl;
					nPrints++;
				}
			}
		}

		cout << '\t' << printWithCommas(foundThreads) << " seeds found so far..." << endl;
		
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
		cout << PARAM_NAMES[i] << "\t" << PARAMS[i] << endl;
	}
	f.close();

	cout << endl;

	TestInfo info = TestInfo(PARAMS, FunctionType::CUSTOM); // FunctionType::BAD_WATCHER);
	string fName = "results/out-" + std::to_string(time(NULL)) + ".txt";
	return runSeedSearch(info, fName.c_str());

}