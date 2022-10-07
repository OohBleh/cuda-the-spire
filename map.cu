#include "rng.cuh"

// ************************************************************** BEGIN Fast Map Functions

constexpr int8 searchLength = 5;

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

	uint8 uniqueFloor6Node = 8;

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

/*
accurate recreate of the gCA function from C++ code
UNOPTIMIZED
*/
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

/*
mostly accurate recreate of the cPPLR function from C++ code
mostly optimized
inaccuracy comes from the small chance that the RNG loop condition is met
*/
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

/*

*/
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

/*
mostly accurate recreate of the cNP function from C++ code
mostly optimized
inaccuracy comes from the small chance that the RNG loop condition is met
*/
__forceinline__ __device__ int8 chooseNewPath(Map& map, uint64& seed0, uint64& seed1, int8 curX, int8 curY) {
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

/*
cPI generates a path using the current state of the RNG, calling previous methods
*/
__forceinline__ __device__ void createPathsIteration(Map& map, uint64& seed0, uint64& seed1, int8 startX) {
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

__forceinline__ __device__ void createSinglePathTestFirstIteration(Map& map, uint64& seed0, uint64& seed1, int8 startX) {
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

__forceinline__ __device__ bool createSinglePathTestIteration(Map& map, uint64& seed0, uint64& seed1, int8 startX) {
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

__forceinline__ __device__ bool createSinglePathTestIterationFinal(Map& map, uint64& seed0, uint64& seed1, int8 startX) {
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

__forceinline__ __device__ bool floor6Bottleneck(uint64 seed) {
	uint64 seed0 = murmurHash3(seed + 1);
	uint64 seed1 = murmurHash3(seed0);
	Map map;
	
	int8 startX = random8Fast<7>(seed0, seed1);
	int8 curX = startX;

	for (int8 curY = 0; curY < 15 - 1; ++curY) {
		int8 newX = chooseNewPath(map, seed0, seed1, curX, curY);
		map.getNode(curX, curY).addEdge(newX);
		map.getNode(newX, curY + 1).addParent(curX);

		if (curY == 4) {
			map.uniqueFloor6Node = newX;
		}

		curX = newX;
	}

	curX = random8Fast<7>(seed0, seed1);
	while (curX == startX) {
		curX = random8Fast<7>(seed0, seed1);
	}
	for (int8 curY = 0; curY < 15 - 1; ++curY) {
		int8 newX = chooseNewPath(map, seed0, seed1, curX, curY);
		map.getNode(curX, curY).addEdge(newX);
		map.getNode(newX, curY + 1).addParent(curX);

		if (curY == 4) {
			if (map.uniqueFloor6Node != newX) {
				return false;
			}
		}

		curX = newX;
	}
	
	for (int i = 0; i < 4; i++) {
		int8 curX = random8Fast<7>(seed0, seed1);
		for (int8 curY = 0; curY < 15 - 1; ++curY) {
			int8 newX = chooseNewPath(map, seed0, seed1, curX, curY);
			map.getNode(curX, curY).addEdge(newX);
			map.getNode(newX, curY + 1).addParent(curX);

			if (curY == 4) {
				if (map.uniqueFloor6Node != newX) {
					return false;
				}
			}

			curX = newX;
		}

	}

	return true;
}
