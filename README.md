# cuda-the-spire
 Using CUDA to sieve Slay the Spire seeds for interesting properties. 

Based on code written by gamerpuppy.  

To run, install CUDA and Windows Visual Studio and open the solutions file... hopefully!  

after opening in VSC, adjust OPTIONS.txt to have the correct values for: 

blocks: adjust based on your GPU and performance benchmarks

threads: adjust based on your GPU and performance benchmarks 

width: the maximum number of seeds that a device can pass to the results array

batchSizeBillions: the size of each batch of seeds, in billions

startBatch: the index of the first batch for your current search

verbosity: 0 for limited printing to std::out, 1 to print the first 20 seeds, 2 to print 20 seeds and stop after 1 batch