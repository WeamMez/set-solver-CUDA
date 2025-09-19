This is a simple solver for The game "Set".

you can give it a file with the cards' description (as in the example files) and it prints ONE set option.

If there is no set option, it prints
> "No set found!".
> 
The calculation is parallelized using CUDA.

If you don't have CUDA libraries installed, this will not work.

Please compile with nvcc, this file will not work if you compile it with regular C++ compilers.
