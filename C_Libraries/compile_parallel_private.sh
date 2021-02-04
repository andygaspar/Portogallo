g++  -c -fopenmp -fPIC offer_eval_parallel_2.cc -o offer_parallel_2.o
g++ -shared -fopenmp -Wl,-soname,liboffers_parallel_2.so -o liboffers_parallel_2.so offer_parallel_2.o