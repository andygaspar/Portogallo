g++ -c -fPIC offer_eval.cc -o offer.o
g++ -shared -Wl,-soname,liboffers.so -o liboffers.so offer.o