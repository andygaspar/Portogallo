#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/resource.h>
#include <sys/time.h>
#include <omp.h>
#include <vector>
#include <time.h>
#include <cmath>







#define DEFAULT_NUM_INTS 10000000
#define DEFAULT_NUM_ITERS 50
#define DEFAULT_NUM_THREADS 16

#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
		  (double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec + \
		     (double)myts.tv_nsec * 1e-9)
#else
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
		   (double)ts.tv_nsec * 1e-9)
#endif


int numthreads;

using std::cout;
using std::endl;












// Auxiliar functions **************
void print(long int* v, int N);
bool equal(long int * v, long int * w,int N);





void serial_sum(long int *v, int N){
  for(int i=1; i<N; i++) v[i]+=v[i-1];
}






void parallel_prefix(long int *nums,long int N)
{
  

  #pragma omp parallel shared(nums,N)
  {
    
    int BLOCK_SIZE= N/(numthreads+1);
    int thnum=omp_get_thread_num();
    int offset=thnum*BLOCK_SIZE;
    int partial=0;
    int i;
    
    //First partial prefix sum **************************
    if(thnum!=0) {
      for(i=offset ; i < offset + BLOCK_SIZE ; i++)   partial+=nums[i]; 
      nums[i-1]=partial;//cumulative partial sum is stored in last elem of the block
      
    }
    else{
      
      for(i=1 ; i <  BLOCK_SIZE ; i++)   nums[i]+=nums[i-1]; //prefix sum of first block
    }

  
    #pragma omp barrier




    int index=BLOCK_SIZE;  //this variable allows to avoid multiplications to compute indeces in 
                           //next omp single operations


    
    #pragma omp single  //prefix sum of just all last elems in each block
    {

      for (i=2; i <= numthreads   ; i++ ){
        nums[index+BLOCK_SIZE-1]+=nums[index-1];
        index+=BLOCK_SIZE;
      }

    }
    
    
    //Second partial prefix sum ******
    offset+=BLOCK_SIZE; //every thread shifts ahead of one block

    if(thnum<numthreads-1) {
        
      for(i=offset ; i < offset + BLOCK_SIZE -1; i++)   nums[i]+=nums[i-1];
    }
    else { //last thread take care of possibile residual as well

      for(i=offset ; i < N; i++)   nums[i]+=nums[i-1];

    }

  } 

}





int main(int argc, char *argv[]) {


  struct  timespec ts;

  int numints = DEFAULT_NUM_INTS;
  numthreads = atoi(argv[1]);



  omp_set_num_threads(numthreads);


  numints=atoi(argv[2]);

  long int * prefix_sums=new long int[numints];
  long int * naive_sums=new long int[numints];


  /* array init 
        this init has been deliberately not parallelised in order to simulate data coming from
        previous computations
  */
  for(int i = 0; i < numints; i++)
  {
    int num = rand();
    prefix_sums[i] = 1;
    naive_sums[i] = 1;
  }


  
  

  double par_time= CPU_TIME;
  parallel_prefix(prefix_sums,numints);
  double par_time_end= CPU_TIME;
  par_time=par_time_end-par_time;


  double ser_time=CPU_TIME;
  serial_sum(naive_sums,numints);
  double ser_time_end=CPU_TIME;
  ser_time=ser_time_end-ser_time;



  cout<<"s "<<ser_time<<"   p "<<par_time<<"  ";
  cout<<equal(prefix_sums,naive_sums,numints)<<endl;


  return 0;
}





















//Auxiliar fun implementation +**************************

void print(long int* v, int N){
  for(int i=0; i<N; i++) cout<<v[i]<<" ";
  cout<<endl;
} 

bool equal(long int * v, long int * w,int N){
  for(int i=0; i<N; i++) {
    if (v[i]!=w[i]) return false;
  }
  return true;
}

