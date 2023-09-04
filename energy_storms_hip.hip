/*
 * Simplified simulation of high-energy particle storms
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2017/2018
 *
 * Version: 2.0
 *
 * Code prepared to be used with the Tablon on-line judge.
 * The current Parallel Computing course includes contests using:
 * OpenMP, MPI, and CUDA.
 *
 * (c) 2018 Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<sys/time.h>

/* Headers for the HIP assignment versions */
#include<hip/hip_runtime.h>

/*
 * Macros to show errors when calling a HIP library function,
 * or after launching a kernel
 */
#define CHECK_HIP_CALL( a )	{ \
	hipError_t ok = a; \
	if ( ok != hipSuccess ) \
		fprintf(stderr, "-- Error HIP call in line %d: %s\n", __LINE__, hipGetErrorString( ok ) ); \
	}
#define CHECK_HIP_LAST()	{ \
	hipError_t ok = hipGetLastError(); \
	if ( ok != hipSuccess ) \
		fprintf(stderr, "-- Error HIP last in line %d: %s\n", __LINE__, hipGetErrorString( ok ) ); \
	}


/* Use fopen function in local tests. The Tablon online judge software 
   substitutes it by a different function to run in its sandbox */
#ifdef CP_TABLON
#include "cputilstablon.h"
#else
#define    cp_open_file(name) fopen(name,"r")
#endif

/* Function to get wall time */
double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}


#define THRESHOLD    0.001f

/* Structure used to store data for one storm of particles */
typedef struct {
    int size;    // Number of particles
    int *posval; // Positions and values
} Storm;

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
void update( float *layer, int layer_size, int k, int pos, float energy ) {
    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = pos - k;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[k] = layer[k] + energy_k;
}


/* ANCILLARY FUNCTIONS: These are not called from the code section which is measured, leave untouched */
/* DEBUG function: Prints the layer status */
void debug_print(int layer_size, float *layer, int *positions, float *maximum, int num_storms ) {
    int i,k;
    /* Only print for array size up to 35 (change it for bigger sizes if needed) */
    if ( layer_size <= 35 ) {
        /* Traverse layer */
        for( k=0; k<layer_size; k++ ) {
            /* Print the energy value of the current cell */
            printf("%10.4f |", layer[k] );

            /* Compute the number of characters. 
               This number is normalized, the maximum level is depicted with 60 characters */
            int ticks = (int)( 60 * layer[k] / maximum[num_storms-1] );

            /* Print all characters except the last one */
            for (i=0; i<ticks-1; i++ ) printf("o");

            /* If the cell is a local maximum print a special trailing character */
            if ( k>0 && k<layer_size-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
                printf("x");
            else
                printf("o");

            /* If the cell is the maximum of any storm, print the storm mark */
            for (i=0; i<num_storms; i++) 
                if ( positions[i] == k ) printf(" M%d", i );

            /* Line feed */
            printf("\n");
        }
    }
}

/*
 * Function: Read data of particle storms from a file
 */
Storm read_storm_file( char *fname ) {
    FILE *fstorm = cp_open_file( fname );
    if ( fstorm == NULL ) {
        fprintf(stderr,"Error: Opening storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    Storm storm;    
    int ok = fscanf(fstorm, "%d", &(storm.size) );
    if ( ok != 1 ) {
        fprintf(stderr,"Error: Reading size of storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    storm.posval = (int *)malloc( sizeof(int) * storm.size * 2 );
    if ( storm.posval == NULL ) {
        fprintf(stderr,"Error: Allocating memory for storm file %s, with size %d\n", fname, storm.size );
        exit( EXIT_FAILURE );
    }
    
    int elem;
    for ( elem=0; elem<storm.size; elem++ ) {
        ok = fscanf(fstorm, "%d %d\n", 
                    &(storm.posval[elem*2]),
                    &(storm.posval[elem*2+1]) );
        if ( ok != 2 ) {
            fprintf(stderr,"Error: Reading element %d in storm file %s\n", elem, fname );
            exit( EXIT_FAILURE );
        }
    }
    fclose( fstorm );

    return storm;
}

__global__ void zero_arrays(float *in1, float *in2, int len) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < len) {
    in1[index] = 0.0f;
    in2[index] = 0.0f;
  }
}

__global__ void update_layer(float *layer, int layer_size, int pos, float energy) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < layer_size) {
     /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = pos - index;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[index] = layer[index] + energy_k;
  }
}

__global__ void swap_layers(float *layer, float* copy, int layer_size){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < layer_size) {
    copy[index] = layer[index];
  }
}


__global__ void layer_update_with_copy(float *layer, float* copy, int layer_size){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > 0 && index < layer_size) {
    layer[index] = (copy[index] + copy[index - 1] + copy[index + 1]) / 3;
  }
}

__global__ void max_serial(float *layer, int layer_size, float *maximum, int *position, int num_storm){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index==0){
      maximum[num_storm] = 0.0f;
      position[num_storm] = 0;
      for( int k=1; k<layer_size-1; k++ ) {
          printf("layer[%d] = %f\n",k,layer[k]);
         /* Check it only if it is a local maximum */
          if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
             if ( layer[k] > maximum[num_storm] ) {
                 maximum[num_storm] = layer[k];
                 position[num_storm] = k;
              }
          }
      }
  }
}

__global__ void max_parallel(float *layer, int layer_size, float *maximum, int *position, int num_storm){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float sharedMax;
    __shared__ int sharedPos;

    if (threadIdx.x == 0){
       sharedMax = 0.0f;
       sharedPos = 0;
    } 

    __syncthreads();

    float localMax = 0.f;
    int localMaxIdx = 0;
    
     for (int i = threadIdx.x; i < layer_size && i> 1; i += blockDim.x)
    {
        float val = layer[i];
  
        if (localMax < abs(val) && val > layer[i-1] && val < layer[i-1])
        {
            localMax = abs(val);
            localMaxIdx = i;
        }
    }
    
    atomicMax(&sharedMax,localMax);

     __syncthreads();
  
    if (sharedMax == localMax)
    {
        sharedPos = localMaxIdx;
    }

     __syncthreads();


    if (index==0){
      maximum[num_storm] = sharedMax;
      position[num_storm] = sharedPos;
    }
}


/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i,j,k;

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    Storm storms[ num_storms ];

    /* 1.2. Read storms information */
    for( i=2; i<argc; i++ ) 
        storms[i-2] = read_storm_file( argv[i] );

    /* 1.3. Intialize maximum levels to zero */
    float maximum[ num_storms ];
    int positions[ num_storms ];
    for (i=0; i<num_storms; i++) {
        maximum[i] = 0.0f;
        positions[i] = 0;
    }

    /* 2. Begin time measurement */
	CHECK_HIP_CALL( hipSetDevice(0) );
	CHECK_HIP_CALL( hipDeviceSynchronize() );
    double ttotal = cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */

    /* 3. Allocate memory for the layer and initialize to zero */
    float *device_layer;
    float *device_copy;
    float *device_max;
    int *device_pos;
      
    CHECK_HIP_CALL(hipMalloc((void **)&device_layer,layer_size*sizeof(float)));
    CHECK_HIP_CALL(hipMalloc((void **)&device_copy,layer_size*sizeof(float)));
    CHECK_HIP_CALL(hipMalloc((void **)&device_max,num_storms*sizeof(float)));
    CHECK_HIP_CALL(hipMalloc((void **)&device_pos,num_storms*sizeof(int)));

    CHECK_HIP_CALL(hipMemcpy(device_max, maximum, num_storms * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_CALL(hipMemcpy(device_pos, positions, num_storms * sizeof(int), hipMemcpyHostToDevice));

    dim3 blockDim(32);
    dim3 gridDim(ceil(((float)layer_size) / ((float)blockDim.x)));

    //Launch kernel for zeroing the arrays
    zero_arrays<<<gridDim, blockDim>>>(device_layer, device_copy, layer_size);
    
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {

        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        for( j=0; j<storms[i].size; j++ ) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Get impact position */
            int position = storms[i].posval[j*2];
             
            update_layer<<<gridDim, blockDim>>>(device_layer,layer_size,position,energy);

            }

        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        swap_layers<<<gridDim, blockDim>>>(device_layer,device_copy,layer_size);

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */

        layer_update_with_copy<<<gridDim,blockDim>>>(device_layer,device_copy,layer_size);

        /* 4.3. Locate the maximum value in the layer, and its position */
    
        max_parallel<<<gridDim,blockDim>>>(device_layer,layer_size, device_max, device_pos,i);

    }

    CHECK_HIP_CALL(hipMemcpy(maximum, device_max, num_storms * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP_CALL(hipMemcpy(positions, device_pos, num_storms * sizeof(int), hipMemcpyDeviceToHost));

    /* END: Do NOT optimize/parallelize the code below this point */
    //Free the device memory
    hipFree(device_layer);
    hipFree(device_copy);    
    hipFree(device_max);
    hipFree(device_pos);

    /* 5. End time measurement */
	CHECK_HIP_CALL( hipDeviceSynchronize() );
    ttotal = cp_Wtime() - ttotal;

    /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
    #ifdef DEBUG
    debug_print( layer_size, layer, positions, maximum, num_storms );
    #endif

    /* 7. Results output, used by the Tablon online judge software */
    printf("\n");
    /* 7.1. Total computation time */
    printf("Time: %lf\n", ttotal );
    /* 7.2. Print the maximum levels */
    printf("Result:");
    for (i=0; i<num_storms; i++)
        printf(" %d %f", positions[i], maximum[i] );
    printf("\n");

    /* 8. Free resources */    
    for( i=0; i<argc-2; i++ )
        free( storms[i].posval );

    /* 9. Program ended successfully */
    return 0;
}

