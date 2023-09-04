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
#include<math.h>
#include<sys/time.h>

/* Headers for the MPI assignment versions */
#include<mpi.h>

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

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i,j,k;

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    MPI_Barrier(MPI_COMM_WORLD);
    double ttotal = cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */
    
    // Everyone gets a part of the array to work with
    // Storm is read by everyone, do not need to send it around
    // Need to send information to neighbors    


    // Split the layer_size into chunks based on the number of processes
    // Need halo cells on either side
    // Only one side if you are at the end

    int layer_local;

    if (rank == 0) {
       layer_local = layer_size / size + 1;
    }
    else if (rank == size - 1) {
       // Account for the leftover cells from the integer division
       layer_local = layer_size - ( layer_size / size) * (size - 1) + 1;
    }
    else {
       layer_local = layer_size / size + 2;
    }

    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * layer_local );
    float *layer_copy = (float *)malloc( sizeof(float) * layer_local );
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    for( k=0; k<layer_local; k++ ) layer[k] = 0.0f;
    for( k=0; k<layer_local; k++ ) layer_copy[k] = 0.0f;
    
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {

        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        for( j=0; j<storms[i].size; j++ ) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Get impact position global and local */
            int position = storms[i].posval[j*2];
            int position_local;
                   
            if ( rank > 0){
                // Need to account for the first part only having one halo cell
                position_local = position - rank * (layer_size / size) + 1;
            }
            else {
                position_local = position;
            }

            /* For each cell in the layer */
            for( k=0; k<layer_local; k++ ) {
                /* Update the energy value for the cell */
                update( layer, layer_size, k, position_local, energy );
            }
        }

        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        for( k=0; k<layer_local; k++ ){ 
            layer_copy[k] = layer[k];
            fflush(stdout);
        }

        // HERE WE NEED TO COMMUNICATE THE HALOS



        // Everyone but the last rank sends the second to last element to the rank above
        // Recieved in the first element of layer_copy
        
        MPI_Request request;
        MPI_Status status;
        int recv_target, send_target;
         
        if (rank > 0){
          
           MPI_Irecv(&layer_copy[0],1,MPI_REAL,rank-1,0,MPI_COMM_WORLD,&request);

        }

        if (rank != size - 1){
          
           MPI_Send(&layer[layer_local-2],1,MPI_REAL,rank+1,0,MPI_COMM_WORLD);

        }
        // Wait for the transfers to finish
        if (rank > 0){ 
            MPI_Wait(&request,&status);
        } 
	
        MPI_Barrier(MPI_COMM_WORLD);


        // Everyone but the first rank sends the second element to the rank below
        // Recieved in the last element of layer_copy



        if (rank != size - 1){

           MPI_Irecv(&layer_copy[layer_local-1],1,MPI_REAL,rank+1,1,MPI_COMM_WORLD,&request);

        }

        if (rank > 0){

           MPI_Send(&layer[1],1,MPI_REAL,rank-1,1,MPI_COMM_WORLD);

        }

         if (rank != size - 1){
            MPI_Status status;
            MPI_Wait(&request,&status);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        for( k=1; k<layer_local-1; k++ )
            layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;

        
        float local_max = 0;
        int local_pos = 0;

        // Find max in each part
        // call MPI_Reduce with MAX_LOC and a struct with the value and index at the same time 
        /* 4.3. Locate the maximum value in the layer, and its position */
        for( k=1; k<layer_local-1; k++ ) {
            /* Check it only if it is a local maximum */
            if (k==1 && rank > 0) {
               if (layer[k] > layer[k+1] ) {
                if ( layer[k] > local_max ) {
                    local_max = layer[k];
                    local_pos = k;
            }
            }
            }
            else{
            if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
                if ( layer[k] > local_max ) {
                    local_max = layer[k];
                    local_pos = k;
                }
            }
            }
        }
        // Have now determined the maximum on each rank, now need the reduction

        struct {
          float val;
          int pos;
        } local_info, global_info;
        int result_rank, result_index;

        local_info.val = local_max;
        local_info.pos = layer_size * rank + local_pos;
        
        // Do reduce to get max to rank 0 for printing
        MPI_Reduce(&local_info,&global_info,1,MPI_FLOAT_INT,MPI_MAXLOC,0,MPI_COMM_WORLD);
        
        result_rank = global_info.pos / layer_size;
        result_index = global_info.pos % layer_size;

        // A whole bunch of logic is needed here
       
        if (rank == 0) { 
           if (result_rank == 0){
               maximum[i] = local_max;
               positions[i] = local_pos;
           }
           else {
               maximum[i] = global_info.val;
               positions[i] = result_index + result_rank * (layer_size / size) - 1;
           }
         }
    }

    /* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
    MPI_Barrier(MPI_COMM_WORLD);
    ttotal = cp_Wtime() - ttotal;

    if ( rank == 0 ) {

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

    }

    /* 8. Free resources */    
    for( i=0; i<argc-2; i++ )
        free( storms[i].posval );

    /* 9. Program ended successfully */
    return 0;
}

