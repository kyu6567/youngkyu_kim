#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
//#include "omp.h"
#include <libiomp/omp.h>
#include <vector>

#define cutoff  0.05
#define density 0.0005

using namespace std;

typedef struct _node
{
    particle_t* p;
    struct _node* next;
}node;

typedef struct _list
{
    int count;
    node* head;
}list;

void init(list* lptr) 
{
    // initialize the list
    lptr->count = 0;
    lptr->head = NULL;
}

void insert(list* lptr, particle_t* particle)
{
    // insert particle_t
    node* new_nptr = (node*)malloc(sizeof(node));
    new_nptr->p = particle;
    if (lptr->count == 0)
    {
        new_nptr->next = NULL;
        lptr->head = new_nptr;
    }
    else
    {
        node* tmp = lptr->head;
        for (int i = 0; i < lptr->count - 1; i++)
            tmp = tmp->next;
        new_nptr->next = NULL;
        tmp->next = new_nptr;
    }
    lptr->count++;
}

void del(list* lptr)
{
    // delete the list
    while (lptr->head != NULL)
    {
        node* tmp = lptr->head;
        if (lptr->count == 1)
        {
            lptr->head = NULL;
            free(tmp);
        }
        else
        {
            for (int i = 1; i < lptr->count - 1; i++)
                tmp = tmp->next;
            node* tmp2 = tmp->next;
            tmp->next = NULL;
            free(tmp2);
        }
        lptr->count--;
    }
}


//
//  benchmarking program
//
int main( int argc, char **argv ) {  
    int navg,nabsavg=0,numthreads;
    double dmin, absmin=1.0,davg,absavg=0.0;
    int X, Y;
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    int grid_size = (sqrt( density * n ))/cutoff + 1;

    //
    // generate grid
    //
    // list* grid[grid_size][grid_size];
    // for(int i=0; i<grid_size;i++){
    //     for (int j = 0; j < grid_size; j++){
    //         grid[i][j] = (list*)malloc(sizeof(list));

    //     }
    // }
    vector<vector<vector<int> > > grid2(grid_size, vector<vector<int> >(grid_size, vector<int>(0)));
    for (int i = 0; i < n; i++) {
        X = (int)(particles[i].x / cutoff);
        Y = (int)(particles[i].y / cutoff);
        //insert(grid[X][Y],&particles[i]);
        grid2[X][Y].push_back(i);

    }


    for( int step = 0; step < NSTEPS; step++ )
    {
        // for(int i=0; i<grid_size;i++){
        //     for (int j = 0; j < grid_size; j++){
        //         init(grid[i][j]);
        //     }
        // }
        //
        // add particles to grid
        //
        //
        //  compute forces
        //
        int XX, YY, I, J;
        node * tmp; 
        //#pragma omp for reduction (+:navg) reduction(+:davg)
        // #pragma omp parallel for schedule(dynamic, 30) private(XX,YY, tmp, I, J) shared(particles, grid, grid_size) reduction (+:navg) reduction(+:davg)
        // for(int i = 0; i < n; i++ ){
        //     // if (step == 0){
        //     //     printf("%i %i\n", i, omp_get_thread_num());
        //     // }
        //     particles[i].ax = particles[i].ay = 0;
        //     int XX = (int)(particles[i].x / cutoff);
        //     int YY = (int)(particles[i].y / cutoff);

        //     for(int I=max(XX-1,0);I<=min(XX+1,grid_size-1);I++){
        //         for(int J=max(YY-1,0);J<=min(YY+1,grid_size-1);J++)
        //         {
        //             node * tmp = grid[I][J]->head;
        //             while (tmp != NULL){
        //                 apply_force(particles[i], *(tmp->p), &dmin, &davg, &navg);
        //                 tmp = tmp->next;
        //             }
        //         }
        //     }
        // }

    // #pragma omp parallel private(dmin)
    // {


        #pragma omp parallel private(tmp, I, J, dmin)
        {
            numthreads = omp_get_num_threads();
            dmin = 1.0;
            navg = 0;
            davg = 0.0;
            #pragma omp for schedule(dynamic, 3)  reduction (+:navg) reduction(+:davg) collapse(2)
            for (int i = 0; i < grid_size; i++) {
                for (int j = 0; j < grid_size; j++) {
                    for(int l = 0; l < grid2[i][j].size(); l++) {
                        int current_index = grid2[i][j][l];
                        
                        particle_t current_particle = particles[current_index];

                        for(int k = 0; k < grid2[i][j].size(); k++) {
                            if(l == k) continue; //current does not interact with itself
                            apply_force(current_particle, particles[(grid2[i][j][k])], &dmin, &davg, &navg);
                        }
                        if(j != 0) {
                            for(int k = 0; k < grid2[i][j-1].size(); k++) {
                                apply_force(current_particle, particles[(grid2[i][j-1][k])], &dmin, &davg, &navg);
                            }
                        }
                        if(j != grid_size-1) {
                            for(int k = 0; k < grid2[i][j+1].size(); k++) {
                                apply_force(current_particle, particles[(grid2[i][j+1][k])], &dmin, &davg, &navg);
                            }
                        }
                        if(i != 0) {
                            for(int k = 0; k < grid2[i-1][j].size(); k++) {
                                apply_force(current_particle, particles[(grid2[i-1][j][k])], &dmin, &davg, &navg);
                            }
                        }
                        if(i != 0 && j != 0) {
                            for(int k = 0; k < grid2[i-1][j-1].size(); k++) {
                                apply_force(current_particle, particles[(grid2[i-1][j-1][k])], &dmin, &davg, &navg);
                            }
                        }
                        if(i != 0 && j != grid_size-1) {
                            for(int k = 0; k < grid2[i-1][j+1].size(); k++) {
                                apply_force(current_particle, particles[(grid2[i-1][j+1][k])], &dmin, &davg, &navg);
                            }
                        }
                        if(i != grid_size-1 && j != 0) {
                            for(int k = 0; k < grid2[i+1][j-1].size(); k++) {
                                apply_force(current_particle, particles[(grid2[i+1][j-1][k])], &dmin, &davg, &navg);
                            }
                        }
                        if(i != grid_size-1) {
                            for(int k = 0; k < grid2[i+1][j].size(); k++) {
                                apply_force(current_particle, particles[(grid2[i+1][j][k])], &dmin, &davg, &navg);
                            }
                        }
                        if(i != grid_size-1 && j != grid_size-1) {
                            for(int k = 0; k < grid2[i+1][j+1].size(); k++) {
                                apply_force(current_particle, particles[(grid2[i+1][j+1][k])], &dmin, &davg, &navg);
                            }
                        }
                    }

                    // node* cur = grid[i][j]->head;
                    // while(cur != NULL){
                    //       cur->p->ax = cur->p->ay = 0;
                    //     for(int I=max(i-1,0);I<=min(i+1,grid_size-1);I++){
                    //         for(int J=max(j-1,0);J<=min(j+1,grid_size-1);J++)
                    //         {
                    //             node* tmp = grid[I][J]->head;
                    //             while (tmp != NULL){
                    //                 apply_force(*(cur->p), *(tmp->p), &dmin, &davg, &navg);
                    //                 tmp = tmp->next;
                    //             }
                    //         }
                    //     }

                    //     cur = cur->next; 
                    // }             
                }
            }

            if( find_option( argc, argv, "-no" ) == -1 ) {
                //
                //  compute statistical data
                //
                #pragma omp master
                if (navg) { 
                    absavg += davg/navg;
                    nabsavg++;
                }

                #pragma omp critical
                if (dmin < absmin) {
                    absmin = dmin; 
                }
                  //
                  //  save if necessary
                  //
                #pragma omp master
                if( fsave && (step%SAVEFREQ) == 0 )
                    save( fsave, n, particles );
            }

        }


        
            //printf("%i\n", step);

        // #pragma omp for reduction (+:navg) reduction(+:davg)
        // for( int i = 0; i < n; i++ ){
        //     particles[i].ax = particles[i].ay = 0;
        //     for (int j = 0; j < n; j++ )
        //         apply_force( particles[i], particles[j],&dmin,&davg,&navg);
        // }
        
        //
        // remove particles from grid
        //
        #pragma omp for collapse(2)
        for (int i = 0; i< grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                for(int k = 0; k < grid2[i][j].size(); k++) {
                    int curr = grid2[i][j][k];
                    X = (int)(particles[curr].x / cutoff);
                    Y = (int)(particles[curr].y / cutoff);
                    if(X != i || Y != j) {
                        grid2[X][Y].push_back(curr);
                        grid2[i][j].erase(grid2[i][j].begin() + k);
                    }
                }
            }
        }

        //
        //  move particles
        //
        #pragma omp for
        for( int i = 0; i < n; i++ ) 
            move( particles[i] );
  

    } // end time iteration

    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -The minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");
    
    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
