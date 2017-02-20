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

//
//  benchmarking program
//
int main( int argc, char **argv )
{
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
    vector<vector<vector<int> > > grid2(grid_size, vector<vector<int> >(grid_size, vector<int>(0)));
    
    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        // initialize grid
        //
        for (int i = 0; i < n; i++)
        {
            X = (int)(particles[i].x / cutoff);
            Y = (int)(particles[i].y / cutoff);
            grid2[X][Y].push_back(i);
            
        }
        
#pragma omp parallel private(dmin)
        {
            numthreads = omp_get_num_threads();
            dmin = 1.0;
            navg = 0;
            davg = 0.0;
            
            //
            //  compute all forces
            //
#pragma omp for schedule(dynamic, 3)  reduction (+:navg) reduction(+:davg) collapse(2)
            for (int i = 0; i < grid_size; i++)
                for (int j = 0; j < grid_size; j++)
                    for(int l = 0; l < grid2[i][j].size(); l++)
                    {
                        int current_index = grid2[i][j][l];
                        particles[current_index].ax = particles[current_index].ay = 0;
                        for(int k = 0; k < grid2[i][j].size(); k++)
                        {
                            if(l == k) continue; //current does not interact with itself
                            apply_force(particles[current_index], particles[(grid2[i][j][k])], &dmin, &davg, &navg);
                        }
                        if(j != 0)
                            for(int k = 0; k < grid2[i][j-1].size(); k++)
                                apply_force(particles[current_index], particles[(grid2[i][j-1][k])], &dmin, &davg, &navg);
                        if(j != grid_size-1)
                            for(int k = 0; k < grid2[i][j+1].size(); k++)
                                apply_force(particles[current_index], particles[(grid2[i][j+1][k])], &dmin, &davg, &navg);
                        if(i != 0)
                            for(int k = 0; k < grid2[i-1][j].size(); k++)
                                apply_force(particles[current_index], particles[(grid2[i-1][j][k])], &dmin, &davg, &navg);
                        if(i != grid_size-1)
                            for(int k = 0; k < grid2[i+1][j].size(); k++)
                                apply_force(particles[current_index], particles[(grid2[i+1][j][k])], &dmin, &davg, &navg);
                        if(i != 0 && j != 0)
                            for(int k = 0; k < grid2[i-1][j-1].size(); k++)
                                apply_force(particles[current_index], particles[(grid2[i-1][j-1][k])], &dmin, &davg, &navg);
                        if(i != 0 && j != grid_size-1)
                            for(int k = 0; k < grid2[i-1][j+1].size(); k++)
                                apply_force(particles[current_index], particles[(grid2[i-1][j+1][k])], &dmin, &davg, &navg);
                        if(i != grid_size-1 && j != 0)
                            for(int k = 0; k < grid2[i+1][j-1].size(); k++)
                                apply_force(particles[current_index], particles[(grid2[i+1][j-1][k])], &dmin, &davg, &navg);
                        if(i != grid_size-1 && j != grid_size-1)
                            for(int k = 0; k < grid2[i+1][j+1].size(); k++)
                                apply_force(particles[current_index], particles[(grid2[i+1][j+1][k])], &dmin, &davg, &navg);
                    } // end l
            
            //
            //  move particles
            //
#pragma omp for
            for( int i = 0; i < n; i++ )
                move( particles[i] );
            
            //
            // remove particles from grid
            //
#pragma omp for collapse(2)
            for (int i = 0; i< grid_size; i++)
                for (int j = 0; j < grid_size; j++)
                    grid2[i][j].clear();
            /*
             for(int k = 0; k < grid2[i][j].size(); k++)
             {
             int curr = grid2[i][j][k];
             X = (int)(particles[curr].x / cutoff);
             Y = (int)(particles[curr].y / cutoff);
             if(X != i || Y != j)
             {
             grid[i][j].erase(grid2[i][j].begin() + k);
             grid2[X][Y].push_back(curr);
             }
             }
             */
            
            if( find_option( argc, argv, "-no" ) == -1 )
            {
                //
                //  compute statistical data
                //
#pragma omp master
                if (navg)
                {
                    absavg += davg/navg;
                    nabsavg++;
                }
                
#pragma omp critical
                if (dmin < absmin)
                    absmin = dmin;
                
                //
                //  save if necessary
                //
#pragma omp master
                if( fsave && (step%SAVEFREQ) == 0 )
                    save( fsave, n, particles );
            } // end if
        } // end parallel
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
