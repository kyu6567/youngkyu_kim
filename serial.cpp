#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

#define cutoff  0.01
#define density 0.0005

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
int main( int argc, char **argv )
{    
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;
    int X,Y;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000);

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );
    
    //
    // compute grid size
    //
    int grid_size = (sqrt( density * n ))/cutoff+1;

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    
    //
    // generate grid
    //
    list* grid[grid_size][grid_size];
    for(int i=0; i<grid_size;i++)
        for (int j = 0; j < grid_size; j++)
            grid[i][j] = (list*)malloc(sizeof(list));
	
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        davg = 0.0;
        dmin = 1.0;
        
        //
        // initialize grid
        //
        for(int i=0; i<grid_size;i++)
            for (int j = 0; j < grid_size; j++)
                init(grid[i][j]);
        
	//
	// add particles to grid
	//
	for (int i = 0; i < n; i++)
	{
		X = (int)(particles[i].x / 0.01);
		Y = (int)(particles[i].y / 0.01);
		insert(grid[X][Y],&particles[i]);
	}

	//
        //  compute forces
        //
        for( int i = 0; i < n; i++ )
	{
            	particles[i].ax = particles[i].ay = 0;
		X = (int)(particles[i].x / 0.01);
		Y = (int)(particles[i].y / 0.01);

		for(int I=max(X-1,0);I<=min(X+1,grid_size-1);I++)
			for(int J=max(Y-1,0);J<=min(Y+1,grid_size-1);J++)
			{
				node* tmp = grid[I][J]->head;
				while (tmp != NULL)
				{
					apply_force(particles[i], *(tmp->p), &dmin, &davg, &navg);
					tmp = tmp->next;
				}
			}
	}

	//
	// remove particles from grid
	//
	for (int i = 0; i< grid_size; i++)
		for (int j = 0; j < grid_size; j++)
			del(grid[i][j]);
       
	//
        //  move particles
        //
        for( int i = 0; i < n; i++ ) 
            move( particles[i] );		

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          //
          // Computing statistical data
          //
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;
		
          //
          //  save if necessary
          //
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

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
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
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
