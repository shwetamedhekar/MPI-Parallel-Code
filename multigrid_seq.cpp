/*
@Description: 	This is the sequential code to Solve 2D Discrete Poisson Equations Using Multigrid method
				To Complie: mpiCC multigrid_seq.cpp -o m1
				To Run: mpirun -np 1 m1
@Author: 	Shweta Medhekar
@Created: 	Jan 2007
*/

#include <stdio.h>
#include <iostream>
#include <math.h>
#include "mpi.h"

#define SIZE 33   //Size of grid i.e 2^m+1
#define MAX_GRIDS 15
#define NPRE 1
#define NPOST 1

using namespace std;

double ** multi(double ** u, int size, int no_cycle);
double ** Assign_mem(int x, int y);
void Delete_mem(double ** matrix);
double ** restrict(double ** uc, double ** uf, int nc);
double ** sol_coarsest(double ** su, double ** rhs);
double ** interpolate(double ** uf, double ** uc, int nf);
double ** copy(double ** a_out, double ** a_in, int n);
double ** relax(double ** u, double ** rhs, int n);
double ** residue(double ** res, double ** u, double ** rhs, int n);
double ** addint (double ** uf, double ** uc, double ** res, int nf);



//--------------------------Start function Assign_mem----------------------------------//
double ** Assign_mem(int x, int y)
{
    int i,j;
    
    double * temp_mem = new double[x*y];
    double ** mem = new double*[x];
    for(i=0;i<x;i++)
        mem[i]=&temp_mem[i*y];

    return mem;
}
//----------------------------End function Assign_mem-----------------------------------//


//---------------------------Start function Delete_mem--------------------------------//
void Delete_mem(double ** matrix)
{
    delete [] matrix[0];
    delete [] matrix;
}
//--------------------------End function Delete_mem----------------------------------//


//---------------------------Start function restrict---------------------------------//
double ** restrict(double ** uc, double ** uf, int nc)//make grid coarser
{
    int ic,iif,jc,jf;
    
	for (jf=2,jc=1;jc<(nc-1);jc++,jf+=2)
    {
		for (iif=2,ic=1;ic<(nc-1);ic++,iif+=2)
        {
			uc[ic][jc]=0.5*uf[iif][jf]+0.125*(uf[iif+1][jf]+uf[iif-1][jf]+uf[iif][jf+1]+uf[iif][jf-1]);
		}
	}
    
	for (jc=0,ic=0;ic<nc;ic++,jc+=2)
    {
		uc[ic][0]=uf[jc][0];
		uc[ic][nc-1]=uf[jc][2*nc - 2];
	}
    
    for (jc=0,ic=0;ic<nc;ic++,jc+=2)
    {
		uc[0][ic]=uf[0][jc];
		uc[nc-1][ic]=uf[2*nc - 2][jc];
	}

    return uc;
}

//--------------------End function restrict--------------------------------------//

//-------------------Start function sol_coarsest---------------------------------------//
double ** sol_coarsest(double ** su, double ** rhs)
    //gives the solution of the coarsest matrix i.e 3x3
{
    double h=0.5;
    int i, j;

    for(i=0; i<3; i++)
    {
        for(j=0; j<3; j++)
        {
            su[i][j]=0.0;
        }
    }
    su[2][2] = -h * h * rhs[2][2] / 4.0;

    return su;
}
//-----------------------End function sol_coarsest----------------------------------------//    

//------------------------Start function interpolate-------------------------------//
double ** interpolate(double ** uf, double ** uc, int nf)//bilinear interpolation
{
    int ic,iif,jc,jf,nc;

    nc = nf/2 + 1;
	for (jc=0, jf=0; jc<nc; jc++, jf++)//mapping coarse to fine
    {
		 for (ic=0; ic<nc; ic++)
             {
                 uf[2*ic][2*jf]=uc[ic][jc];
             }
    }
    
	for (jf=0; jf<nf; jf+=2)//even cols--interpolating vertically
    {
		for (iif=1; iif<(nf-1); iif+=2)
        {
			uf[iif][jf]=0.5*(uf[iif+1][jf]+uf[iif-1][jf]);
        }
    }

	for (jf=1;jf<(nf-1); jf+=2)//odd cols--interpolating horizontally
    {
		for (iif=0; iif<nf; iif++)
        {
			uf[iif][jf]=0.5*(uf[iif][jf+1]+uf[iif][jf-1]);
        }
    }

    return uf;
}

//------------------------End function interpolate---------------------------------//


//----------------------Start function copy--------------------------------------//
double ** copy(double ** a_out, double ** a_in, int n)
{
    int i, j;

    for(i=0; i<n; i++)
    {
        for(j=0; j<n; j++)
        {
            a_out[i][j] = a_in[i][j];
        }
    }
    return a_out;
}
//------------------------End function copy--------------------------------------//

//----------------------Start relaxation function------------------------------//
double ** relax(double ** u, double ** rhs, int n)//Gauss-siedal relaxation
{
    int i,ipass,isw,j,jsw=1;
	double h,h2;

	h=1.0/(n-1);
    //h=1.0/5;
    h2=h*h;

    for (ipass=0; ipass<2; ipass++, jsw=3-jsw)//red & black sweeps
    {
		isw=jsw;
		for (j=1; j<(n-1); j++, isw=3-isw)
        {
			for (i=isw; i<n-1; i+=2)
            {
				u[i][j]=0.25*(u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1]-h2*rhs[i][j]);
            }
        }
    }
    return u;
}
//---------------------End relaxation function--------------------------------//

//--------------------Start function residue----------------------------------//
double ** residue(double ** res, double ** u, double ** rhs, int n)
{
    int i,j;
	double h,h2;

    //h=1.0/5;    
	h=1.0/(n-1);
	h2=1.0/(h*h);
    
	for (j=1; j<(n-1); j++)//interior points
    {
        for (i=1; i<(n-1); i++)
        {
            res[i][j] = -h2*(u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1]-4.0*u[i][j])+rhs[i][j];
        }
    }
    
	for (i=0; i<n; i++)//boundary points
    {
        res[i][0]=0.0;
        res[i][n-1]=0.0;
        res[0][i]=0.0;
        res[n-1][i]=0.0;
    }
    
    return res;
}
//---------------------End function residue-----------------------------------//

//-------------------------Start function addint------------------------------//
double ** addint (double ** uf, double ** uc, double ** res, int nf)
{
    double ** interpolate(double ** uf, double ** uc, int nf);
	int i,j;

	interpolate(res, uc, nf);
    
	for (j=0; j<nf; j++)
    {
		for (i=0; i<nf; i++)
        {
			uf[i][j] += res[i][j];
        }
    }
    
    return uf;
}
//--------------------------End function addint-------------------------------//



//---------------------------Start function multi--------------------------------//
double ** multi(double ** u, int size, int no_cycle)
{
    int n, no_of_grids=0, grid_no, i, j, cycle, nf, jj, jpre, jpost, x, y;
    //initializing the 3D pointers
    double **ires[MAX_GRIDS+1], **irho[MAX_GRIDS+1], **irhs[MAX_GRIDS+1], **iu[MAX_GRIDS+1];//why +1

    n=size;
    while(n >>= 1)//depending on the given size of field finding out
                  //how many grid levels will be needed
    {
        no_of_grids++;
    }
    cout<<"no_of_grids: "<<no_of_grids<<endl;
    
    //------------------Error checking on the input values-------------------------------//
    if(size != 1+(1L << no_of_grids))//now shift left no_of _grids times
    {
        cout<<"Error: (n-1) must be a power of 2"<<endl;
        exit(1);
    }

    if(no_of_grids > MAX_GRIDS)
    {
        cout<<"Error: Increase MAX_GRIDS"<<endl;
        exit(1);
    }
    //----------------------------End of error checking-------------------------------------//

    n = size/2 + 1;
    grid_no = no_of_grids - 1;

    irho[grid_no] = Assign_mem(n, n);//topmost grid
    restrict(irho[grid_no], u, n);//call to make grid coarser

    while (n>3)
    {
        n = n/2 +1;
        irho[--grid_no] = Assign_mem(n, n);
        restrict(irho[grid_no], irho[grid_no + 1], n);
    }//All the grids are now created and assigned memory
 
    n = 3;
    iu[1] = Assign_mem(n, n);
    irhs[1] = Assign_mem(n, n);
   
    sol_coarsest(iu[1], irho[1]);
    Delete_mem(irho[1]);

    grid_no = no_of_grids;

    for(j=2; j<=grid_no; j++)
    {
        n = 2*n - 1;
        iu[j] = Assign_mem(n, n);
        irhs[j] = Assign_mem(n, n);
        ires[j] = Assign_mem(n, n);
        interpolate(iu[j], iu[j-1], n);
        
        copy(irhs[j], (j != grid_no ? irho[j] : u), n);

        for(cycle=0; cycle<no_cycle; cycle++)//2 'V' cycles
        {
            nf = n;
            for(jj=j; jj>=2; jj--)
            {
                for(jpre=0; jpre<NPRE; jpre++)
                {
                    relax(iu[jj], irhs[jj], nf);
                }
                residue(ires[jj], iu[jj], irhs[jj], nf);
                nf = nf/2 + 1;
                restrict(irhs[jj-1], ires[jj], nf);

                for(x=0; x<nf; x++)
                {
                    for(y=0; y<nf; y++)
                    {
                        iu[jj-1][x][y]=0.0;
                    }
                }
            }
            sol_coarsest(iu[1], irhs[1]);
            nf = 3;
            for(jj=2; jj<=j; jj++)
            {
                nf = 2*nf -1;
                addint(iu[jj], iu[jj-1], ires[jj], nf);

                for(jpost=0; jpost<NPOST; jpost++)
                {
                    relax(iu[jj], irhs[jj], nf);
                }
            }
        }
    }
    
    copy(u, iu[grid_no], size);

    for (n=size, j=no_of_grids; j>=2; j--, n=n/2+1)
    {
        Delete_mem(ires[j]);
        Delete_mem(irhs[j]);
        Delete_mem(iu[j]);
        if(j != no_of_grids)
        {
            Delete_mem(irho[j]);
        }
    }

    Delete_mem(irhs[1]);
    Delete_mem(iu[1]);

    return u;
}
    
        
    


//----------------------------End func multi--------------------------------------//

int main( int argc, char ** argv )
{
    int rank, size_p;
    int i,j;
    int size;

    MPI_Status status;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size_p );

    if(size_p==1)
    {
        double time_start=0, time_end=0;

        size=SIZE;
        double * temp = new double[size * size];
        double ** u = new double * [size];
        for(i=0; i<size; i++)
            u[i]=&temp[i*size];

        int cntr = 0;
    
        for(i=0; i<size; i++)
        {
            for(j=0; j<size; j++)
            {            
                u[i][j]=0.0;
            }
        }
        u[int(size/2)][int(size/2)]=2.0;//electric charge at the center

        //------------------------------finished initializing i/p----------------------------------------//
        time_start=MPI_Wtime();//start over all timer
        //--------------------------call to main multigrid function--------------------------------------//
        multi(u,size,2);//why the 2
        //-----------------------------------------------------------------------------------------------//

        delete [] temp;
        delete [] u;

        time_end=MPI_Wtime();//stop over all timer
        cout<<"Time taken: ";
        cout<<(time_end-time_start)<<endl;
    }

    else
    {
        cout<<endl<<"Error: This code is only for 1 processor."<<endl;
    }
    
    MPI_Finalize( );
    return 0;
}

#undef NPRE
#undef NPOST
#undef MAX_GRIDS
#undef SIZE
