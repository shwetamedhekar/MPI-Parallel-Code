/*
@Description: 	This is the parallel code to Solve 2D Discrete Poisson Equations Using Multigrid method
				To Complie: mpiCC multigrid_parallel.cpp -o m
				To Run: mpirun -np <number of processors> m
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
#define NO_OF_CYCLES 2 //2 'V' cycles

using namespace std;

double ** Assign_mem(int x, int y);
void Delete_mem(double ** matrix);
double ** restrict_com(double ** uc, double ** uf, int nc);
double ** restrict(double ** uc, double ** uf, int nc, double ** data, int right, int bottom);
double ** sol_coarsest(double ** su, double ** rhs);
double ** interpolate_com(double ** uf, double ** uc, int nf);
double ** interpolate_ver(double ** uf, double ** uc, int nf, double ** data, int top);
double ** interpolate_hori(double ** uf, int nf, double ** data, int left);
double ** copy(double ** a_out, double ** a_in, int n);
double ** relax_com(double ** u, double ** rhs, int n);
double ** relax(double ** u, double ** rhs, int n, double ** data, int left, int right, int top, int bottom, int s);
double ** residue(double ** res, double ** u, double ** rhs, int n, double ** data, int left, int right, int top, int bottom, int s);
double ** residue_com(double ** res, double ** u, double ** rhs, int n);
double ** addint (double ** u, double ** res, int nf);
double ** addint_com(double ** uf, double ** uc, double ** res, int nf);
        
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


//---------------------------Start function restrict_com---------------------------------//
double ** restrict_com(double ** uc, double ** uf, int nc)//make grid coarser
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
//--------------------End function restrict_com--------------------------------------//


//---------------------------Start function restrict---------------------------------//
double ** restrict(double ** uc, double ** uf, int nc, double ** data, int right, int bottom)//make grid coarser
{
    //    cout<<endl<<"DEBUG: In restrict"<<endl;
    int i, j, nc_r, nc_b;
    int ic,iif,jc,jf;
    
    double * temp = new double[(2*nc + 1) * (2*nc +1)];
    double ** u = new double * [(2*nc + 1)];
    for(i=0; i<(2*nc +1); i++)
        u[i]=&temp[i*(2*nc+1)];

    for(i=0;i<2*nc;i++)
    {
        for(j=0;j<2*nc;j++)
        {
            u[i][j]=uf[i][j];
        }
    }

    for(j=0,i=1; i<(2*nc + 1); i+=2,j++)
    {
        u[i][2*nc] = data[j][0];
        u[2*nc][i] = data[j][1];
    }

    nc_r = nc;
    nc_b = nc;
        
    if(right==1)
    {
        nc_r = nc-1;
        for (jc=1,ic=0;ic<nc;ic++,jc+=2)
        {
            uc[ic][nc-1]=u[jc][2*nc - 1];
        }
    }

    if(bottom==1)
    {
        nc_b = nc-1;
        for (jc=1,ic=0;ic<nc;ic++,jc+=2)
        {
            uc[nc-1][ic]=u[2*nc - 1][jc];
        }
    }

    for (jf=1,jc=0;jc<nc_r;jc++,jf+=2)
    {
		for (iif=1,ic=0;ic<nc_b;ic++,iif+=2)
        {
			uc[ic][jc]=0.5*u[iif][jf]+0.125*(u[iif+1][jf]+u[iif-1][jf]+u[iif][jf+1]+u[iif][jf-1]);
		}
	}

    delete [] temp;
    delete [] u;
    
    return uc;
}
//---------------------------End of function restrict---------------------------------//


//-------------------Start function sol_coarsest---------------------------------------//
double ** sol_coarsest(double ** su, double ** rhs)
    //gives the solution of the coarsest matrix i.e 3x3
{
    //cout<<endl<<"DEBUG: In sol_coarsest"<<endl;
    
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


//------------------------Start function interpolate_com-------------------------------//
double ** interpolate_com(double ** uf, double ** uc, int nf)//bilinear interpolation
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

//------------------------End function interpolate_com---------------------------------//


//------------------------Start func interpolate_ver------------------------------------//
double ** interpolate_ver(double ** uf, double ** uc, int nf, double ** data, int top)
{
    int i,j,ic,iif,jc,jf,nc;

    double * temp = new double[(nf + 1) * (nf +1)];
    double ** u = new double * [(nf + 1)];
    for(i=0; i<(nf +1); i++)
        u[i]=&temp[i*(nf+1)];
    
    for(i=1;i<=nf;i++)
    {
        for(j=1;j<=nf;j++)
        {
            u[i][j]=uf[i-1][j-1];
        }
    }

    for(i=0,j=2; j<=nf; j+=2, i++)
    {
        u[0][j]=data[i][3];
    }
    
    nc = nf/2;
	for (jc=0, jf=2; jc<nc; jc++, jf+=2)//mapping coarse to fine
    {
        for (iif=2, ic=0; ic<nc; ic++, iif+=2)
        {
            u[iif][jf]=uc[ic][jc];
        }
    }
    
 	for (jf=2; jf<=nf; jf+=2)//even cols--interpolating vertically
    {
 		for (iif=3; iif<(nf); iif+=2)
        {
 			u[iif][jf]=0.5*(u[iif+1][jf]+u[iif-1][jf]);
        }
    }

    if(top==1)
    {
        for(jf=2; jf<=nf; jf+=2)
        {
            u[1][jf]=0.5*(u[2][jf]);
        }
    }
    else
    {
        for(jf=2; jf<=nf; jf+=2)
        {
            u[1][jf]=0.5*(u[2][jf] + u[0][jf]);
        }
    }
    
    for(i=1;i<=nf;i++)
    {
        for(j=1;j<=nf;j++)
        {
            uf[i-1][j-1]=u[i][j];
        }
    }

    delete [] temp;
    delete [] u;
    
    return uf;
}

//---------------------------End function interpolate_ver-----------------------------//


//-------------------------Start func interpolate_hori------------------------------//
double ** interpolate_hori(double ** uf, int nf, double ** data, int left)
{
    int i,j,ic,iif,jc,jf,nc;
    nc = nf/2;

    double * temp = new double[(nf + 1) * (nf +1)];
    double ** u = new double * [(nf + 1)];
    for(i=0; i<(nf +1); i++)
        u[i]=&temp[i*(nf+1)];
    
    for(i=1;i<=nf;i++)
    {
        for(j=1;j<=nf;j++)
        {
            u[i][j]=uf[i-1][j-1];
        }
    }

    for(i=0,j=1; j<=nf; j++, i++)
    {
        u[j][0]=data[i][2];
    }
    
 	for (jf=3;jf<(nf); jf+=2)//odd cols--interpolating horizontally
    {
 		for (iif=1; iif<=nf; iif++)
        {
 			u[iif][jf]=0.5*(u[iif][jf+1]+u[iif][jf-1]);
        }
    }

    if(left==1)
    {
        for(jf=1; jf<=nf; jf++)
        {
            u[jf][1]=0.5*(u[jf][2]);
        }
    }
    else
    {
        for(jf=1; jf<=nf; jf++)
        {
            u[jf][1]=0.5*(u[jf][2] + u[jf][0]);
        }
    }

    for(i=1;i<=nf;i++)
    {
        for(j=1;j<=nf;j++)
        {
            uf[i-1][j-1]=u[i][j];
        }
    }

    delete [] temp;
    delete [] u;
    return uf;
}
//-------------------------End func interpolate_hori--------------------------------//


//----------------------Start function copy--------------------------------------//
double ** copy(double ** a_out, double ** a_in, int n)
{
    //cout<<endl<<"DEBUG: In copy"<<endl;
    
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
double ** relax_com(double ** u, double ** rhs, int n)//Gauss-siedal relaxation
{
    int i,ipass,isw,j,jsw=1;
	double h,h2;

	h=1.0/(n-1);
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


//----------------------Start relaxation function------------------------------//
double ** relax(double ** u, double ** rhs, int n, double ** data, int left, int right, int top, int bottom, int s)
{
    //cout<<endl<<"DEBUG: In relax"<<endl;
    
    int i,ipass,isw,j,jsw=1;
	double h,h2;

    h=1.0/(s*n);//sxn
    h2=h*h;

    double * temp = new double[(n + 2) * (n + 2)];
    double ** u1 = new double * [(n + 2)];
    for(i=0; i<(n + 2); i++)
        u1[i]=&temp[i*(n + 2)];

    n=n+2;

    for(i=0; i<n; i++)//initialized to zero
    {
        u1[i][0]=0;
        u1[i][n-1]=0;
        u1[0][i]=0;
        u1[n-1][i]=0;
    }

    for(i=1; i<n-1; i++)
    {
        for(j=1; j<n-1; j++)
        {
            u1[i][j] = u[i-1][j-1];
        }
    }

    if(left==0)
    {
        for(i=1; i<n-1; i++)
        {
            u1[i][0] = data[i-1][2];
        }
    }
    
    if(right==0)
    {
        for(i=1; i<n-1; i++)
        {
            u1[i][n-1] = data[i-1][0];
        }
    }
    
    if(top==0)
    {
        for(i=1; i<n-1; i++)
        {
            u1[0][i] = data[i-1][3];
        }
    }
    
    if(bottom==0)
    {
        for(i=1; i<n-1; i++)
        {
            u1[n-1][i] = data[i-1][1];
        }
    }

    int n_col, n_row;

    if(right==1)//do not calc for dis
    {
        n_col=n-1;
    }
    else
    {
        n_col=n;
    }

    if(bottom == 1)
    {
        n_row=n-1;
    }
    else
    {
        n_row=n;
    }
    
    
    for (ipass=0; ipass<2; ipass++, jsw=3-jsw)//red & black sweeps
    {
		isw=jsw;
		for (j=1; j<(n_col-1); j++, isw=3-isw)
        {
			for (i=isw; i<(n_row-1); i+=2)
            {
				u1[i][j]=0.25*(u1[i+1][j]+u1[i-1][j]+u1[i][j+1]+u1[i][j-1]-h2*rhs[i-1][j-1]);//check dis
            }
        }
    }

    for(i=1; i<n-1; i++)
    {
        for(j=1; j<n-1; j++)
        {
            u[i-1][j-1] = u1[i][j];
        }
    }

    delete [] temp;
    delete [] u1;
    return u;
}
//---------------------End relaxation function--------------------------------//


//--------------------Start function residue_com----------------------------------//
double ** residue_com(double ** res, double ** u, double ** rhs, int n)
{
    int i,j;
	double h,h2;

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
//---------------------End function residue_com-----------------------------------//


//--------------------Start function residue----------------------------------//
double ** residue(double ** res, double ** u, double ** rhs, int n, double ** data, int left, int right, int top, int bottom, int s)
{
    //cout<<endl<<"DEBUG: In residue"<<endl;
    
    int i,j;
	double h,h2;

    h=1.0/(s*n);//sxn
    h2=1.0/(h*h);

    double * temp = new double[(n + 2) * (n + 2)];
    double ** u1 = new double * [(n + 2)];
    for(i=0; i<(n + 2); i++)
        u1[i]=&temp[i*(n + 2)];

    n=n+2;
    for(i=0; i<n; i++)//initialized to zero
    {
        u1[i][0]=0;
        u1[i][n-1]=0;
        u1[0][i]=0;
        u1[n-1][i]=0;
    }

    for(i=1; i<n-1; i++)
    {
        for(j=1; j<n-1; j++)
        {
            u1[i][j] = u[i-1][j-1];
        }
    }

    if(left==0)
    {
        for(i=1; i<n-1; i++)
        {
            u1[i][0] = data[i-1][2];
        }
    }
    
    if(right==0)
    {
        for(i=1; i<n-1; i++)
        {
            u1[i][n-1] = data[i-1][0];
        }
    }
    
    if(top==0)
    {
        for(i=1; i<n-1; i++)
        {
            u1[0][i] = data[i-1][3];
        }
    }
    
    if(bottom==0)
    {
        for(i=1; i<n-1; i++)
        {
            u1[n-1][i] = data[i-1][1];
        }
    }

	for (j=1; j<(n-1); j++)//interior points
    {
        for (i=1; i<(n-1); i++)
        {
            res[i-1][j-1] = -h2*(u1[i+1][j]+u1[i-1][j]+u1[i][j+1]+u1[i][j-1]-4.0*u1[i][j])+rhs[i-1][j-1];
        }
    }
    //boundary
    if(right==1)
    {
        for(i=0; i<(n-2); i++)
        {
            res[i][n-3]=0;
        }
    }

    if(bottom==1)
    {
        for(i=0; i<(n-2); i++)
        {
            res[n-3][i]=0;
        }
    }
    

    return res;
}
//---------------------End function residue-----------------------------------//


//-------------------------Start function addint------------------------------//
double ** addint_com (double ** uf, double ** uc, double ** res, int nf)
{
    double ** interpolate_com(double ** uf, double ** uc, int nf);
	int i,j;

	interpolate_com(res, uc, nf);
    
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


//-------------------------Start function addint------------------------------//
double ** addint (double ** u, double ** res, int nf)
{
    //cout<<endl<<"DEBUG: In addint"<<endl;
    
 	int i,j;
    
	for (j=0; j<nf; j++)
    {
		for (i=0; i<nf; i++)
        {
			u[i][j] += res[i][j];
        }
    }

    return u;
}
//--------------------------End function addint-------------------------------//



int main( int argc, char ** argv )
{
    int rank, size_p, size_each, n, total_no_of_grids=0, n_each, no_of_grids=0, mark=0, no_cycle;
    int i, j, jj, jpre, jpost, x, y;
    int size, left_wall=0, right_wall=0, top_wall=0, bottom_wall=0, tmp_res;
    int s, s1;
    double s_temp, time_start=0, time_end=0;
    int tag=0;
            
    size = SIZE;
    no_cycle = NO_OF_CYCLES;
    
    MPI_Status status;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size_p );

    n=size;
    while(n >>= 1)//depending on the given size of field finding out
        //how many grid levels will be needed
    {
        total_no_of_grids++;
    }//now everyone knows the total number of grids needed
    cout<<"Total no. of grids: "<<total_no_of_grids<<endl;

    s_temp = sqrt(double(size_p));
    s = int(s_temp);//s^2 number of processors
    s1 = s;

    size_each = (size - 1)/s;//each slave will get size_each x size_each

    if(size_each < 1)
    {
        cout<<"Error: Increase the size of input or decrease the number of processors"<<endl;
        exit(1);
    }

    while (s1 >>= 1)
    {
        mark++;
    }

    n_each = size_each;
    while(n_each >>= 1)//depending on the given size of field
        //finding out how many grid levels will be needed
    {
        no_of_grids++;
    }//now everyone knows the number of grids needed for each

    //------------------Error checking on the input values-------------------------------//
    if(size != 1+(1L << total_no_of_grids))//now shift left no_of _grids times
    {
        cout<<"Error: (n-1) must be a power of 2"<<endl;
        exit(1);
    }

    if(s_temp != (1L << mark))//now shift left mark times
    {
        cout<<"Error: sqrt(number of processors) must be a power of 2"<<endl;
        exit(1);
    }

    if(total_no_of_grids > MAX_GRIDS)
    {
        cout<<"Error: Increase MAX_GRIDS"<<endl;
        exit(1);
    }
    //----------------------------End of error checking-------------------------------------//

    //-----------------------------------------Master---------------------------------//
    if(rank==0)//master
    {
        int n_master, grid_no_m, cycle_m, nf_m, cycle_com, nf_com;
        double **irho_m[MAX_GRIDS+1], **irhs_m[MAX_GRIDS+1], **iu_m[MAX_GRIDS+1], **ires_m[MAX_GRIDS+1], **irho_com[MAX_GRIDS+1], **irhs_com[MAX_GRIDS+1], **iu_com[MAX_GRIDS+1], **ires_com[MAX_GRIDS+1];

        time_start=MPI_Wtime();//start over all timer

        top_wall=1;
        left_wall=1;
        
        double * temp = new double[size * size];
        double ** u = new double * [size];
        for(i=0; i<size; i++)
            u[i]=&temp[i*size];

        for(i=0; i<size; i++)
        {
            for(j=0; j<size; j++)
            {
                u[i][j]=0.0;
            }
        }
        u[int(size/2)][int(size/2)]=2.0;//electric charge at the center
    
        //------------------------------finished initializing i/p----------------------------------------//
        
        
        //-------------Sending u to slaves------------------------------------------------//
        
        double * u_send = new double[size_each*size_each];
        int start_row=1, start_col=1, z, k;

        double * temp_work = new double[size_each * size_each];
        double ** u_work = new double * [size_each];
        for(i=0; i<size_each; i++)
            u_work[i]=&temp_work[i*size_each];
        
        //send to each of the slave processors including master
        z=0;
        for(i=0; i<s; i++)
        {
            start_col = 1;
            if(i>0)
            {
                start_row = start_row + size_each;
            }
                        
            for(j=0; j<s; j++)
            {
                k=0;
                if(z==0)//matrix with master filled up
                {
                    for(x=0; x<size_each; x++)
                    {
                        for(y=0; y<size_each; y++)
                        {
                            u_work[x][y] = u[x + start_row][y + start_col];
                        }
                    }
                    z++;
                }
                
                else
                {
                    for(x=0; x<size_each; x++)
                    {
                        for(y=0; y<size_each; y++)
                        {
                            u_send[k] = u[x + start_row][y + start_col];
                            k++;
                        }
                    }
                    MPI_Send (u_send, (size_each*size_each), MPI_DOUBLE, z, tag, MPI_COMM_WORLD);
                    z++;
                }
                start_col = start_col + size_each;
            }
        }
        delete [] u_send;


        //-------------------------------------------multi starts-----------------//
        n_master = size_each/2;
        grid_no_m = total_no_of_grids - 1;

        irho_m[grid_no_m] = Assign_mem(n_master, n_master);

        //get data frm neighbours
        double * m_restrict = new double[size_each];
        double * temp_data = new double[4*(size_each*2)];
        double ** data_m = new double * [(size_each*2)];
        for(i=0; i<size_each*2; i++)
            data_m[i]=&temp_data[i*4];

        for(x=0; x<size_each*2; x++)
        {
            for(y=0; y<4; y++)
            {
                data_m[x][y] = 0;
            }
        }
        
        if(right_wall==0)//frm right neighbour
        {
            MPI_Recv(m_restrict, n_master, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
            for(i=0; i<n_master; i++)
            {
                data_m[i][0] = m_restrict[i];
            }
        }

        if(bottom_wall==0)//frm bottom neighbour
        {
            MPI_Recv(m_restrict, n_master, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
            for(i=0; i<n_master; i++)
            {
                data_m[i][1] = m_restrict[i];
            }
        }
        
        restrict(irho_m[grid_no_m], u_work, n_master, data_m, right_wall, bottom_wall);

        while (n_master > 1)//smallest has to be 1x1
        {
            for(x=0; x<n_master; x++)
            {
                for(y=0; y<4; y++)
                {
                    data_m[x][y] = 0;
                }
            }
            n_master = n_master/2;
            irho_m[--grid_no_m] = Assign_mem(n_master, n_master);

            if(right_wall==0)//frm right neighbour
            {
                MPI_Recv(m_restrict, n_master, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                for(i=0; i<n_master; i++)
                {
                    data_m[i][0] = m_restrict[i];
                }
            }
            
            if(bottom_wall==0)//frm bottom neighbour
            {
                MPI_Recv(m_restrict, n_master, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                for(i=0; i<n_master; i++)
                {
                    data_m[i][1] = m_restrict[i];
                }
            }
            
            restrict(irho_m[grid_no_m], irho_m[grid_no_m + 1], n_master, data_m, right_wall, bottom_wall);
        }

        //--now everyone has only 1 element left---receive frm slaves    
        //----------COMMON REGION BEGINS-------------------
        
        irho_com[grid_no_m] = Assign_mem((s+1), (s+1));
        for(i=0; i<s+1; i++)
            irho_com[grid_no_m][i][0] = 0;
        for(i=0; i<s+1; i++)
            irho_com[grid_no_m][0][i] = 0;
        irho_com[grid_no_m][1][1] = irho_m[grid_no_m][0][0];

        double * one_rcv = new double[1];

        i=1;
        j=2;
        for(z=1; z<size_p; z++)
        {
            MPI_Recv(one_rcv, 1, MPI_DOUBLE, z, tag, MPI_COMM_WORLD, &status);
            irho_com[grid_no_m][i][j] = one_rcv[0];
            if(j>=s)
            {
                i++;
                j=1;
            }
            else
            {
                j++;
            }
        }
        
        //now check if wat u got back is indeed 3x3 or more----this
        //depends on the number of slaves
        int n_com;
        n_com = s+1;
        
        while(grid_no_m > 1)//if no of proc > 4
        {
            n_com = n_com/2 + 1;
            irho_com[--grid_no_m] = Assign_mem(n_com, n_com);
            restrict_com(irho_com[grid_no_m], irho_com[grid_no_m + 1], n_com);
        }

        n_com = 3;
        iu_com[1] = Assign_mem(n_com, n_com);

        sol_coarsest(iu_com[1], irho_com[1]);

        double * iu_send = new double[1];

        //this code is for np = 4
        if(mark==1)
        {
            i=1;
            j=2;
            for(z=1; z<size_p; z++)
            {
                iu_send[0] = iu_com[1][i][j];
                MPI_Send (iu_send, 1, MPI_DOUBLE, z, tag, MPI_COMM_WORLD);
                if(j>=s)
                {
                    i++;
                    j=1;
                }
                else
                {
                    j++;
                }
            }
        }

        grid_no_m = total_no_of_grids;
        if(mark>1)
        {
            n_com = 3;
            irhs_com[1] = Assign_mem(n_com, n_com);

            for(j=2; j<=mark; j++)
            {
                n_com = 2*n_com - 1;
                iu_com[j] = Assign_mem(n_com, n_com);
                irhs_com[j] = Assign_mem(n_com, n_com);
                ires_com[j] = Assign_mem(n_com, n_com);
                interpolate_com(iu_com[j], iu_com[j-1], n_com);

                copy(irhs_com[j], (j != grid_no_m ? irho_com[j] : u_work), n_com);

                for(cycle_com=0; cycle_com<no_cycle; cycle_com++)//2 'V' cycles
                {
                    nf_com = n_com;
                    for(jj=j; jj>=2; jj--)
                    {
                        for(jpre=0; jpre<NPRE; jpre++)
                        {
                            relax_com(iu_com[jj], irhs_com[jj], nf_com);
                        }
                        residue_com(ires_com[jj], iu_com[jj], irhs_com[jj], nf_com);
                        nf_com = nf_com/2 + 1;
                        restrict_com(irhs_com[jj-1], ires_com[jj], nf_com);

                        for(x=0; x<nf_com; x++)
                        {
                            for(y=0; y<nf_com; y++)
                            {
                                iu_com[jj-1][x][y]=0.0;
                            }
                        }
                    }
                    sol_coarsest(iu_com[1], irhs_com[1]);
                    nf_com = 3;
                    for(jj=2; jj<=j; jj++)
                    {
                        nf_com = 2*nf_com -1;
                        addint_com(iu_com[jj], iu_com[jj-1], ires_com[jj], nf_com);

                        for(jpost=0; jpost<NPOST; jpost++)
                        {
                            relax_com(iu_com[jj], irhs_com[jj], nf_com);
                        }
                    }
                }
            }
            
            i=1;
            j=2;
            for(z=1; z<size_p; z++)
            {
                iu_send[0] = iu_com[mark][i][j];
                MPI_Send (iu_send, 1, MPI_DOUBLE, z, tag, MPI_COMM_WORLD);
            
                if(j>=s)
                {
                    i++;
                    j=1;
                }
                else
                {
                    j++;
                }
            }
        }
        
        //---------------COMMON ENDS-------------------------------------//
        iu_m[mark] = Assign_mem(1,1);
        irhs_m[mark] = Assign_mem(1,1);
        
        iu_m[mark][0][0] = iu_com[mark][1][1];
        
        grid_no_m = total_no_of_grids;
        n_master = 1;//will always be d starting condi

        for(j=(mark+1); j<=grid_no_m; j++)//check this for np>4
        {
            n_master = 2*n_master;
            iu_m[j] = Assign_mem(n_master, n_master);
            irhs_m[j] = Assign_mem(n_master, n_master);
            ires_m[j] = Assign_mem(n_master, n_master);
            for(x=0; x<n_master; x++)
            {
                for(y=0; y<4; y++)
                {
                    data_m[x][y] = 0;
                }
            }
            
            if(bottom_wall==0)//to bottom neighbour
            {
                tmp_res=0;
                for(i=0; i<(n_master/2); i++)
                {
                    m_restrict[tmp_res] = iu_m[j-1][n_master/2 - 1][i];
                    tmp_res++;
                }
                MPI_Send (m_restrict, n_master/2, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
            }
            interpolate_ver(iu_m[j], iu_m[j-1], n_master, data_m, top_wall);

            if(right_wall==0)//to right neighbour
            {
                tmp_res=0;
                for(i=0; i<(n_master); i++)
                {
                    m_restrict[tmp_res] = iu_m[j][i][n_master - 1];
                    tmp_res++;
                }
                MPI_Send (m_restrict, (n_master), MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
            }
            interpolate_hori(iu_m[j], n_master, data_m, left_wall);
            
            copy(irhs_m[j], (j != grid_no_m ? irho_m[j] : u_work), n_master);//check dis u_work
            
            for(cycle_m=0; cycle_m<no_cycle; cycle_m++)
            {
                nf_m = n_master;
                for(jj=j; jj>=2 && jj>mark; jj--)//modify this for np >4
                {
                    for(x=0; x<nf_m; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_m[x][y] = 0;
                        }
                    }

                    for(jpre=0; jpre<NPRE; jpre++)
                    {
                        //master will always hav wall on top and left
                        if(right_wall==0)//frm right neighbour
                        {
                            MPI_Recv(m_restrict, nf_m, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<nf_m; i++)
                            {
                                data_m[i][0] = m_restrict[i];
                            }
                        }
                        
                        if(bottom_wall==0)//frm bottom neighbour
                        {
                            MPI_Recv(m_restrict, nf_m, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<nf_m; i++)
                            {
                                data_m[i][1] = m_restrict[i];
                            }
                        }

                        //now send ur data out

                        if(right_wall==0)//to right neighbour
                        {
                            tmp_res=0;
                            for(i=0; i<(nf_m); i++)
                            {
                                m_restrict[tmp_res] = iu_m[jj][i][nf_m - 1];
                                tmp_res++;
                            }
                            MPI_Send (m_restrict, nf_m, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                        }

                        if(bottom_wall==0)//to bottom neighbour
                        {
                            tmp_res=0;
                            for(i=0; i<nf_m; i++)
                            {
                                m_restrict[tmp_res] = iu_m[jj][nf_m - 1][i];
                                tmp_res++;
                            }
                            MPI_Send (m_restrict, nf_m, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                        }
                        relax(iu_m[jj], irhs_m[jj], nf_m, data_m, left_wall, right_wall, top_wall, bottom_wall,s);
                    }

                    for(x=0; x<nf_m; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_m[x][y] = 0;
                        }
                    }

                    //master will always hav wall on top and left
                    if(right_wall==0)//frm right neighbour
                    {
                        MPI_Recv(m_restrict, nf_m, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                        for(i=0; i<nf_m; i++)
                        {
                            data_m[i][0] = m_restrict[i];
                        }
                    }
                    
                    if(bottom_wall==0)//frm bottom neighbour
                    {
                        MPI_Recv(m_restrict, nf_m, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                        for(i=0; i<nf_m; i++)
                        {
                            data_m[i][1] = m_restrict[i];
                        }
                    }
                    
                    //now send ur data out
                    
                    if(right_wall==0)//to right neighbour
                    {
                        tmp_res=0;
                        for(i=0; i<(nf_m); i++)
                        {
                            m_restrict[tmp_res] = iu_m[jj][i][nf_m - 1];
                            tmp_res++;
                        }
                        MPI_Send (m_restrict, nf_m, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                    }
                    
                    if(bottom_wall==0)//to bottom neighbour
                    {
                        tmp_res=0;
                        for(i=0; i<nf_m; i++)
                        {
                            m_restrict[tmp_res] = iu_m[jj][nf_m - 1][i];
                            tmp_res++;
                        }
                        MPI_Send (m_restrict, nf_m, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                    }
                    residue(ires_m[jj], iu_m[jj], irhs_m[jj], nf_m, data_m, left_wall, right_wall, top_wall, bottom_wall, s);

                    nf_m = nf_m/2;
                    
                    for(x=0; x<nf_m; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_m[x][y] = 0;
                        }
                    }

                    if(right_wall==0)//frm right neighbour
                    {
                        MPI_Recv(m_restrict, nf_m, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                        for(i=0; i<nf_m; i++)
                        {
                            data_m[i][0] = m_restrict[i];
                        }
                    }
                    
                    if(bottom_wall==0)//frm bottom neighbour
                    {
                        MPI_Recv(m_restrict, nf_m, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                        for(i=0; i<nf_m; i++)
                        {
                            data_m[i][1] = m_restrict[i];
                        }
                    }
                    restrict (irhs_m[jj-1], ires_m[jj], nf_m, data_m, right_wall, bottom_wall);
                    
                    for(x=0; x<nf_m; x++)
                    {
                        for(y=0; y<nf_m; y++)
                        {
                            iu_m[jj-1][x][y]=0.0;
                        }
                    }
                }
                
                //--now everyone has only 1 element left---receive frm slaves    
                //----------COMMON REGION BEGINS-------------------
        
                if(mark==1)
                    irhs_com[mark] = Assign_mem((s+1), (s+1));
                for(i=0; i<s+1; i++)
                    irhs_com[mark][i][0] = 0;
                for(i=0; i<s+1; i++)
                    irhs_com[mark][0][i] = 0;
                irhs_com[mark][1][1] = irhs_m[mark][0][0];

                i=1;
                int j_temp=2;
                for(z=1; z<size_p; z++)
                {
                    MPI_Recv(one_rcv, 1, MPI_DOUBLE, z, tag, MPI_COMM_WORLD, &status);
                    irhs_com[mark][i][j_temp] = one_rcv[0];
                    if(j_temp>=s)
                    {
                        i++;
                        j_temp=1;
                    }
                    else
                    {
                        j_temp++;
                    }
                }
                
                //now check if wat u got back is indeed 3x3 or more----this
                //depends on the number of slaves

                if(mark==1)//np=4
                {
                    for(i=0; i<3; i++)
                    {
                        for(j_temp=0; j_temp<3; j_temp++)
                        {
                            iu_com[1][i][j_temp] = 0;
                        }
                    }
                
                    sol_coarsest(iu_com[1], irhs_com[1]);
                }

                else
                {
                    for(x=0; x<nf_com; x++)
                    {
                        for(y=0; y<nf_com; y++)
                        {
                            iu_com[mark][x][y]=0.0;
                        }
                    }
                    nf_com = s+1;
                    for(jj=mark; jj>=2; jj--)
                    {
                        for(jpre=0; jpre<NPRE; jpre++)
                        {
                            relax_com(iu_com[jj], irhs_com[jj], nf_com);
                        }
                        residue_com(ires_com[jj], iu_com[jj], irhs_com[jj], nf_com);
                        nf_com = nf_com/2 + 1;
                        restrict_com(irhs_com[jj-1], ires_com[jj], nf_com);

                        for(x=0; x<nf_com; x++)
                        {
                            for(y=0; y<nf_com; y++)
                            {
                                iu_com[jj-1][x][y]=0.0;
                            }
                        }
                    }
                    sol_coarsest(iu_com[1], irhs_com[1]);
                    nf_com = 3;
                    for(jj=2; jj<=mark; jj++)
                    {
                        nf_com = 2*nf_com -1;
                        addint_com(iu_com[jj], iu_com[jj-1], ires_com[jj], nf_com);

                        for(jpost=0; jpost<NPOST; jpost++)
                        {
                            relax_com(iu_com[jj], irhs_com[jj], nf_com);
                        }
                    }
                }
                
                iu_m[mark][0][0] = iu_com[mark][1][1];

                i=1;
                j_temp=2;
                for(z=1; z<size_p; z++)
                {
                    iu_send[0] = iu_com[mark][i][j];
                    MPI_Send (iu_send, 1, MPI_DOUBLE, z, tag, MPI_COMM_WORLD);
            
                    if(j_temp>=s)
                    {
                        i++;
                        j_temp=1;
                    }
                    else
                    {
                        j_temp++;
                    }
                }

                //---------------COMMON ENDS-------------------------------------//

                nf_m=1;
                for(jj=(mark+1); jj<=j; jj++)
                {
                    nf_m = 2*nf_m;
                    for(x=0; x<nf_m; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_m[x][y] = 0;
                        }
                    }

                    if(bottom_wall==0)//to bottom neighbour
                    {
                        tmp_res=0;
                        for(i=0; i<(nf_m/2); i++)
                        {
                            m_restrict[tmp_res] = iu_m[jj-1][nf_m/2 - 1][i];
                            tmp_res++;
                        }
                        MPI_Send (m_restrict, nf_m/2, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                    }
                    interpolate_ver(ires_m[jj], iu_m[jj-1], nf_m, data_m, top_wall);
            
                    if(right_wall==0)//to right neighbour
                    {
                        tmp_res=0;
                        for(i=0; i<(nf_m); i++)
                        {
                            m_restrict[tmp_res] = iu_m[jj][i][nf_m - 1];
                            tmp_res++;
                        }
                        MPI_Send (m_restrict, (nf_m), MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                    }
                    interpolate_hori(ires_m[jj], nf_m, data_m, left_wall);
                    
                    addint(iu_m[jj], ires_m[jj], nf_m);
                    
                    for(jpost=0; jpost<NPOST; jpost++)
                    {
                        for(x=0; x<nf_m; x++)
                        {
                            for(y=0; y<4; y++)
                            {
                                data_m[x][y] = 0;
                            }
                        }

                        //master will always hav wall on top and left
                        if(right_wall==0)//frm right neighbour
                        {
                            MPI_Recv(m_restrict, nf_m, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<nf_m; i++)
                            {
                                data_m[i][0] = m_restrict[i];
                            }
                        }
                        
                        if(bottom_wall==0)//frm bottom neighbour
                        {
                            MPI_Recv(m_restrict, nf_m, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<nf_m; i++)
                            {
                                data_m[i][1] = m_restrict[i];
                            }
                        }

                        //now send ur data out

                        if(right_wall==0)//to right neighbour
                        {
                            tmp_res=0;
                            for(i=0; i<(nf_m); i++)
                            {
                                m_restrict[tmp_res] = iu_m[jj][i][nf_m - 1];
                                tmp_res++;
                            }
                            MPI_Send (m_restrict, nf_m, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                        }

                        if(bottom_wall==0)//to bottom neighbour
                        {
                            tmp_res=0;
                            for(i=0; i<nf_m; i++)
                            {
                                m_restrict[tmp_res] = iu_m[jj][nf_m - 1][i];
                                tmp_res++;
                            }
                            MPI_Send (m_restrict, nf_m, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                        }
                        relax(iu_m[jj], irhs_m[jj], nf_m, data_m, left_wall, right_wall, top_wall, bottom_wall,s);
                    }
                }
            }
        }
        
        copy (u_work, iu_m[grid_no_m], size_each);

        //work done now receive u_work from each of the slaves to
        //form the 'u' again.
        
        double * u_rcv = new double[size_each*size_each];

        start_row = 1;
        z=0;
        for(i=0; i<s; i++)
        {
            start_col = 1;
            if(i>0)
            {
                start_row = start_row + size_each;
            }
                        
            for(j=0; j<s; j++)
            {
                k=0;
                if(z==0)//matrix with master filled up
                {
                    for(x=0; x<size_each; x++)
                    {
                        for(y=0; y<size_each; y++)
                        {
                            u[x + start_row][y + start_col] = u_work[x][y];
                        }
                    }
                    z++;
                }
                
                else
                {
                    MPI_Recv(u_rcv, (size_each*size_each), MPI_DOUBLE, z, tag, MPI_COMM_WORLD, &status);
       
                    for(x=0; x<size_each; x++)
                    {
                        for(y=0; y<size_each; y++)
                        {
                            u[x + start_row][y + start_col] = u_rcv[k];
                            k++;
                        }
                    }
                    z++;
                }
                start_col = start_col + size_each;
            }
        }
        delete [] u_rcv;

        time_end=MPI_Wtime();//stop over all timer

        for(j=(total_no_of_grids); j>=(mark+1); j--)
        {
            Delete_mem(ires_m[j]);
            Delete_mem(irhs_m[j]);
            Delete_mem(iu_m[j]);
            if(j != total_no_of_grids)
            {
                Delete_mem(irho_m[j]);
            }
        }

        Delete_mem(irhs_m[mark]);
        Delete_mem(iu_m[mark]);
        Delete_mem(irho_m[mark]);
        
        for(j=mark; j>=2; j--)
        {
            Delete_mem(ires_com[j]);
            Delete_mem(irhs_com[j]);
            Delete_mem(iu_com[j]);
            if(j != total_no_of_grids)
            {
                Delete_mem(irho_com[j]);
            }
        }

        Delete_mem(irhs_com[1]);
        Delete_mem(iu_com[1]);
        Delete_mem(irho_com[1]);

        delete [] temp_data;
        delete [] data_m;
        delete [] m_restrict;
        delete [] one_rcv;
        delete [] iu_send;
        delete [] temp_work;
        delete [] u_work;
        delete [] temp;
        delete [] u;

        cout<<"-----------------------Time taken: ";
        cout<<(time_end-time_start)<<" sec----------------------------"<<endl;
        cout<<"DEBUG: Exiting master"<<endl;
    }
    
    //------------------------slave starts-------------------------------------------    
    else
    {
        int k, n_slave, grid_no_s, cycle_s, nf_s, factor;
        double **irho_s[MAX_GRIDS+1], **irhs_s[MAX_GRIDS+1], **iu_s[MAX_GRIDS+1], **ires_s[MAX_GRIDS+1];

        for(factor=1; factor<=s; factor++)
        {
            if(rank == (factor*s))//left
                left_wall=1;
            if(rank == ((factor*s)-1))//right
                right_wall=1;
        }

        for(factor=0; factor<s; factor++)
        {
            if(rank == factor)//top
                top_wall=1;
            if(rank == ((s*(s-1))+factor))//bottom
                bottom_wall=1;
        }
        
        double * u_rcvd = new double[size_each*size_each];

        MPI_Recv(u_rcvd, (size_each*size_each), MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
                
        double * temp1_work = new double[size_each*size_each];
        double ** u1_work = new double*[size_each];
        for(i=0;i<size_each;i++)
            u1_work[i]=&temp1_work[i*size_each];
                
        k=0;
        for(i=0;i<(size_each);i++)
        {
            for(j=0;j<(size_each);j++)
            {
                u1_work[i][j]=u_rcvd[k];
                k++;
            }
        }
        delete [] u_rcvd;

        //-------------------------------------------multi starts-----------------//
        n_slave = size_each/2;
        grid_no_s = total_no_of_grids - 1;

        irho_s[grid_no_s] = Assign_mem(n_slave, n_slave);
        
        //get/send data frm neighbours
        double * s_restrict = new double[size_each];
        double * temp1_data = new double[4*(2*size_each)];
        double ** data_s = new double * [2*size_each];
        for(i=0; i<2*size_each; i++)
            data_s[i]=&temp1_data[i*4];

        for(x=0; x<2*size_each; x++)
        {
            for(y=0; y<4; y++)
            {
                data_s[x][y] = 0;
            }
        }
        
        for(k=0; k<s; k+=2)//rows alternate
        {
            for(j=0; j<s; j+=2)
            {
                if((rank == (s*k)+j) || (rank == (s*(k+1))+(j+1)))
                {
                    if(right_wall==0)//frm right neighbour
                    {
                        MPI_Recv(s_restrict, n_slave, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                        for(i=0; i<n_slave; i++)
                        {
                            data_s[i][0] = s_restrict[i];
                        }
                    }
                    
                    if(bottom_wall==0)//frm bottom neighbour
                    {
                        MPI_Recv(s_restrict, n_slave, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                        for(i=0; i<n_slave; i++)
                        {
                            data_s[i][1] = s_restrict[i];
                        }
                    }

                    if(left_wall==0)//to left neighbour
                    {
                        tmp_res=0;
                        for(i=1; i<2*n_slave; i+=2)
                        {
                            s_restrict[tmp_res] = u1_work[i][0];
                            tmp_res++;
                        }
                        MPI_Send (s_restrict, n_slave, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                    }
                    
                    if(top_wall==0)//to top neighbour
                    {
                        tmp_res=0;
                        for(i=1; i<2*n_slave; i+=2)
                        {
                            s_restrict[tmp_res] = u1_work[0][i];
                            tmp_res++;
                        }
                        MPI_Send (s_restrict, n_slave, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                    }
                }

                if((rank == (s*k)+j+1) || (rank == (s*(k+1))+j))
                {
                    if(left_wall==0)//to left neighbour
                    {
                        tmp_res=0;
                        for(i=1; i<2*n_slave; i+=2)
                        {
                            s_restrict[tmp_res] = u1_work[i][0];
                            tmp_res++;
                        }
                        MPI_Send (s_restrict, n_slave, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                    }
                    
                    if(top_wall==0)//to top neighbour
                    {
                        tmp_res=0;
                        for(i=1; i<2*n_slave; i+=2)
                        {
                            s_restrict[tmp_res] = u1_work[0][i];
                            tmp_res++;
                        }
                        MPI_Send (s_restrict, n_slave, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                    }

                    if(right_wall==0)//frm right neighbour
                    {
                        MPI_Recv(s_restrict, n_slave, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                        for(i=0; i<n_slave; i++)
                        {
                            data_s[i][0] = s_restrict[i];
                        }
                    }
                    
                    if(bottom_wall==0)//frm bottom neighbour
                    {
                        MPI_Recv(s_restrict, n_slave, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                        for(i=0; i<n_slave; i++)
                        {
                            data_s[i][1] = s_restrict[i];
                        }
                    }
                }
            }
        }
        restrict(irho_s[grid_no_s], u1_work, n_slave, data_s, right_wall, bottom_wall);

        while (n_slave > 1)//smallest has to be 1x1
        {
            for(x=0; x<n_slave; x++)
            {
                for(y=0; y<2; y++)
                {
                    data_s[x][y] = 0;
                }
            }
            n_slave = n_slave/2;
            irho_s[--grid_no_s] = Assign_mem(n_slave, n_slave);

            for(k=0; k<s; k+=2)//rows alternate
            {
                for(j=0; j<s; j+=2)
                {
                    if((rank == (s*k)+j) || (rank == (s*(k+1))+(j+1)))
                    {
                        if(right_wall==0)//frm right neighbour
                        {
                            MPI_Recv(s_restrict, n_slave, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<n_slave; i++)
                            {
                                data_s[i][0] = s_restrict[i];
                            }
                        }
                        
                        if(bottom_wall==0)//frm bottom neighbour
                        {
                            MPI_Recv(s_restrict, n_slave, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<n_slave; i++)
                            {
                                data_s[i][1] = s_restrict[i];   
                            }
                        }
                        
                        if(left_wall==0)//to left neighbour
                        {
                            tmp_res=0;
                            for(i=1; i<2*n_slave; i+=2)
                            {
                                s_restrict[tmp_res] = irho_s[grid_no_s+1][i][0];
                                tmp_res++;
                            }
                            MPI_Send (s_restrict, n_slave, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                        }
                        
                        if(top_wall==0)//to top neighbour
                        {
                            tmp_res=0;
                            for(i=1; i<2*n_slave; i+=2)
                            {
                                s_restrict[tmp_res] = irho_s[grid_no_s+1][0][i];
                                tmp_res++;
                            }
                            MPI_Send (s_restrict, n_slave, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                        }
                    }
                    
                    if((rank == (s*k)+j+1) || (rank == (s*(k+1))+j))
                    {
                        if(left_wall==0)//to left neighbour
                        {
                            tmp_res=0;
                            for(i=1; i<2*n_slave; i+=2)
                            {
                                s_restrict[tmp_res] = irho_s[grid_no_s+1][i][0];
                                tmp_res++;
                            }
                            MPI_Send (s_restrict, n_slave, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                        }
                        
                        if(top_wall==0)//to top neighbour
                        {
                            tmp_res=0;
                            for(i=1; i<2*n_slave; i+=2)
                            {
                                s_restrict[tmp_res] = irho_s[grid_no_s+1][0][i];
                                tmp_res++;
                            }
                            MPI_Send (s_restrict, n_slave, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                        }
                        
                        if(right_wall==0)//frm right neighbour
                        {
                            MPI_Recv(s_restrict, n_slave, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<n_slave; i++)
                            {
                                data_s[i][0] = s_restrict[i];  
                            }
                        }
                        
                        if(bottom_wall==0)//frm bottom neighbour
                        {
                            MPI_Recv(s_restrict, n_slave, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<n_slave; i++)
                            {
                                data_s[i][1] = s_restrict[i];
                            }
                        }
                    }
                }
            }
            restrict(irho_s[grid_no_s], irho_s[grid_no_s + 1], n_slave, data_s, right_wall, bottom_wall);
        }
        
        //--now everyone has only 1 element left---send to master   
        double * one_send = new double[1];
        one_send[0] = irho_s[grid_no_s][0][0];
        
        MPI_Send (one_send, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);

        //now receive the coarsest solution
        double * iu_rcv = new double[1];

        MPI_Recv(iu_rcv, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);

        iu_s[mark] = Assign_mem(1,1);
        irhs_s[mark] = Assign_mem(1,1);
        iu_s[mark][0][0] = iu_rcv[0];
        
        grid_no_s = total_no_of_grids;
        n_slave = 1;//will always be d starting condi

        int j_loop;
        for(j=(mark+1); j<=grid_no_s; j++)//assume np=4
        {
            n_slave = 2*n_slave;
            iu_s[j] = Assign_mem(n_slave, n_slave);
            irhs_s[j] = Assign_mem(n_slave, n_slave);
            ires_s[j] = Assign_mem(n_slave, n_slave);
            for(x=0; x<n_slave; x++)
            {
                for(y=0; y<4; y++)
                {
                    data_s[x][y] = 0;
                }
            }
        
            for(k=0; k<s; k+=2)//rows alternate
            {
                for(j_loop=0; j_loop<s; j_loop+=2)
                {
                    if((rank == (s*k)+j_loop) || (rank == (s*(k+1))+(j_loop+1)))//0 or 3
                    {
                        if(top_wall==0)//frm top neighbour
                        {
                            MPI_Recv(s_restrict, (n_slave/2), MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<(n_slave/2); i++)
                            {
                                data_s[i][3] = s_restrict[i];   
                            }
                        }
                        
                        if(bottom_wall==0)//to bottom neighbour
                        {
                            tmp_res=0;
                            for(i=0; i<(n_slave/2); i++)
                            {
                                s_restrict[tmp_res] = iu_s[j-1][n_slave/2 - 1][i];
                                tmp_res++;
                            }
                            MPI_Send (s_restrict, (n_slave/2), MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                        }
                    }
                    
                    if((rank == (s*k)+j_loop+1) || (rank == (s*(k+1))+j_loop))//1 or 2
                    {
                        if(bottom_wall==0)//to bottom neighbour
                        {
                            tmp_res=0;
                            for(i=0; i<(n_slave/2); i++)
                            {
                                s_restrict[tmp_res] = iu_s[j-1][n_slave/2 - 1][i];
                                tmp_res++;
                            }
                            MPI_Send (s_restrict, (n_slave/2), MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                        }
                        
                        if(top_wall==0)//frm top neighbour
                        {
                            MPI_Recv(s_restrict, n_slave/2, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<n_slave/2; i++)
                            {
                                data_s[i][3] = s_restrict[i];
                            }
                        }
                    }
                }
            }
            interpolate_ver(iu_s[j], iu_s[j-1], n_slave, data_s, top_wall);

            for(x=0; x<n_slave; x++)
            {
                for(y=0; y<4; y++)
                {
                    data_s[x][y] = 0;
                }
            }

            for(k=0; k<s; k+=2)//rows alternate
            {
                for(j_loop=0; j_loop<s; j_loop+=2)
                {
                    if((rank == (s*k)+j_loop) || (rank == (s*(k+1))+(j_loop+1)))
                    {
                        if(left_wall==0)//frm left neighbour
                        {
                            MPI_Recv(s_restrict, (n_slave), MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<(n_slave); i++)
                            {
                                data_s[i][2] = s_restrict[i];   
                            }
                        }
                        
                        if(right_wall==0)//to right neighbour
                        {
                            tmp_res=0;
                            for(i=0; i<(n_slave); i++)
                            {
                                s_restrict[tmp_res] = iu_s[j][i][n_slave - 1];
                                tmp_res++;
                            }
                            MPI_Send (s_restrict, (n_slave), MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                        }
                    }
                    
                    if((rank == (s*k)+j_loop+1) || (rank == (s*(k+1))+j_loop))
                    {
                        if(right_wall==0)//to right neighbour
                        {
                            tmp_res=0;
                            for(i=0; i<(n_slave); i++)
                            {
                                s_restrict[tmp_res] = iu_s[j][i][n_slave - 1];
                                tmp_res++;
                            }
                            MPI_Send (s_restrict, (n_slave), MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                        }
                        
                        if(left_wall==0)//frm left neighbour
                        {
                            MPI_Recv(s_restrict, (n_slave), MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                            for(i=0; i<(n_slave); i++)
                            {
                                data_s[i][2] = s_restrict[i];   
                            }
                        }
                    }
                }
            }
            interpolate_hori(iu_s[j], n_slave, data_s, left_wall);

            copy(irhs_s[j], (j != grid_no_s ? irho_s[j] : u1_work), n_slave);//check dis u_work
            
            for(cycle_s=0; cycle_s<no_cycle; cycle_s++)
            {
                nf_s = n_slave;
                for(jj=j; jj>=2 && jj>mark; jj--)//modify this for np >4
                {
                    for(x=0; x<nf_s; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_s[x][y] = 0;
                        }
                    }


                    for(jpre=0; jpre<NPRE; jpre++)
                    {
                        for(k=0; k<s; k+=2)//rows alternate
                        {
                            for(j_loop=0; j_loop<s; j_loop+=2)
                            {
                                if((rank == (s*k)+j_loop) || (rank == (s*(k+1))+(j_loop+1)))//0 or 3
                                {
                                    if(left_wall==0)//frm left neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][2] = s_restrict[i];
                                        }
                                    }

                                    if(right_wall==0)//frm right neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][0] = s_restrict[i];
                                        }
                                    }

                                    if(top_wall==0)//frm top neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][1] = s_restrict[i];
                                        }
                                    }

                                    if(bottom_wall==0)//frm bottom neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][1] = s_restrict[i];
                                        }
                                    }

                                    //--------nw send--------------
                                    if(left_wall==0)//to left neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][i][0];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                                    }

                                    if(right_wall==0)//to right neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][i][nf_s-1];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                                    }

                                    if(top_wall==0)//to top neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][0][i];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                                    }

                                    if(bottom_wall==0)//to bottom neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][nf_s-1][i];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                                    }
                                }

                                if((rank == (s*k)+j_loop+1) || (rank == (s*(k+1))+j_loop))//1 or 2
                                {
                                    if(left_wall==0)//to left neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][i][0];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                                    }

                                    if(right_wall==0)//to right neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][i][nf_s-1];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                                    }

                                    if(top_wall==0)//to top neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][0][i];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                                    }

                                    if(bottom_wall==0)//to bottom neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][nf_s-1][i];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                                    }

                                    //nw receive
                                    if(left_wall==0)//frm left neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][2] = s_restrict[i];
                                        }
                                    }

                                    if(right_wall==0)//frm right neighbour

                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][0] = s_restrict[i];
                                        }
                                    }

                                    if(top_wall==0)//frm top neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][1] = s_restrict[i];
                                        }
                                    }

                                    if(bottom_wall==0)//frm bottom neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][1] = s_restrict[i];
                                        }
                                    }
                                }
                            }
                        }
                        relax(iu_s[jj], irhs_s[jj], nf_s, data_s, left_wall, right_wall, top_wall, bottom_wall,s);
                    }
                    
                    for(x=0; x<nf_s; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_s[x][y] = 0;
                        }
                    }

                    for(k=0; k<s; k+=2)//rows alternate
                    {
                        for(j_loop=0; j_loop<s; j_loop+=2)
                        {
                            if((rank == (s*k)+j_loop) || (rank == (s*(k+1))+(j_loop+1)))//0 or 3
                            {
                                if(left_wall==0)//frm left neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][2] = s_restrict[i];
                                    }
                                }

                                if(right_wall==0)//frm right neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][0] = s_restrict[i];
                                    }
                                }

                                if(top_wall==0)//frm top neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][1] = s_restrict[i];
                                    }
                                }

                                if(bottom_wall==0)//frm bottom neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][1] = s_restrict[i];
                                    }
                                }

                                //--------nw send--------------
                                if(left_wall==0)//to left neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<nf_s; i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][i][0];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                                }

                                if(right_wall==0)//to right neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<nf_s; i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][i][nf_s-1];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                                }

                                if(top_wall==0)//to top neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<nf_s; i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][0][i];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                                }

                                if(bottom_wall==0)//to bottom neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<nf_s; i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][nf_s-1][i];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                                }
                            }

                            if((rank == (s*k)+j_loop+1) || (rank == (s*(k+1))+j_loop))//1 or 2
                            {
                                if(left_wall==0)//to left neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<nf_s; i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][i][0];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                                }

                                if(right_wall==0)//to right neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<nf_s; i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][i][nf_s-1];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                                }

                                if(top_wall==0)//to top neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<nf_s; i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][0][i];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                                }

                                if(bottom_wall==0)//to bottom neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<nf_s; i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][nf_s-1][i];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                                }


                                //nw receive
                                if(left_wall==0)//frm left neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][2] = s_restrict[i];
                                    }
                                }

                                if(right_wall==0)//frm right neighbour

                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][0] = s_restrict[i];
                                    }
                                }

                                if(top_wall==0)//frm top neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][1] = s_restrict[i];
                                    }
                                }

                                if(bottom_wall==0)//frm bottom neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][1] = s_restrict[i];
                                    }
                                }
                            }
                        }
                    }
                    residue(ires_s[jj], iu_s[jj], irhs_s[jj], nf_s, data_s, left_wall, right_wall, top_wall, bottom_wall, s);
                    nf_s = nf_s/2;

                    for(x=0; x<nf_s; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_s[x][y] = 0;
                        }
                    }
                    
                    for(k=0; k<s; k+=2)//rows alternate
                    {
                        for(j_loop=0; j_loop<s; j_loop+=2)
                        {
                            if((rank == (s*k)+j_loop) || (rank == (s*(k+1))+(j_loop+1)))
                            {
                                if(right_wall==0)//frm right neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][0] = s_restrict[i];
                                    }
                                }
                        
                                if(bottom_wall==0)//frm bottom neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][1] = s_restrict[i];   
                                    }
                                }
                        
                                if(left_wall==0)//to left neighbour
                                {
                                    tmp_res=0;
                                    for(i=1; i<2*nf_s; i+=2)
                                    {
                                        s_restrict[tmp_res] = ires_s[jj][i][0];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                                }
                        
                                if(top_wall==0)//to top neighbour
                                {
                                    tmp_res=0;
                                    for(i=1; i<2*nf_s; i+=2)
                                    {
                                        s_restrict[tmp_res] = ires_s[jj][0][i];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                                }
                            }
                    
                            if((rank == (s*k)+j_loop+1) || (rank == (s*(k+1))+j_loop))
                            {
                                if(left_wall==0)//to left neighbour
                                {
                                    tmp_res=0;
                                    for(i=1; i<2*nf_s; i+=2)
                                    {
                                        s_restrict[tmp_res] = ires_s[jj][i][0];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                                }
                        
                                if(top_wall==0)//to top neighbour
                                {
                                    tmp_res=0;
                                    for(i=1; i<2*nf_s; i+=2)
                                    {
                                        s_restrict[tmp_res] = ires_s[jj][0][i];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                                }
                        
                                if(right_wall==0)//frm right neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][0] = s_restrict[i];  
                                    }
                                }
                        
                                if(bottom_wall==0)//frm bottom neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s; i++)
                                    {
                                        data_s[i][1] = s_restrict[i];
                                    }
                                }
                            }
                        }
                    }
                    restrict (irhs_s[jj-1], ires_s[jj], nf_s, data_s, right_wall, bottom_wall);
                    
                    for(x=0; x<nf_s; x++)
                    {
                        for(y=0; y<nf_s; y++)
                        {
                            iu_s[jj-1][x][y]=0.0;
                        }
                    }
                }

                //--now everyone has only 1 element left---send to master

                one_send[0] = irhs_s[mark][0][0];
                MPI_Send (one_send, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);

                //--now wait for master to get back
                MPI_Recv(iu_rcv, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
                iu_s[mark][0][0] = iu_rcv[0];
                
                nf_s=1;
                for(jj=(mark+1); jj<=j; jj++)
                {
                    nf_s = 2*nf_s;

                    for(x=0; x<nf_s; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_s[x][y] = 0;
                        }
                    }

                    for(k=0; k<s; k+=2)//rows alternate
                    {
                        for(j_loop=0; j_loop<s; j_loop+=2)
                        {
                            if((rank == (s*k)+j_loop) || (rank == (s*(k+1))+(j_loop+1)))//0 or 3
                            {
                                if(top_wall==0)//frm top neighbour
                                {
                                    MPI_Recv(s_restrict, (nf_s/2), MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<(nf_s/2); i++)
                                    {
                                        data_s[i][3] = s_restrict[i];   
                                    }
                                }
                        
                                if(bottom_wall==0)//to bottom neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<(nf_s/2); i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj-1][nf_s/2 - 1][i];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, (nf_s/2), MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                                }
                            }
                    
                            if((rank == (s*k)+j_loop+1) || (rank == (s*(k+1))+j_loop))//1 or 2
                            {
                                if(bottom_wall==0)//to bottom neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<(nf_s/2); i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj-1][nf_s/2 - 1][i];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, (nf_s/2), MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                                }
                        
                                if(top_wall==0)//frm top neighbour
                                {
                                    MPI_Recv(s_restrict, nf_s/2, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<nf_s/2; i++)
                                    {
                                        data_s[i][3] = s_restrict[i];
                                    }
                                }
                            }
                        }
                    }
                    interpolate_ver(ires_s[jj], iu_s[jj-1], nf_s, data_s, top_wall);

                    for(x=0; x<nf_s; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_s[x][y] = 0;
                        }
                    }

                    
                    for(k=0; k<s; k+=2)//rows alternate
                    {
                        for(j_loop=0; j_loop<s; j_loop+=2)
                        {
                            if((rank == (s*k)+j_loop) || (rank == (s*(k+1))+(j_loop+1)))
                            {
                                if(left_wall==0)//frm left neighbour
                                {
                                    MPI_Recv(s_restrict, (nf_s), MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<(nf_s); i++)
                                    {
                                        data_s[i][2] = s_restrict[i];   
                                    }
                                }
                        
                                if(right_wall==0)//to right neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<(nf_s); i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][i][nf_s - 1];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, (nf_s), MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                                }
                            }
                    
                            if((rank == (s*k)+j_loop+1) || (rank == (s*(k+1))+j_loop))
                            {
                                if(right_wall==0)//to right neighbour
                                {
                                    tmp_res=0;
                                    for(i=0; i<(nf_s); i++)
                                    {
                                        s_restrict[tmp_res] = iu_s[jj][i][nf_s - 1];
                                        tmp_res++;
                                    }
                                    MPI_Send (s_restrict, (nf_s), MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                                }
                        
                                if(left_wall==0)//frm left neighbour
                                {
                                    MPI_Recv(s_restrict, (nf_s), MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                                    for(i=0; i<(nf_s); i++)
                                    {
                                        data_s[i][2] = s_restrict[i];   
                                    }
                                }
                            }
                        }
                    }

                    interpolate_hori(ires_s[jj], nf_s, data_s, left_wall);
                   
                    addint(iu_s[jj], ires_s[jj], nf_s);

                    for(x=0; x<nf_s; x++)
                    {
                        for(y=0; y<4; y++)
                        {
                            data_s[x][y] = 0;
                        }
                    }
                    
                    for(jpost=0; jpost<NPOST; jpost++)
                    {
                        for(k=0; k<s; k+=2)//rows alternate
                        {
                            for(j_loop=0; j_loop<s; j_loop+=2)
                            {
                                if((rank == (s*k)+j_loop) || (rank == (s*(k+1))+(j_loop+1)))//0 or 3
                                {
                                    if(left_wall==0)//frm left neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][2] = s_restrict[i];
                                        }
                                    }

                                    if(right_wall==0)//frm right neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][0] = s_restrict[i];
                                        }
                                    }

                                    if(top_wall==0)//frm top neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][1] = s_restrict[i];
                                        }
                                    }

                                    if(bottom_wall==0)//frm bottom neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][1] = s_restrict[i];
                                        }
                                    }

                                    //--------nw send--------------
                                    if(left_wall==0)//to left neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][i][0];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                                    }

                                    if(right_wall==0)//to right neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][i][nf_s-1];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                                    }

                                    if(top_wall==0)//to top neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][0][i];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                                    }

                                    if(bottom_wall==0)//to bottom neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][nf_s-1][i];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                                    }
                                }

                                if((rank == (s*k)+j_loop+1) || (rank == (s*(k+1))+j_loop))//1 or 2
                                {
                                    if(left_wall==0)//to left neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][i][0];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD);
                                    }

                                    if(right_wall==0)//to right neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][i][nf_s-1];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD);
                                    }

                                    if(top_wall==0)//to top neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][0][i];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD);
                                    }

                                    if(bottom_wall==0)//to bottom neighbour
                                    {
                                        tmp_res=0;
                                        for(i=0; i<nf_s; i++)
                                        {
                                            s_restrict[tmp_res] = iu_s[jj][nf_s-1][i];
                                            tmp_res++;
                                        }
                                        MPI_Send (s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD);
                                    }


                                    //nw receive
                                    if(left_wall==0)//frm left neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-1), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][2] = s_restrict[i];
                                        }
                                    }

                                    if(right_wall==0)//frm right neighbour

                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+1), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][0] = s_restrict[i];
                                        }
                                    }

                                    if(top_wall==0)//frm top neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank-s), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][1] = s_restrict[i];
                                        }
                                    }

                                    if(bottom_wall==0)//frm bottom neighbour
                                    {
                                        MPI_Recv(s_restrict, nf_s, MPI_DOUBLE, (rank+s), tag, MPI_COMM_WORLD, &status);
                                        for(i=0; i<nf_s; i++)
                                        {
                                            data_s[i][1] = s_restrict[i];
                                        }
                                    }
                                }
                            }
                        }
                        relax(iu_s[jj], irhs_s[jj], nf_s, data_s, left_wall, right_wall, top_wall, bottom_wall,s);
                    }
                }
            }
        }

        copy (u1_work, iu_s[grid_no_s], size_each);

        //work done send back u to master
        double * u_sen = new double[size_each*size_each];

        k=0;
        for(i=0;i<(size_each);i++)
        {
            for(j=0;j<(size_each);j++)
            {
                u_sen[k] = u1_work[i][j];
                k++;
            }
        }
        MPI_Send (u_sen, (size_each*size_each), MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        
        for(j=total_no_of_grids; j>=(mark+1); j--)
        {
            Delete_mem(ires_s[j]);
            Delete_mem(irhs_s[j]);
            Delete_mem(iu_s[j]);
            if(j != total_no_of_grids)
            {
                Delete_mem(irho_s[j]);
            }
        }

        Delete_mem(irhs_s[mark]);
        Delete_mem(iu_s[mark]);
        Delete_mem(irho_s[mark]);
        

        delete [] temp1_data;
        delete [] data_s;
        delete [] s_restrict;
        delete [] u_sen;
        delete [] one_send;
        delete [] iu_rcv;
        delete [] temp1_work;
        delete [] u1_work;
       
        cout<<"DEBUG: exiting slave"<<endl;

    }

    MPI_Finalize( );
    return 0;
}

#undef NPRE
#undef NPOST
#undef MAX_GRIDS
#undef SIZE
#undef NO_OF_CYCLES

