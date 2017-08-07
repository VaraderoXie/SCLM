//
// Created by joe on 2/3/17.
//


#include "matrix.h"
#include "mex.h"
//#include "tbb/task_scheduler_init.h"

#include <cmath>
#include <iostream>
#include <stdint.h>
#include <limits>

/* Taken from: "Similarity Learning for High-Dimensional Sparse Data" AAAI '15
 * get value corresponding to element (i,j) in sparse matrix
 * i,j start at 1
 * based on binary search
 */
double getValue (mwIndex *Ir, mwIndex *Jc, double *Pr, int i, int j) {

    int k, left, right, mid, cas;
    left = Jc[j-1];
    right = Jc[j]-1;
    while (left <= right) {
        mid = left + (right-left)/2;
        if (Ir[mid]+1 == i)
            return Pr[mid];
        else if (Ir[mid]+1 > i)
            right = mid - 1;
        else
            left = mid + 1;
    }
    return 0.0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //Expected inputs:
    // Sparse matrix G
    // double lambda
    // third input -- random number between 1 and # rows in G

    //Outputs
    // i -- row index
    // j -- col index
    // sign -- (-1 or 1)

    if(nrhs != 3)
        mexErrMsgTxt("Wrong # of input arguments!");

    if(!mxIsSparse(prhs[0]))
        mexErrMsgTxt("First input must be a sparse matrix");

    if(!mxIsDouble(prhs[1]))
        mexErrMsgTxt("Second input must be a double");

    if(!mxIsDouble(prhs[2]))
        mexErrMsgTxt("Third input must be a double");

    int nzmax = mxGetNzmax(prhs[0]);
    int rows = mxGetM(prhs[0]);
    int cols = mxGetN(prhs[0]);

    int rand_start = (int)mxGetScalar(prhs[2]);
    if(rand_start < 1 || rand_start > rows)
        mexErrMsgTxt("Third input out of range!");

    //start by loading in G and lambda
    double* gPr = mxGetPr(prhs[0]);
    mwIndex* gJc = mxGetJc(prhs[0]);
    mwIndex* gIr = mxGetIr(prhs[0]);
    double lambda = mxGetScalar(prhs[1]);


    //std::cout << "rows: " << rows << std::endl;
    //std::cout << "cols: " << cols << std::endl;
    //std::cout << "lambda: " << lambda << std::endl;

    double best_score = std::numeric_limits<double>::max();
    double G_vals[4];

    int best_i = 1;
    int best_j = 1;
    int sign = 1;

    double score;

    // start out with row=rand_start
    G_vals[0] = getValue(gIr, gJc, gPr, rand_start, rand_start);
    for(int j = 1; j <=cols; ++j) {
        if (j == rand_start){
            score = lambda * G_vals[0];
            if (score < best_score) {
                best_score = score;
                best_j = j;
                sign = 1;
            }
        }else{

            G_vals[1] = getValue(gIr, gJc, gPr, rand_start, j);
            G_vals[2] = getValue(gIr, gJc, gPr, j, rand_start);
            G_vals[3] = getValue(gIr, gJc, gPr, j, j);

            score = lambda*(G_vals[0] + G_vals[1] + G_vals[2] + G_vals[3]);
            if(score < best_score){
                best_score = score;
                best_j = j;
                sign = 1;
            }
            score = lambda*(G_vals[0] - G_vals[1] - G_vals[2] + G_vals[3]);
            if(score < best_score){
                best_score = score;
                best_j = j;
                sign = -1;
            }
        }
    }

    //we now have best_j set, so we can steam ahead and find the best_i
    G_vals[3] = getValue(gIr, gJc, gPr, best_j, best_j);
    for(int i = 1; i <=rows; ++i){
        if(i == best_j){
            score = lambda * G_vals[0];

            if (score < best_score) {
                best_score = score;
                best_i = i;
                sign = 1;
            }
        }else {

            G_vals[0] = getValue(gIr, gJc, gPr, i, i);
            G_vals[1] = getValue(gIr, gJc, gPr, i, best_j);
            G_vals[2] = getValue(gIr, gJc, gPr, best_j, i);

            score = lambda*(G_vals[0] + G_vals[1] + G_vals[2] + G_vals[3]);
            if (score < best_score) {
                best_score = score;
                best_i = i;
                sign = 1;
            }
            score = lambda*(G_vals[0] - G_vals[1] - G_vals[2] + G_vals[3]);
            if (score < best_score) {
                best_score = score;
                best_i = i;
                sign = -1;
            }
        }
    }

    plhs[0] = mxCreateDoubleScalar((double)best_i);
    plhs[1] = mxCreateDoubleScalar((double)best_j);
    plhs[2] = mxCreateDoubleScalar((double)sign);

    return;
}




/*
 */
