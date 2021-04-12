// Kernel for matrix transposition.
__kernel
void transposeMat( __global float *a, __global float *b, __global int *row_p, __global int *col_p)
{
	// The global id tells us the index of the vector for this thread.

    int col = *col_p;
    int row = *row_p;

	int j = get_global_id(0) % col;
	int i = (get_global_id(0) - j) / col;

	//b[j * row + i] = a[i * col + j];
	b[0] = a[0];
	b[1] = a[3];
	//b[get_global_id(0)] = get_global_id(0);

	// Perform the addition.
	//c[gid] = a[gid];
}