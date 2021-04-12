// Kernel for matrix transposition.
__kernel
void transposeMat( __constant float *a, __global float *b,  int row_p, int col_p)
{
	// The global id tells us the index of the vector for this thread.

    int col = col_p;
    int row = row_p;

	int j = get_global_id(0) % col;
	int i = (get_global_id(0) - j) / col;

	b[j * row + i] = a[i * col + j];
	//b[get_global_id(0)] = get_global_id(0);

	// Perform the addition.
	//c[gid] = a[gid];
}