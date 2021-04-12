// Kernel for matrix transposition.
__kernel
void transposeMat( __constant float *a, __global float *b,  int row, int col)
{

	int j = get_global_id(0) % col;
	int i = (get_global_id(0) - j) / col;

	b[j * row + i] = a[i * col + j];
}