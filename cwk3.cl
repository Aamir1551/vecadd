// Kernel for matrix transposition.
__kernel
void transposeMat( __constant float *a, __global float *b, int cols, int rows)
{

	//int j = get_global_id(0) % col;
	//int i = (get_global_id(0) - j) / col;

	//b[j * row + i] = a[i * col + j];

    unsigned int x_pos = get_global_id(1);
    unsigned int y_pos = get_global_id(0);

    unsigned int index_in = x_pos + cols * y_pos;
    unsigned int index_out = y_pos + rows + x_pos;

    b[index_out] = a[index_in];

}