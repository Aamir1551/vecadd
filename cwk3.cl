// Kernel for matrix transposition.
__kernel
void transposeMat( __global float *a)
{
	// The global id tells us the index of the vector for this thread.
	int gid = get_global_id(0);

	// Perform the addition.
	c[gid] = a[gid];
}