#include <iostream>
#include <cuda_runtime.h>

int main()
{
	int4 a={1,2,3,4};
	std::cout << a.x << " " << a.y << " " << a.z << " " << a.w << std::endl;
	return 0;
}
