int get_GPU_Prop() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("deviceProp.clockRate = %d\n", deviceProp.clockRate);
    printf("deviceProp.totalGlobalMem = %zu\n", deviceProp.totalGlobalMem);
    printf("deviceProp.warpSize = %d\n", deviceProp.warpSize);
    printf("deviceProp.totalConstMem = %zu\n", deviceProp.totalConstMem);
    printf("deviceProp.canMapHostMemory = %d\n", deviceProp.canMapHostMemory);
    printf("deviceProp.minor = %d\n", deviceProp.minor); // Minor compute capability, e.g. cuda 9.0

    // about shared memory 
    printf("\n");
    printf("deviceProp.sharedMemPerBlockOptin = %zu\n", deviceProp.sharedMemPerBlockOptin);
    printf("deviceProp.sharedMemPerBlock = %zu\n", deviceProp.sharedMemPerBlock);
    printf("deviceProp.sharedMemPerMultiprocessor = %zu\n", deviceProp.sharedMemPerMultiprocessor);
    
    // about SM and block
    printf("\n");
    printf("deviceProp.multiProcessorCount = %d\n", deviceProp.multiProcessorCount);
    printf("deviceProp.maxBlocksPerMultiProcessor = %d\n", deviceProp.maxBlocksPerMultiProcessor);
    printf("deviceProp.maxThreadsPerBlock = %d\n", deviceProp.maxThreadsPerBlock);
    printf("deviceProp.maxThreadsPerMultiProcessor = %d\n", deviceProp.maxThreadsPerMultiProcessor);
    
    // about registers  
    printf("\n");  
    printf("deviceProp.regsPerBlock = %d\n", deviceProp.regsPerBlock);
    printf("deviceProp.regsPerMultiprocessor = %d\n", deviceProp.regsPerMultiprocessor);

    // something
    printf("\n");
    printf("deviceProp.maxGridSize[0] = %d, deviceProp.maxGridSize[1] = %d, deviceProp.maxGridSize[2] = %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    return deviceProp.clockRate;
}


int main () {
    int a = get_GPU_Prop();
}