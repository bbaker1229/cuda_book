// Copyright 2021 Bryan Baker

// Compile using:
// nvcc Ch3_test.cu -o test -lnvjpeg

#include <nvjpeg.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    nvjpegStatus_t stat;
    nvjpegHandle_t *handle;
    nvjpegJpegState_t *jpeg_handle;

    if (argc != 2) {
        fprintf(stderr, "USAGE: %s filename.jpg\n", argv[0]);
        exit(1);
    }

    printf("Starting test.\n");
    stat = nvjpegCreateSimple(handle);
    stat = nvjpegJpegStateCreate(*handle, jpeg_handle);
    printf("Done\n");
}
