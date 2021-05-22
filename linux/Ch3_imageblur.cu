// Copyright 2021 Bryan Baker

// Compile using:
// nvcc Ch3_imageblur.cu -o imageblur -ljpeg

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <sys/time.h>
#include <jpeglib.h>

#define BLUR_SIZE 1

void blurCPU(unsigned char *in, unsigned char *out, int w, int h) {
    int Row, Col;
    for(Row = 0; Row < h; Row++) {
        for(Col = 0; Col < w; Col++) {
            int pixVal = 0;
            int pixels = 0;
    
            // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
            for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                    int curRow = Row + blurRow;
                    int curCol = Col + blurCol;
                    // Verify we have a valid image pixel
                    if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                        pixVal += in[curRow * w + curCol];
                        pixels++;  // Keep track of number of pixels in the avg
                    }
                }
            }
            // Write our new pixel value out
            out[Row * w + Col] = (unsigned char)(pixVal / pixels);
        }
    }
}

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        int pixVal = 0;
        int pixels = 0;

        // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                // Verify we have a valid image pixel
                if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    pixels++;  // Keep track of number of pixels in the avg
                }
            }
        }
        // Write our new pixel value out
        out[Row * w + Col] = (unsigned char)(pixVal / pixels);
    }
}

void colorToGreyscaleConversion(unsigned char *Pout, unsigned char *Pin
                                , int width, int height, int channels) {
    int buf_size, rgbOffset;
    unsigned char r, g, b;

    buf_size = width * height;
    for (int i = 0; i < buf_size; i++) {
        rgbOffset = i * channels;
        r = Pin[rgbOffset];
        g = Pin[rgbOffset + 1];
        b = Pin[rgbOffset + 2];
        Pout[i] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

__global__
void colorToGreyscaleConversionKernel(unsigned char *Pout, unsigned char *Pin
                                        , int width, int height, int channels) {
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height) {
        // get 1D coordinate for the greyscale image
        int greyOffset = Row*width + Col;
        // one can think of the RGB image having
        // channels times columns than the grayscale image
        int rgbOffset = greyOffset*channels;
        unsigned char r = Pin[rgbOffset];  // red value for pixel
        unsigned char g = Pin[rgbOffset+1];  // green value for pixel
        unsigned char b = Pin[rgbOffset+2];  // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        Pout[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

double wctime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

int main(int argc, char *argv[]) {
    int rc, i;
    double t1;
    float nops;

    if (argc != 2) {
        fprintf(stderr, "USAGE: %s filename.jpg\n", argv[0]);
        exit(1);
    }

    // Variables for the source jpg
    struct stat file_info;
    u_int64_t jpg_size;
    unsigned char *jpg_buffer;

    // Variables for the decompressor
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // Variables for the output buffer, and how long each row is
    u_int64_t bmp_size;
    unsigned char *bmp_buffer;
    unsigned char *grey_buffer;
    unsigned char *blur_buffer;
    unsigned char *d_bmp_buffer;
    unsigned char *d_grey_buffer;
    unsigned char *d_blur_buffer;
    int row_stride, width, height, pixel_size;

    // Get file info
    rc = stat(argv[1], &file_info);
    jpg_size = file_info.st_size;
    jpg_buffer = (unsigned char*) malloc(jpg_size + 100);
    int fd = open(argv[1], O_RDONLY);
    i = 0;
    while (i < jpg_size) {
        rc = read(fd, jpg_buffer + i, jpg_size - i);
        i += rc;
    }
    close(fd);
    cinfo.err = jpeg_std_error(&jerr);

    // Decompress file
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, jpg_buffer, jpg_size);
    rc = jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    // Create output buffer
    width = cinfo.output_width;
    height = cinfo.output_height;
    pixel_size = cinfo.output_components;
    bmp_size = width * height * pixel_size;
    bmp_buffer = (unsigned char*) malloc(bmp_size);
    row_stride = width * pixel_size;
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *buffer_array[1];
        buffer_array[0] = bmp_buffer + \
                            (cinfo.output_scanline) * row_stride;
        jpeg_read_scanlines(&cinfo, buffer_array, 1);
    }

    // clean up
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    free(jpg_buffer);

    // Convert to grey scale
    grey_buffer = (unsigned char*) malloc(width * height);
    blur_buffer = (unsigned char*) malloc(width * height);
    colorToGreyscaleConversion(grey_buffer, bmp_buffer
                                , width, height, pixel_size);
    t1 = wctime();
    blurCPU(grey_buffer, blur_buffer, width, height);
    t1 = wctime() - t1;

    printf("CPU:\n");
    printf("Finished in %lf seconds.\n", t1);
    t1 *= (1.E+09);
    nops = (float) (5 * width * height);
    printf("Performance = %f GFLOPs\n", nops/t1);
    printf("\n");

    cudaMalloc((void**)&d_grey_buffer, width*height);
    cudaMalloc((void**)&d_bmp_buffer, width*height*pixel_size);
    cudaMalloc((void**)&d_blur_buffer, width*height);
    t1 =wctime();
    cudaMemcpy(d_grey_buffer, grey_buffer, width*height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bmp_buffer, bmp_buffer, width*height*pixel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blur_buffer, blur_buffer, width*height, cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(width/32.0), ceil(height/32.0), 1);
    dim3 dimBlock(32, 32, 1);
    colorToGreyscaleConversionKernel<<<dimGrid, dimBlock>>>(d_grey_buffer, d_bmp_buffer, width, height, pixel_size);
    blurKernel<<<dimGrid, dimBlock>>>(d_grey_buffer, d_blur_buffer, width, height);
    cudaMemcpy(blur_buffer, d_blur_buffer, width*height, cudaMemcpyDeviceToHost);
    t1 = wctime() - t1;

    printf("Cuda with data transfer:\n");
    printf("Finished in %lf seconds.\n", t1);
    t1 *= (1.E+09);
    nops = (float) (5 * width * height);
    printf("Performance = %f GFLOPs\n", nops/t1);
    printf("\n");

    cudaMalloc((void**)&d_grey_buffer, width*height);
    cudaMemcpy(d_grey_buffer, grey_buffer, width*height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bmp_buffer, bmp_buffer, width*height*pixel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blur_buffer, blur_buffer, width*height, cudaMemcpyHostToDevice);
    colorToGreyscaleConversionKernel<<<dimGrid, dimBlock>>>(d_grey_buffer, d_bmp_buffer, width, height, pixel_size);
    t1 = wctime();
    blurKernel<<<dimGrid, dimBlock>>>(d_grey_buffer, d_blur_buffer, width, height);
    t1 = wctime() - t1;
    cudaMemcpy(blur_buffer, d_blur_buffer, width*height, cudaMemcpyDeviceToHost);

    printf("Cuda without data transfer:\n");
    printf("Finished in %lf seconds.\n", t1);
    t1 *= (1.E+09);
    nops = (float) (5 * width * height);
    printf("Performance = %f GFLOPs\n", nops/t1);

    // Write the blur map to a file.
    fd = open("blur_output.pgm", O_CREAT | O_WRONLY, 0666);
    char buf[1024];
    rc = snprintf(buf, sizeof(buf), "P5 %d %d 255\n", width, height);
    write(fd, buf, rc);  // Write the PGM image header before data
    write(fd, blur_buffer, width * height);  // Write out all pixel data
    close(fd);

    // Free buffer
    cudaFree(d_bmp_buffer);
    cudaFree(d_grey_buffer);
    cudaFree(d_blur_buffer);
    free(bmp_buffer);
    free(grey_buffer);
    free(blur_buffer);

    // Exit program
    return EXIT_SUCCESS;
}
