#ifndef CONV2D
#define CONV2D

#include "types.h"
#include <mc_scverify.h>

// Convolution with stride 1 and odd size kernel

#pragma hls_design interface
void conv2d(float *in, float *out, int HEIGHT, int WIDTH, int IN_FMAP, int OUT_FMAP, int KSIZE){
	float in_fmaps[IN_FMAP][HEIGHT][WIDTH] = *in;
	float kernel[OUT_FMAP][IN_FMAP][KSIZE][KSIZE];
	float out_fmaps[OUT_FMAP][HEIGHT][WIDTH]); //prepei na mpei arxikopoihsh
	

    float acc = 0;
    float acc_buf[HEIGHT][WIDTH];
    float data;

    OF: for (int ofm=0; ofm<OUT_FMAP; ofm++) { // Output feature map
      // Clear the accum buffer
      ROW_CLR: for (int r=0; r<HEIGHT; r++) { // Feature map
        COL_CLR: for (int c=0; c<WIDTH; c++) {
          acc_buf[r][c] = 0;
        }
      }
      IFM: for (int ifm=0; ifm<IN_FMAP; ifm++) { // Input feature map
        ROW: for (int r=0; r<HEIGHT; r++) { // Process feature map
          COL: for (int c=0; c<WIDTH; c++) {
            acc = 0;
            K_X: for (int kr=0; kr<KSIZE; kr++) { // Odd size kernel
              K_Y: for (int kc=0; kc<KSIZE; kc++) { // Odd size kernel
                int ridx = r + kr - KSIZE/2; // Compute indices based on filter kernel location
                int cidx = c + kc - KSIZE/2;
                if (ridx < 0 || ridx >= HEIGHT || cidx < 0 || cidx >= WIDTH) { // Zero pad boundary when index out of bounds
                  data = 0;
                } else {
                  data = in_fmaps[ifm][ridx][cidx];
                }
                acc += data*kernel[ofm][ifm][kr][kc]; // Perform convolution against input fmap
              }
            }
            acc_buf[r][c] += acc; // Sum current fmap activation across all input channels/feature maps
          }
        }
      }
      // Write output feature map
      ROW_CPY: for (int r=0; r<HEIGHT; r++) { // Feature map
        COL_CPY: for (int c=0; c<WIDTH; c++) {
          out_fmaps[ofm][r][c] = acc_buf[r][c];
        }
      }
    }
  }
};
#endif
