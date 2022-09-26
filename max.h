#ifndef CONV2D
#define CONV2D

#include "types.h"
#include <mc_scverify.h>



#pragma hls_design interface
void max_pool(float *in, float *out, int IN_FMAP, int in_row, int in_col, int K_SIZE)
{
  float acc=0;
  float temp;
  for (int ifm=0; ifm<IN_FMAP; ifm++) {
  for (int i=0; i<in_row; i++) {
    if (i+K_SIZE-1 == in_row)
    { break; }
    for (int j=0; j<in_col; j++) {
      acc = 0;
      if (j+K_SIZE-1 == in_col) { break; }
      for (int k=0; k<K_SIZE; k++) {
        for (int l=0; l<K_SIZE; l++) {
          double inp_tmp = *( in + (((i+k)*in_col)+(j+l)) );
          temp =  (temp > inp_tmp) ? temp : inp_tmp; //temp > in[i+k][j+l] or what
        }
      }
      if (temp>0) {
        *( out + i*(in_col-K_SIZE+1) + j ) = temp;
      } else { //out[i][j] = temp
        * ( out + i*(in_col-K_SIZE+1) + j ) =  0;
      }
    }
  }
  }
}
#endif