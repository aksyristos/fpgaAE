#ifndef _INCLUDED_LEAKYRELUALG_H_
#define _INCLUDED_LEAKYRELUALG_H_

#include "types.h"
#include "top.h"
#include <mc_scverify.h>

// Convolution with stride 1 and odd size kernel
template<>

void leakyReLuAlg(dt a[LEN], int MAX_HEIGHT, int MAX_WIDTH, int IN_FMAP, int MEM_SIZE){



	dt out_fmaps[LEN] = a;



    IFM: for (int ifm=0; ifm<IN_FMAP; ifm++) { // Input feature map
      ROW: for (int r=0; r<MAX_HEIGHT; r++) { // Process feature map
        COL: for (int c=0; c<MAX_WIDTH; c++) {
          data = in_fmaps[read_offset + ifm*height*width + r*width + c];
          if (data < 0) { 
			//data = data * SAT_TYPE(0.1); 
			data = 0;
			}
          out_fmaps[write_offset + ifm*height*width + r*width + c] = data; //antistoixhsh me b or c
          if (c == width-1) { break; }
        }
        if (r == height-1) { break; }
      }
      if (ifm == num_in_fmaps-1) { break; }
    }
  }
};
#endif