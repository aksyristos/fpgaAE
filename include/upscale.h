/**************************************************************************
 *                                                                        *
 *  Catapult(R) Machine Learning Reference Design Library                 *
 *                                                                        *
 *  Software Version: 1.5                                                 *
 *                                                                        *
 *  Release Date    : Fri Oct 29 16:53:36 PDT 2021                        *
 *  Release Type    : Production Release                                  *
 *  Release Build   : 1.5.0                                               *
 *                                                                        *
 *  Copyright 2021 Siemens                                                *
 *                                                                        *
 **************************************************************************
 *  Licensed under the Apache License, Version 2.0 (the "License");       *
 *  you may not use this file except in compliance with the License.      * 
 *  You may obtain a copy of the License at                               *
 *                                                                        *
 *      http://www.apache.org/licenses/LICENSE-2.0                        *
 *                                                                        *
 *  Unless required by applicable law or agreed to in writing, software   * 
 *  distributed under the License is distributed on an "AS IS" BASIS,     * 
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or       *
 *  implied.                                                              * 
 *  See the License for the specific language governing permissions and   * 
 *  limitations under the License.                                        *
 **************************************************************************
 *                                                                        *
 *  The most recent version of this package is available at github.       *
 *                                                                        *
 *************************************************************************/
#ifndef _INCLUDED_UPSCALE_H_
#define _INCLUDED_UPSCALE_H_

#include "types.h"
#include <mc_scverify.h>

// Convolution with stride 1 and odd size kernel
template<typename DTYPE, int MAX_HEIGHT, int MAX_WIDTH, int IN_FMAP, int MEM_SIZE>
class upscaleAlg
{
  enum {
    HEIGHT_BITS = ac::nbits<MAX_HEIGHT>::val,
    WIDTH_BITS = ac::nbits<MAX_WIDTH>::val,
    IN_FMAP_BITS = ac::nbits<IN_FMAP>::val,
    IMG_SIZE = MAX_HEIGHT*MAX_WIDTH,    // Max feature map size
    MEM_SIZE_BITS = ac::nbits<MEM_SIZE>::val
  };
public:
  typedef ac_int<HEIGHT_BITS,false> HEIGHT_TYPE;
  typedef ac_int<WIDTH_BITS,false> WIDTH_TYPE;
  typedef ac_int<IN_FMAP_BITS,false> IN_FMAP_TYPE;
  typedef ac_int<MEM_SIZE_BITS,false> OFFSET_TYPE;
  typedef ac_fixed<DTYPE::width,DTYPE::i_width,true,AC_RND,AC_SAT> SAT_TYPE;

public:
  upscaleAlg() {}

  void run(DTYPE in_fmaps[MEM_SIZE],
           DTYPE out_fmaps[MEM_SIZE],
           IN_FMAP_TYPE num_in_fmaps,
           HEIGHT_TYPE height,
           WIDTH_TYPE width,
           OFFSET_TYPE read_offset,
           OFFSET_TYPE write_offset) {
    SAT_TYPE data;

    IFM: for (int ifm=0; ifm<IN_FMAP; ifm++) { // Input feature map
      ZERO_I: for (int r=0; r<MAX_HEIGHT; r++) { // Process upscaled map
        ZERO_J: for (int c=0; c<MAX_WIDTH; c++) {
          out_fmaps[write_offset + ifm*4*height*width - 2*ifm*height - 2*ifm*width + ifm + r*2*width-r + c] = SAT_TYPE(0.0);
          if (c == 2*width-2) { break; }
        }
        if (r == 2*height-2) { break; }
      }
  
      int offset=0;
      ROW: for (int r=0; r<MAX_HEIGHT; r++) { // Process feature map
        COL: for (int c=0; c<MAX_WIDTH; c++) {
          data = in_fmaps[read_offset + ifm*height*width + r*width + c];
          out_fmaps[write_offset + ifm*4*height*width - 2*ifm*height - 2*ifm*width + ifm + 2*r*width + 2*c + offset] = data;
          if (c == width-1) { offset += 2*width - 2; break; }
        }
        if (r == height-1) { break; }
      }
      if (ifm == num_in_fmaps-1) { break; }
    }
  }
};
#endif
