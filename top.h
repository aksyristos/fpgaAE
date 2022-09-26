#ifndef TOP
#define TOP

#include "ac_int.h"
#include "ac_fixed.h"
#include "ac_channel.h"
#include "ac_math.h"
#include "ac_array.h"
#include "ac_reg.h"

#define LEN 10

typedef ac_fixed<16,8,false> dt;

void top(dt a[LEN], dt b[LEN]);

#endif
