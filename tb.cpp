#include "top.h"
#include "stdio.h"

int main(){

    dt a[LEN] = {1.1,2.3,0.3,-1.1,0.8,-3.1,-0.4,0.42,1,-2.2};
    dt b[LEN];

    top(a, b);
    
    for (int i=0; i<LEN; i++){
        printf("b[%d]: %f\n", i, b[i].to_double());
    }
    return 0;
}