#include "top.h"
#include "stdio.h"
#include "conv2d.h"
#include "relu.h"
int main(){

    //dt a[LEN] = {1.1.,2.3,0.3,-1.1.,0.8,-3.1.,-0.4,0.42,1.,-2.2}; //test me dianusma
    //dt b[LEN];


    //top(a, b);
	

	
	conv2d(in,b,28,28,1,6,3); 		//to encoder kommati
	relu(b,26,26,6,sizeof(b));
	maxpool2d(b,c,6,26,26,2);
	conv2d(c,d,12,12,6,16,3); 
	relu(d,10,10,16,sizeof(c));
    maxpool2d(d,out,16,10,10,2);
	
    for (int i=0; i<LEN; i++){
        //printf("b[%d]: %f\n", i, b[i].to_double());
    }
    return 0;
}
