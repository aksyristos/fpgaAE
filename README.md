# Characteristics
Input is a 28x28 mnist image, target: 7, unnormalised

2 layers enc/dec (4 in total)

can be modified as Denoising AE

can be imported to HLS tools (catapult in this case)

-forked from https://github.com/hlslibs/ac_ml

# PY model specs
python 3.8.8

torch 1.10.1

torchvision 0.11.2



# How to build
-Source the "set_vars" script file to download and build the various open-source components used in these examples.

  Example for C-Shell users:
  
    source set_vars.csh
    
  Example for Bourne Shell users:
  
    . set_vars.sh
    
   

-To compile and execute the design do: "make all"


-Use "make help" for a list of options
