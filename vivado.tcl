# Establish the location of this script and use it to reference all
# other files in this example
set sfd [file dirname [info script]]
# Reset the options to the factory defaults
solution new -state initial
solution options defaults

solution options set /ComponentLibs/SearchPath . -append
solution options set /Interface/DefaultClockPeriod 2
solution options set /Input/CppStandard c++11
solution options set /Input/CompilerFlags -DCONNECTIONS_ACCURATE_SIM
solution options set /Output/GenerateCycleNetlist false
solution options set /Input/SearchPath ../fpgaAE/include/

# Start a new project for option changes to take place
project new


flow package require /SCVerify
flow package option set /SCVerify/USE_MSIM true
flow package option set /SCVerify/USE_NCSIM true
flow package option set /SCVerify/USE_VCS true
flow package option set /SCVerify/INVOKE_ARGS ../fpgaAE/file_io/
flow package require /QuestaSIM
flow package option set /QuestaSIM/SCCOM_OPTS {-O3 -x c++ -Wall -Wno-unused-label -Wno-unknown-pragmas}
flow package option set /QuestaSIM/MSIM_AC_TYPES false


solution file add $sfd/testbench.cpp  -type C++ -exclude true
solution file add $sfd/conv2d.h -type C++

directive set -ARRAY_INDEX_OPTIMIZATION true
directive set -ASSUME_ARRAY_INDEX_IN_RANGE true

go compile

solution library add mgc_Xilinx-ZYNQ-uplus-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family ZYNQ-uplus -speed -2 -part xczu7ev-ffvc1156-2-e
solution library add amba
solution library add Xilinx_FIFO
go libraries
directive set -CLOCKS {clk {-CLOCK_PERIOD 10 -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME 1.0 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND async -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low -ENABLE_ACTIVE high}}
go assembly
directive set /conv2d/run/line_buffers:rsc -BLOCK_SIZE 28
