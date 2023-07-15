# Establish the location of this script and use it to reference all
# other files in this example
set sfd [file dirname [info script]]

# Reset the options to the factory defaults
options defaults

options set Input/SearchPath $sfd/../include
options set ComponentLibs/SearchPath . -append
options set Interface/DefaultClockPeriod 2
options set Input/CppStandard c++11
options set Input/CompilerFlags -DCONNECTIONS_ACCURATE_SIM
options set Output/GenerateCycleNetlist false

# Start a new project for option changes to take place
#project new

flow package require /SCVerify
flow package option set /SCVerify/USE_MSIM true
flow package option set /SCVerify/USE_NCSIM true
flow package option set /SCVerify/USE_VCS true
flow package option set /SCVerify/INVOKE_ARGS "$sfd/../file_io"
flow package require /QuestaSIM
flow package option set /QuestaSIM/SCCOM_OPTS {-O3 -x c++ -Wall -Wno-unused-label -Wno-unknown-pragmas}
flow package option set /QuestaSIM/MSIM_AC_TYPES false

solution file add $sfd/testbench.cpp  -type C++

directive set -ARRAY_INDEX_OPTIMIZATION true
directive set -ASSUME_ARRAY_INDEX_IN_RANGE true

go compile

solution library add nangate-45nm_beh -file {$MGC_HOME/pkgs/siflibs/nangate/nangate-45nm_beh.lib} -- -rtlsyntool OasysRTL

solution library add ccs_sample_mem -file {$MGC_HOME/pkgs/siflibs/ccs_sample_mem.lib}
#solution library add ccs_ramifc_w_handshake
go libraries
directive set -CLOCKS {clk {-CLOCK_PERIOD 10 -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME 1.0 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND async -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low -ENABLE_ACTIVE high}}
go assembly
directive set /conv2d/run/K_X0 -PIPELINE_INIT_INTERVAL 1
directive set /conv2d/run/COL -PIPELINE_INIT_INTERVAL 1
directive set /conv2d/run/K_X -UNROLL yes
directive set /conv2d/run/K_Y -UNROLL yes
directive set /conv2d/run/line_buffers:rsc -BLOCK_SIZE 28
directive set /conv2d/run/COL_CPY -PIPELINE_INIT_INTERVAL 1
go architect
go extract
