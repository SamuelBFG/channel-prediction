import argparse
import os


global SHOW_PLOTS, CONV_WIDTH, CFG_L1, DATA_PATH, \
CFG_L2, INPUT_WIDTH, OUT_STEPS, SHIFT, \
MAX_EPOCHS, BATCHSIZE, DROPOUT, NORM, FIGURES_DIR

# =====================================================================
#          PARSE parameters from shell file or terminal script
# =====================================================================

parser = argparse.ArgumentParser() #create parser object
#define each argument / input

parser.add_argument("--input_width", help="specify input data width", type=int) 
parser.add_argument("--out_steps", help="specify number of steps in the future to predict", type=int) 
parser.add_argument("--shift", help="specify offset", type=int) 
args = parser.parse_args()
# args, unknown = parser.parse_known_args() 

# # ==================================================
# #              Define program variables
# # ==================================================

# # when specifying zero vaules, the default will be set 
# # A 'hard coded' varible will need set to set zero values

# INPUT_WIDTH
if args.input_width:
    print('Specified INPUT_WIDTH:', args.input_width)
    INPUT_WIDTH = args.input_width
else:
    print('No specified INPUT_WIDTH - default: 30')
    INPUT_WIDTH = 25

# OUT_STEPS
if args.out_steps:
    print('Specified OUT_STEPS:', args.out_steps)
    OUT_STEPS = args.out_steps
else:
    print('No specified OUT_STEPS - default: 15')
    OUT_STEPS = 4

# INPUT_WIDTH = 50
# OUT_STEPS = 33
SHOW_PLOTS = False
DATA_PATH = "/content/test/pathBA_SSF_dB_AP4_downsampled2Khz_win100.txt" # mmWave AP4 - PATH: BA
# DATA_PATH = "test/fast_fading_dB_NLOS_Head_Indoor_downsampled100hz_n50.txt" # D2D for testing purposes
CFG_L1 = [5, 32, 64, 128, 256, 512] # hidden units layer 1
# CFG_L2 = [1, 5, 10, 25, 50, 100, 200, 500] # hidden units layer 2
CFG_L2 = [] # declare this variable as an empty list for one-layer model
CONV_WIDTH = 5
SHIFT = OUT_STEPS
MAX_EPOCHS = 50
BATCHSIZE = 32
DROPOUT = 0.3

# Types of normalization/scaling implemented here
# NORM = 0 #standardization
# NORM = 1 #centred mean and min = -1
# NORM = 2 #minmax
# Chosen min-max normalisation for the paper
NORM = 2

FIGURES_DIR = '/home/nidhisimmons/git/channel_prediction_2022/mmwave/'#+str(MODEL)+'_input_'+str(INPUT_WIDTH)+'_output_'+str(OUT_STEPS)
if not os.path.isdir(FIGURES_DIR):
  os.makedirs(FIGURES_DIR)



