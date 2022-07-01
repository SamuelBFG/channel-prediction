import argparse
import os


global SHOW_PLOTS, MODEL, CFG_L1, DATA_PATH, \
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

# LSTM_SIZE_1
# if args.lstm_size_1:
#     print('Specified LSTM_SIZE_1:', args.lstm_size_1)
#     LSTM_SIZE_1 = args.lstm_size_1
# else:
#     print('No specified LSTM_SIZE_1 - default: 50')
#     LSTM_SIZE_1 = 50 #50,100,200

# # LSTM_SIZE_2
# if args.lstm_size_2:
#     print('Specified LSTM_SIZE_2:', args.lstm_size_2)
#     LSTM_SIZE_2 = args.lstm_size_2
# else:
#     print('No specified LSTM_SIZE_2 - default: 0')
#     LSTM_SIZE_2 = 50 #50,100,200

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

SHOW_PLOTS = True
DATA_PATH = "pathAB_SSF_dB_AP1_downsampled2Khz_win100.txt"


# TRAIN_STARTINDEX = 0
# TEST_ENDINDEX = 18113
# MODEL = 'LSTM'
CFG_L1 = [50, 100, 200] # hidden units layer 1
# CFG_L2 = [1, 5, 10, 25, 50, 100, 200, 500] # hidden units layer 2
CFG_L2 = [] # declare this variable as an empty list for one-layer model
# INPUT_WIDTH = 50
# OUT_STEPS = 33
SHIFT = OUT_STEPS
MAX_EPOCHS = 50

## DON'T CHANGE
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