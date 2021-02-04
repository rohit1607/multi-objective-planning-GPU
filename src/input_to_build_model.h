//Prob type: time, energy1, energy2, energy3
std::string prob_type = "time"; //verify num_ac_speeds 
std::string prob_name = "test_g70x70x5_r5k";

int32_t nt = 5;
float dt = 1;
int32_t gsize = 70;
float dx = 1; float dy = dx;

float x0 = dx/2;
float y0 = dy/2;

float F = 1;
int num_ac_speeds = 1; //verify prob_type
int num_ac_angles = 16;
int32_t num_actions = num_ac_speeds*num_ac_angles;

int32_t nrzns = 5000;     // verify with probname
int32_t bDimx = nrzns;

float r_outbound = -1000;
float r_terminal = 100;

// i_term and j_term are (i,j) coords for the TOP LEFT CORNER
// of the square subgrid that constitutes the terminal states
int32_t i_term = (int)(0.8*gsize);    // verify if within grid
int32_t j_term = (int)(0.8*gsize);    // verify if within grid
// int32_t i_term = 50;
// int32_t j_term = 90;
// int32_t i_term = 100; //50
// int32_t j_term = 180; //90

int term_subgrid_size = 1; //number of cells al
float nmodes = 1;   // verify: same as in velocity field

// size of nighbourhood grid- should always be odd
float neighb_gsize = 21;

int32_t is_stationary = 0;  // 0 is false

float z = -9999;