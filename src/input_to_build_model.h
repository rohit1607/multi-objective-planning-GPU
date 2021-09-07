//Prob type: time, energy1, energy2, energy3

std::string prob_type = "custom1"; //verify num_ac_speeds 
std::string prob_name = "DG3_g200x200x200_r5k_DynObs";

float alpha_header = 0.05; // 1 for only energy, 0 for only time
// extern float alpha;

int32_t nt = 200;
float dt = 1;
int32_t gsize = 200;

float dx = 1; float dy = dx;

float x0 = dx/2; //explain later
float y0 = dy/2;

float F = 2;
int num_ac_speeds = 2; //verify prob_type
int num_ac_angles = 16;
int32_t num_actions = num_ac_speeds*num_ac_angles;

int32_t nrzns = 5000;     // verify with probname
int32_t bDimx = 64; // based on optimum threeads per block obtained after multiple test runs

float r_outbound = -1000;
float r_terminal = 100;

// i_term and j_term are (i,j) coords for the TOP LEFT CORNER
// of the square subgrid that constitutes the terminal states
// int32_t i_term = (int)(0.8*gsize);    // verify if within grid
// int32_t j_term = (int)(0.8*gsize);    // verify if within grid
int32_t i_term = (int)(0.15*gsize);    // verify if within grid
int32_t j_term = (int)(0.4*gsize);    // verify if within grid
// int32_t i_term = 50;
// int32_t j_term = 90;
// int32_t i_term = 100; //50
// int32_t j_term = 180; //90

int term_subgrid_size = 2; //number of cells al
float nmodes = 4;   // verify: same as in velocity field

// size of nighbourhood grid- should always be odd
float neighb_gsize = 21;

int32_t is_stationary = 0;  // 0 is false

float z = -9999;