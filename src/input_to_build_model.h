
std::string prob_type = "energy1"; //time, energy1, energy2, energy3
std::string prob_name = "hw_g10x10x10_r5";

int32_t nt = 10;
float dt = 1;
int32_t gsize = 10;
float dx = 1; float dy = 1;
float x0 = dx/2;
float y0 = dy/2;
int num_ac_speeds = 2;
int num_ac_angles = 8;
int32_t num_actions = num_ac_speeds*num_ac_angles;
int32_t nrzns = 5;
int32_t bDimx = nrzns;
float F = 1;
float r_outbound = -100;
float r_terminal = 100;
// i_term and j_term are (i,j) coords for the TOP LEFT CORNER
// of the square subgrid that constitutes the terminal states
int32_t i_term = 6;
int32_t j_term = 7;
// int32_t i_term = 50;
// int32_t j_term = 90;
// int32_t i_term = 100; //50
// int32_t j_term = 180; //90
int term_subgrid_size = 1; //number of cells al
float nmodes = 1;

int32_t is_stationary = 0;

float z = -9999;