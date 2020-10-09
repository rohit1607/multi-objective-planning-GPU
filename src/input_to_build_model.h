//Prob type: time, energy1, energy2, energy3
std::string prob_type = "energy1"; //verify num_ac_speeds 
std::string prob_name = "AF_DG";

int32_t nt = 80;
float dt = 1;
int32_t gsize = 50;
float dx = 0.4; float dy = dx;

float x0 = dx/2;q
float y0 = dy/2;

float F = 0.4;
int num_ac_speeds = 3; //verify prob_type
int num_ac_angles = 16;
int32_t num_actions = num_ac_speeds*num_ac_angles;

int32_t nrzns = 100;     // verify with probname
int32_t bDimx = nrzns;

float r_outbound = -100;
float r_terminal = 1000;

// i_term and j_term are (i,j) coords for the TOP LEFT CORNER
// of the square subgrid that constitutes the terminal states
int32_t i_term = 7;    // verify if within grid
int32_t j_term = 7;    // verify if within grid
// int32_t i_term = 50;
// int32_t j_term = 90;
// int32_t i_term = 100; //50
// int32_t j_term = 180; //90

int term_subgrid_size = 2; //number of cells al
float nmodes = 4;   // verify: same as in velocity field

// extra parameters for scalar field like radiation
float rad_nmodes = 1;
float rad_nrzns = 1;

int32_t is_stationary = 0;  // 0 is false

float z = -9999;