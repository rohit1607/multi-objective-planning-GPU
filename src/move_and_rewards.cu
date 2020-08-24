#include "move_and_rewards.h"
#include "get_funcs.h"
#include <cmath>


__device__ float calculate_reward_const_dt(float* xs, float* ys, int32_t i_old, int32_t j_old, float xold, float yold, int32_t* newposids, float* params, float vnet_x, float vnet_y ){
    // xold and yold are centre of old state (i_old, j_old)
    int gsize = params[0];
    float dt = params[4];
    float r1, r2, theta1, theta2, theta, h;
    float dt_new;
    float xnew, ynew;
    if (newposids[0] == i_old && newposids[1] == j_old)
        dt_new = dt;
    else
    {
        get_xypos_from_ij(newposids[0], newposids[1], gsize, xs, ys, &xnew, &ynew); //get centre of new states
        h = sqrtf((xnew - xold)*(xnew - xold) + (ynew - yold)*(ynew - yold));
        r1 = h/(sqrtf((vnet_x*vnet_x) + (vnet_y*vnet_y)));
        theta1 = get_angle_in_0_2pi(atan2f(vnet_y, vnet_x));
        theta2 = get_angle_in_0_2pi(atan2f(ynew - yold, xnew - xold));
        theta = fabsf(theta1 -theta2);
        r2 = fabsf(sinf(theta));
        dt_new = r1 + r2;
        if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
        {
            params[24] = r1;
            params[25] = r2;
        }
    }
    return -dt_new;
}


__device__ float calculate_one_step_reward(float ac_speed, float ac_angle, float* xs, float* ys, int32_t i_old, int32_t j_old, 
    float xold, float yold, int32_t* newposids, float* params, float vnet_x, float vnet_y){

        int method = params[13];
        float dt = params[4];

        if (method == 0)    //time
            return -dt;

        else if (method == 1){   //energy1
            return -ac_speed*ac_speed;
        } 

        else
            return 0;

    }

__device__ void move(float ac_speed, float ac_angle, float vx, float vy, int32_t T, float* xs, float* ys, int32_t* posids, float* params, float* r ){
    int32_t gsize = params[0];
    int32_t n = params[0] - 1;      // gsize - 1
    // int32_t num_actions = params[1];
    // int32_t nrzns = params[2];
    // float F = params[3];
    float F = ac_speed;
    float dt = params[4];
    float r_outbound = params[5];
    float r_terminal = params[6];
    // int32_t nT = params[10];
    float Dj = fabsf(xs[1] - xs[0]);
    float Di = fabsf(ys[1] - ys[0]);
    float r_step = 0;
    *r = 0;
    int32_t i0 = posids[0];
    int32_t j0 = posids[1];
    float vnetx = F*cosf(ac_angle) + vx;
    float vnety = F*sinf(ac_angle) + vy;
    float x, y;
    get_xypos_from_ij(i0, j0, gsize, xs, ys, &x, &y); // x, y stores centre coords of state i0,j0
    float xnew = x + (vnetx * dt);
    float ynew = y + (vnety * dt);
    
    //checks TODO: remove checks once verified
    // if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
    // {
    //     params[14] = x;
    //     params[15] = y;
    //     params[16] = vnetx;
    //     params[17] = vnety;
    //     params[18] = xnew;
    //     params[19] = ynew;
    //     params[20] = ac_angle;
    // }
    if (xnew > xs[n])
        {
            xnew = xs[n];
            *r += r_outbound;
        }
    else if (xnew < xs[0])
        {
            xnew = xs[0];
            *r += r_outbound;
        }
    if (ynew > ys[n])
        {
            ynew =  ys[n];
            *r += r_outbound;
        }
    else if (ynew < ys[0])
        {
            ynew =  ys[0];
            *r += r_outbound;
        }
    // TODO:xxDONE check logic wrt remainderf. remquof had issue
    int32_t xind, yind;
    //float remx = remquof((xnew - xs[0]), Dj, &xind);
    //float remy = remquof(-(ynew - ys[n]), Di, &yind);
    float remx = remainderf((xnew - xs[0]), Dj);
    float remy = remainderf(-(ynew - ys[n]), Di);
    xind = ((xnew - xs[0]) - remx)/Dj;
    yind = (-(ynew - ys[n]) - remy)/Di;
    if ((remx >= 0.5 * Dj) && (remy >= 0.5 * Di))
        {
            xind += 1;
            yind += 1;
        }
    else if ((remx >= 0.5 * Dj && remy < 0.5 * Di))
        {
            xind += 1;
        }
    else if ((remx < 0.5 * Dj && remy >= 0.5 * Di))
        {
            yind += 1;
        }
    if (!(my_isnan(xind) || my_isnan(yind)))
        {
            posids[0] = yind;
            posids[1] = xind;
            if (is_edge_state(posids[0], posids[1]))     //line 110
                {
                    *r += r_outbound;
                }
            
            if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
            {
                // params[26] = 9999;
            }
        }

    r_step = calculate_one_step_reward(ac_speed, ac_angle, xs, ys, i0, j0, x, y, posids, params, vnetx, vnety);
    // r_step = -dt;
    *r += r_step; //TODO: numerical check remaining
    if (is_terminal(posids[0], posids[1], params))
        {
            *r += r_terminal;
        }
    
    // if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
    // {
    //     params[19] = xnew;
    //     params[20] = ynew;
    //     params[21] = yind;
    //     params[22] = xind;
    //     // params[23] = *r;
    //     //params[17] = ynew;
    //     //params[18] = ac_angle;
    // }
}
  

