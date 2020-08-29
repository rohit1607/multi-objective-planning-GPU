#ifndef _MOVE_AND_REWARDS_H
#define _MOVE_AND_REWARDS_H


extern __device__ float calculate_reward_const_dt(float* xs, float* ys, int32_t i_old, int32_t j_old, float xold, float yold, int32_t* newposids, float* params, float vnet_x, float vnet_y );

extern __device__ void move(float ac_speed, float ac_angle, float vx, float vy, int32_t T, float* xs, float* ys, int32_t* posids, float* params, float* r );

extern __device__ float calculate_one_step_reward(float ac_speed, float ac_angle, float rad1, float rad2, float* params);


#endif