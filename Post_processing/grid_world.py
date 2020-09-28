import itertools
from utils.custom_functions import calculate_reward_const_dt, get_minus_dt
import numpy as np
import math
from definition import Sampling_interval

Pi = math.pi

class Grid:
    def __init__(self, xs, ys, term_subgrid_size, dt, nt, start, end):
        self.nj = len(xs)
        self.ni = len(ys)
        self.nt = nt
        # print("shapes=", len(xs), len(ys))

        self.dj = np.abs(xs[1] - xs[0])
        self.di = np.abs(ys[1] - ys[0])
        self.dt = dt
        # print("diffs=", self.dj, self.di)

        self.xs = xs
        self.ys = ys
        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0]

        self.tsg_size = term_subgrid_size

        self.x = xs[start[1]]
        self.y = ys[self.ni - 1 - start[0]]

        # t, i, j , start and end store indices!!
        self.t = 0
        self.i = int(start[0])
        self.j = int(start[1])

        self.endpos = end
        self.startpos = start
        self.start_state = (0, start[0], start[1])
        # self.edge_states = self.edge_states()

        self.r_outbound = -1
        self.r_terminal = 1
        # TODO: CHANGE ACCORDINGLY. TESTING -1 REWARD> Also change in move()
        self.r_otherwise = calculate_reward_const_dt
        # self.r_otherwise = get_minus_dt

        self.reward_structure = ['oubound_penalty = '+ str(self.r_outbound), 'Terminal_Reward =' + str(self.r_terminal), 'General_reward: ' + self.r_otherwise.__name__]
        # self.bcrumb_states = set()
        self.bcrumb_dict = dict.fromkeys([i for i in range(5000)])


    def compute_cell(self, s):
        # s is (x, y)
        remx = (s[0] - self.xs[0]) % self.dj
        remy = -(s[1] - self.ys[-1]) % self.di
        xind = (s[0] - self.xs[0]) // self.dj
        yind = -(s[1] - self.ys[-1]) // self.di

        if remx >= 0.5 * self.dj and remy >= 0.5 * self.di:
            xind+=1
            yind+=1
        elif remx >= 0.5 * self.dj and remy < 0.5 * self.di:
            xind+=1
        elif remx < 0.5 * self.dj and remy >= 0.5 * self.di:
            yind+=1

        return int(yind), int(xind)


    def make_bcrumb_dict(self, paths, train_id_list):
        # initialise bcrumb_dict as dictionary of sets, with key = rzn, value = set of states
        for key in self.bcrumb_dict.keys():
            self.bcrumb_dict[key] = set()


        for k in train_id_list:
            trajectory = paths[k, 0] #kth trajectory, array of shape (n,2)

            # append starting point to traj
            s_t = int(0)
            for i in range(0, len(trajectory), Sampling_interval):
                s_i, s_j = self.compute_cell( trajectory[i])         # compute indices corrsponding to first coord  
                self.bcrumb_dict[k].add((s_t, s_i, s_j))
                s_t += 1

            # add last point to the trajectories
            s_i, s_j = self.compute_cell( trajectory[-1])
            self.bcrumb_dict[k].add((s_t, s_i, s_j))

        return


    # def make_bcrumb_set(self, paths, train_id_list):

    #     for k in train_id_list:
    #         trajectory = paths[k, 0] #kth trajectory, array of shape (n,2)

    #         # append starting point to traj
    #         s_t = int(0)
    #         state_traj = []
    #         for i in range(0, len(trajectory), Sampling_interval):
    #             s_i, s_j = self.compute_cell( trajectory[i])         # compute indices corrsponding to first coord  
    #             self.bcrumb_states.add((s_t, s_i, s_j))
    #             s_t += 1

    #         # add last point to the trajectories
    #         s_i, s_j = self.compute_cell( trajectory[-1])
    #         self.bcrumb_states.add((s_t, s_i, s_j))



    # Rewards and Actions to be dictionaries
    def set_AR(self, Actions):
        self.actions = Actions
        # self.rewards= Rewards


    # explicitly set state. state is a tuple of indices(m,n,p)
    def set_state(self, state, xcoord=None, ycoord=None):
        self.t = state[0]
        self.i = state[1]
        self.j = state[2]
  
        self.x = self.xs[self.j]
        self.y = self.ys[self.ni - 1 - self.i]

        if xcoord != None and ycoord != None:
            self.x = xcoord
            self.y = ycoord


    def current_state(self):
        return (int(self.t), int(self.i), int(self.j))

    def current_pos(self):
        return (int(self.i), int(self.j))


    # MAY NEED TO CHANGE DEFINITION
    def is_terminal(self, query_s = None, almost = None):
        # return self.actions[state]==None
        if query_s == None:
            if almost == True:
                end_i, end_j = self.endpos
                accepted_i = [end_i, end_i-1, end_i+1]
                accepted_j = [end_j, end_j-1, end_j+1]
                i, j = self.current_pos()
                if i in accepted_i and j in accepted_j:
                    return True                
            else:
                i, j = self.current_pos()
                i_term, j_term = self.endpos
                tsg_size = self.tsg_size
                return (i >= i_term and i < i_term + tsg_size and j >= j_term and j < j_term + tsg_size)
 
        else:
            t, i, j = query_s
            return((i,j) == self.endpos)                

    def move_exact(self, action, Vx, Vy, rzn = None):
        r = 0
        if not self.is_terminal() and self.if_within_actionable_time():
            thrust, angle = action
            s0 = self.current_state()
            # print("check: thrust, angle ", thrust, angle)
            # print("self.x, self.y", self.x,self.y)
            vnetx = (thrust * math.cos(angle)) + (Vx)
            vnety = (thrust * math.sin(angle)) + (Vy)
            xnew = self.x + (vnetx * self.dt)
            ynew = self.y + (vnety * self.dt)
            # print("xnew, ynew",xnew,ynew)

            # if state happens to go out of of grid, bring it back inside
            if xnew > self.xs[-1]:
                xnew = self.xs[-1]
                r += self.r_outbound
            elif xnew < self.xs[0]:
                xnew = self.xs[0]
                r += self.r_outbound
            if ynew > self.ys[-1]:
                ynew = self.ys[-1]
                r += self.r_outbound
            elif ynew < self.ys[0]:
                ynew = self.ys[0]
                r += self.r_outbound
            # print("xnew, ynew after boundingbox", xnew, ynew)
            # rounding to prevent invalid keys

            self.x = xnew
            self.y = ynew

            remx = (xnew - self.xs[0]) % self.dj
            remy = -(ynew - self.ys[-1]) % self.di
            xind = (xnew - self.xs[0]) // self.dj
            yind = -(ynew - self.ys[-1]) // self.di


            # print("rex,remy,xind,yind", remx,remy,xind,yind)
            if remx >= 0.5 * self.dj and remy >= 0.5 * self.di:
                xind += 1
                yind += 1
            elif remx >= 0.5 * self.dj and remy < 0.5 * self.di:
                xind += 1
            elif remx < 0.5 * self.dj and remy >= 0.5 * self.di:
                yind += 1
            #print("rex,remy,xind,yind after upate", remx, remy, xind, yind)
            # print("after update, (i,j)", (yind,xind))
            if not (math.isnan(xind) or math.isnan(yind)):
                self.i = int(yind)
                self.j = int(xind)
                if self.if_edge_state((self.i, self.j)):
                    r += self.r_outbound

            s_new = self.current_state()
            if rzn != None: #if you know and have provided the rzn
                if s_new in self.bcrumb_dict[rzn]:
                    r += self.r_terminal/100

            r += self.r_otherwise(self.dt, self.xs, self.ys, s0, s_new, vnetx, vnety, action)
            # r += self.r_otherwise(self.dt) # get_minus_dt()


            if self.is_terminal():
                r += self.r_terminal
            elif not self.if_within_actionable_time(): #not terminal anb not within actionable time
                r += self.r_outbound

            
            # increment time index by 1
            self.t += 1 

        return r


    # !! time to mentioned by index !!
    def ac_state_space(self, time=None):
        """
        returns states where action is allowed to be performed
        :param time: ip: int. if time := k , then function returns all states at time k, i.e. (k, ., .)
        :return: sorted set of states in which action is allowed to be performed
        """
        a=set()
        if time == None:
            for t in range((self.nt) - 1):  # does not include the states in the last time stice.
                for i in range(self.ni):
                    for j in range(self.nj):
                        if ((i,j)!=self.endpos and not self.if_edge_state((i,j)) ):# does not include states with pos as endpos
                            a.add((t,i,j))

        else:
            for i in range(self.ni):
                for j in range(self.nj):
                    if ((i, j) != self.endpos and not self.if_edge_state((i, j))):  # does not include states with pos as endpos
                          a.add((time,i,j))

        return sorted(a)




    def state_space(self):
        a = set()
        for t in range(self.nt):
            for i in range(self.ni):
                for j in range(self.nj):
                    a.add((t,i,j))

        return sorted(a)


    def if_within_time(self):
        return (self.t >= 0 and self.t < self.nt)


    def if_within_actionable_time(self, query_s=None):
        if query_s != None:
            t, i, j = query_s
            return (t >= 0 and t < self.nt - 1)
        else:
            return (self.t >= 0 and self.t < self.nt - 1)


    def if_within_TD_actionable_time(self):
        return (self.t >= 0 and self.t < self.nt - 2*1)


    def if_within_grid(self,s):
        """TODO: check whether boudary to be treate as a part of grid or not depending on its usage."""
        t = s[0]
        i = s[1]
        j = s[2]
        return (j <= (self.nj) - 1 and j >= 0 and i <= (self.ni) - 1 and i >= 0 and t <= (self.nt) - 1 and t >= 0)

    def if_edge_state(self, pos_or_state):
        """
        returns True if state is at the edge
        :param pos:
        :return:
        """
        if len(pos_or_state)==2:
            i = pos_or_state[0]
            j = pos_or_state[1]
        elif len(pos_or_state)==3:
            t = pos_or_state[0]
            i = pos_or_state[1]
            j = pos_or_state[2]
        else:
            assert ( len(pos_or_state)==2 or len(pos_or_state)==3 ), "Wrong input to if_edge_state()"
        if (i == 0) or (i == self.ni - 1) or (j == 0) or (j == self.nj - 1):
            return True
        else:
            return False


"""Class ends here"""


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def populate_ac_speeds(speed_list, num_ac_speeds, Fmax):

    if num_ac_speeds == 1:
        speed_list.append(Fmax)

    elif num_ac_speeds > 1:
        delF = Fmax/(num_ac_speeds - 1)
        for i in range(num_ac_speeds):
            speed_list.append(i*delF)

    return speed_list

    
def timeOpt_grid(xs, ys, term_subgrid_size, dt, nt, Fmax, startpos, endpos, num_ac_speeds=3, num_ac_angles=8):
    g = Grid(xs, ys, term_subgrid_size, dt, nt, startpos, endpos)

    # define actions and rewards
    Pi = math.pi
    # speeds in m/s
    speed_list = []
    speed_list = populate_ac_speeds(speed_list, num_ac_speeds, Fmax)
    # print("in timeOPt_grid: speed list", speed_list)

    angle_list = []
    for i in range(num_ac_angles):
        angle_list.append(round(i * 2 * Pi / num_ac_angles ,14))
   
    # print("in timeOPt_grid: angle list", angle_list)

    action_list = list(itertools.product(speed_list, angle_list))
    # print("in timeOPt_grid: action list", action_list)
    # set actions for grid
    g.set_AR(action_list)

    return g
