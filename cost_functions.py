from helpers import logistic, to_equation, differentiate, nearest_approach_to_any_vehicle, get_f_and_N_derivatives
from constants import *
import numpy as np
# COST FUNCTIONS
def time_diff_cost(traj, target_vehicle, delta, T, predictions):
    # print("time_diff_cost: ", T)
    """
    Penalizes trajectories that span a duration which is longer or 
    shorter than the duration requested.
    """
    _, _, t = traj
    return logistic(float(abs(t-T)) / T)

def s_diff_cost(traj, target_vehicle, delta, T, predictions):
    print("s_diff_cost: ", T)
    """
    Penalizes trajectories whose s coordinate (and derivatives) 
    differ from the goal.
    """
    s, _, T = traj
    target = predictions[target_vehicle].state_in(T)
    target = list(np.array(target) + np.array(delta))
    s_targ = target[:3]
    S = [f(T) for f in get_f_and_N_derivatives(s, 2)]
    cost = 0
    for actual, expected, sigma in zip(S, s_targ, SIGMA_S):
        diff = float(abs(actual-expected))
        cost += logistic(diff/sigma)
    return cost

def d_diff_cost(traj, target_vehicle, delta, T, predictions):
    print("d_diff_cost: ", T)
    """
    Penalizes trajectories whose d coordinate (and derivatives) 
    differ from the goal.
    """
    _, d_coeffs, T = traj
    
    d_dot_coeffs = differentiate(d_coeffs)
    d_ddot_coeffs = differentiate(d_dot_coeffs)

    d = to_equation(d_coeffs)
    d_dot = to_equation(d_dot_coeffs)
    d_ddot = to_equation(d_ddot_coeffs)

    D = [d(T), d_dot(T), d_ddot(T)]
    
    target = predictions[target_vehicle].state_in(T)
    target = list(np.array(target) + np.array(delta))
    d_targ = target[3:]
    cost = 0
    for actual, expected, sigma in zip(D, d_targ, SIGMA_D):
        diff = float(abs(actual-expected))
        cost += logistic(diff/sigma)
    return cost

def collision_cost(traj, target_vehicle, delta, T, predictions):
    print("collision_cost: ", T)
    """
    Binary cost function which penalizes collisions.
    """
    nearest = nearest_approach_to_any_vehicle(traj, predictions)
    if nearest < 2*VEHICLE_RADIUS: return 1.0
    else : return 0.0

def buffer_cost(traj, target_vehicle, delta, T, predictions):
    print("buffer_cost: ", T)
    """
    Penalizes getting close to other vehicles.
    """
    nearest = nearest_approach_to_any_vehicle(traj, predictions)
    return logistic(2*VEHICLE_RADIUS / nearest)
    
def stays_on_road_cost(traj, target_vehicle, delta, T, predictions):
    print("stays_on_road_cost: ", T)
    pass

def exceeds_speed_limit_cost(traj, target_vehicle, delta, T, predictions):
    print("exceeds_speed_limit_cost: ", T)
    pass

def efficiency_cost(traj, target_vehicle, delta, T, predictions):
    print("efficiency_cost: ", T)
    """
    Rewards high average speeds.
    """
    s, _, t = traj
    s = to_equation(s)
    avg_v = float(s(t)) / t
    targ_s, _, _, _, _, _ = predictions[target_vehicle].state_in(t)
    targ_v = float(targ_s) / t
    return logistic(2*float(targ_v - avg_v) / avg_v)

# calculate log of (average acceleration per second/ desired acceleration per second)
def total_accel_cost(traj, target_vehicle, delta, T, predictions):
    # print("total_accel_cost: ", T)
    s, d, t = traj
    s_dot = differentiate(s)
    s_d_dot = differentiate(s_dot)
    a = to_equation(s_d_dot)
    total_acc = 0
    dt = float(T) / 100.0
    for i in range(100):
        t = dt * i
        acc = a(t)
        # total_acc += abs(acc*dt)
        total_acc += abs(acc)
    # acc_per_second = total_acc / T
    acc_per_second = total_acc / 100 # average
    return logistic(acc_per_second / EXPECTED_ACC_IN_ONE_SEC )

# check if acceleration ever goes over legal limit
def max_accel_cost(traj, target_vehicle, delta, T, predictions):
    # print("max_accel_cost: ", T)
    s, d, t = traj
    s_dot = differentiate(s)
    s_d_dot = differentiate(s_dot)
    a = to_equation(s_d_dot)
    all_accs = [a(float(T)/100 * i) for i in range(100)]
    max_acc = max(all_accs, key=abs)
    if abs(max_acc) > MAX_ACCEL: return 1
    else: return 0
    

# check if the jerk along the path is all below the allowed limit
def max_jerk_cost(traj, target_vehicle, delta, T, predictions):
    # print("max_jerk_cost: ", T)
    s, d, t = traj
    s_dot = differentiate(s)
    s_d_dot = differentiate(s_dot)
    jerk = differentiate(s_d_dot)
    jerk = to_equation(jerk)
    all_jerks = [jerk(float(T)/100 * i) for i in range(100)]
    max_jerk = max(all_jerks, key=abs)
    if abs(max_jerk) > MAX_JERK: return 1
    else: return 0

# calculate log of (average jerk per second / desired jerkper second)
def total_jerk_cost(traj, target_vehicle, delta, T, predictions):
    # print("total_jerk_cost: ", T)
    s, d, t = traj
    s_dot = differentiate(s) # velocity
    s_d_dot = differentiate(s_dot) # acceleration
    jerk = to_equation(differentiate(s_d_dot)) # jerk
    total_jerk = 0
    dt = float(T) / 100.0 # .05
    for i in range(100): # 0 to 99
        t = dt * i
        # print("t:", t)
        j = jerk(t) # jerk at time t
        # print("jerk: ", j)
        # print("abs: ", abs(j*dt))
        # total_jerk += abs(j*dt)
        # print("abs: ", abs(j))
        total_jerk += abs(j)
    # jerk_per_second = total_jerk / T
    jerk_per_second = total_jerk / 100 # average jerk
    # print("jerk per second: ", jerk_per_second)
    return logistic(jerk_per_second / EXPECTED_JERK_IN_ONE_SEC )