import datetime
from gurobipy import *
from itertools import product
import numpy as np

def new_model():
    env = Env()
    env.setParam(GRB.Param.LogToConsole, 0)
    env.setParam(GRB.Param.OptimalityTol, 1.0e-4)
    env.setParam(GRB.Param.FeasibilityTol, 1.0e-4)
    env.setParam(GRB.Param.MIPGapAbs, 1.0e-4)
    env.setParam(GRB.Param.Method, 3)
    env.setParam(GRB.Param.Presolve, 2)
    model = Model("lp_model", env)
    return model

def creat_var(lp_model, data, epsilon):
    start = datetime.datetime.now()
    input_shape = data.shape
    print('x input_shape: ', input_shape)
    x = lp_model.addVars(*input_shape, obj=1.0, vtype=GRB.CONTINUOUS, name="input_vars")
    ub, lb = np.clip(data + epsilon, 0, 1), np.clip(data - epsilon, 0, 1)
    if input_shape != ub.shape:
        raise Exception("input_shape is different with the dimension of image")
    ndim = ub.ndim
    if ndim == 1:
        h = input_shape[0]
        iter_item = range(h)
    elif ndim == 3:
        h, w, c = input_shape
        iter_item = product(range(h), range(w), range(c))
    else:
        raise Exception("not support dimenstion: ", ndim)

    # set upper and lower bound to variable
    for idx in iter_item:
        x[idx].setAttr(GRB.Attr.LB, lb[idx])
        x[idx].setAttr(GRB.Attr.UB, ub[idx])

    lp_model.update()
    
    end = datetime.datetime.now()
    print('create variable time: ', (end-start).seconds)
    return lp_model, x

def get_solution_value(lp_model, x, shape, A_u, A_l, B_u, B_l, pad, stride, p_n, eps):
    
    y = lp_model.addVars(*(A_u.shape[0], A_u.shape[1], A_u.shape[2]), obj=1.0, vtype=GRB.CONTINUOUS, name="output_vars")
    
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    
    # print('A_u.shape: ', A_u.shape)
    # print('A_l.shape: ', A_l.shape)
    # print('A_u.shape[0]:', A_u.shape[0])
    # print('A_u.shape[1]:',A_u.shape[1])
    # print('A_u.shape[2]:',A_u.shape[2])
    # print('A_u.shape[3]:',A_u.shape[3])
    # print('A_u.shape[4]:',A_u.shape[4])
    # print('A_u.shape[5]:',A_u.shape[5])
    # print('shape:', shape)
    
    for a in range(A_u.shape[0]):
        for b in range(A_u.shape[1]):
            for c in range(A_u.shape[2]):
                for i in range(A_u.shape[3]):
                    for j in range(A_u.shape[4]):
                        for k in range(A_u.shape[5]):
                            if 0<=s_h*a+i-p_hl<shape[0] and 0<=s_w*b+j-p_wl<shape[1]:
                                y[a,b,c] += A_l[a,b,c,i,j,k]*x[s_h*a+i-p_hl,s_w*b+j-p_wl,k]
    

    for a in range(A_u.shape[0]):
        for b in range(A_u.shape[1]):
            for c in range(A_u.shape[2]):
                y[a,b,c] = y[a,b,c] + B_l[a,b,c]
    
    lp_model.update()

    i,j,k = A_u.shape[0]-1, A_u.shape[1]-1, A_u.shape[2]-1
    lp_model.setObjective(y[i,j,k], GRB.MINIMIZE)
    lp_model.optimize()
    
    if lp_model.status == GRB.OPTIMAL:
        min_value = lp_model.objVal
        print('Optimal objective: %g' % min_value)
        if min_value <= 0:
            ndim = len(shape)
            if ndim == 1:
                h = shape[0]
                iter_item = range(h)
            elif ndim == 3:
                h, w, c = shape
                iter_item = product(range(h), range(w), range(c))
            else:
                raise Exception("input_shape dimension abnormal")
            values = np.zeros(shape)
            for idx in iter_item:
                values[idx] = x[idx].x
            return values, min_value
        return None, min_value
    elif lp_model.status != GRB.INFEASIBLE:
        print('Optimization was stopped with status %d' % lp_model.status)
        return None, None
    
