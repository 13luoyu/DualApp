from numba import njit
import numpy as np

@njit
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

@njit
def sigmoidd(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

@njit
def sigmoidid(x):
    return 2.0*np.arccosh(1.0/(2.0*np.sqrt(x)))

@njit 
def sigmoidut(l, u):
    act = sigmoid
    actd = sigmoidd
    actid = sigmoidid
    upper = u
    lower = 0
    al = act(l)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (act(guess)-al)/(guess-l)
        if guesss >= guesst:
            upper = guess
        else:
            lower = guess
    return upper
    
@njit 
def sigmoidlt(l, u):
    act = sigmoid
    actd = sigmoidd
    actid = sigmoidid
    upper = 0
    lower = l
    au = act(u)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (au-act(guess))/(u-guess)
        if guesss >= guesst:
            lower = guess
        else:
            upper = guess
    return lower

@njit 
def sigmoidup(l, u, k):
    act = sigmoid
    actd = sigmoidd
    upper = u
    lower = max(l,0.0)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            upper = guess
        elif k < guesst:
            lower = guess
        else:
            upper = guess
            break
    return upper

@njit 
def sigmoidlow(l, u, k):
    act = sigmoid
    actd = sigmoidd
    upper = min(u,0.0)
    lower = l
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            lower = guess
        elif k < guesst:
            upper = guess
        else:
            lower = guess
            break
    return lower

# DeepCert

@njit
def deepCert_first_case(UB, LB, act, actd, actup, actlow):
    alpha = (act(UB) - act(LB))/(UB - LB)
    
    alpha_u = alpha
    beta_u = act(LB) - alpha * LB

    d = actlow(LB, UB, alpha)
    alpha_l = actd(d)
    beta_l = act(d) - actd(d) * d
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def deepCert_second_case(UB, LB, act, actd, actup, actlow):
    alpha = (act(UB) - act(LB))/(UB - LB)

    d = actup(LB, UB, alpha)
    alpha_u = actd(d)
    beta_u = act(d) - actd(d) * d
    
    alpha_l = alpha
    beta_l = act(LB) - alpha * LB
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def deepCert_third_case(UB, LB, act, actd, actut, actlt):
    du = actut(LB, UB)
    dus = (act(du) - act(LB))/(du - LB)
    dut = actd(du)
    if dut < dus:
        alpha_u = dut
        beta_u = act(du) - dut * du
    else:
        alpha_u = dus
        beta_u = act(LB) - LB * dus
    dl = actlt(LB, UB)
    dls = (act(dl) - act(UB))/(dl - UB)
    dlt = actd(dl)
    if dlt < dls:
        alpha_l = dlt
        beta_l = act(dl) - dlt * dl
    else:
        alpha_l = dls
        beta_l = act(UB) - UB * dls
    
    return alpha_u, beta_u, alpha_l, beta_l

# Verinet

@njit
def minimal_area_first_case(UB, LB, act, actd):
    # upper bound
    alpha = (act(UB) - act(LB))/(UB - LB)
    alpha_u = alpha
    beta_u = act(LB) - alpha * LB
    # lower bound
    mid_x = (UB + LB)/2
    alpha_l = actd(mid_x)
    beta_l = act(mid_x) - actd(mid_x) * mid_x
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def minimal_area_third_case(UB, LB, act, actd):
    # upper bound
    mid_x = (UB + LB)/2
    alpha_u = actd(mid_x)
    beta_u = act(mid_x) - actd(mid_x) * mid_x
    # lower bounds
    alpha = (act(UB) - act(LB))/(UB - LB)
    alpha_l = alpha
    beta_l = act(LB) - alpha * LB
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def minimal_area_fifth_case(UB, LB, act, actd, actut, actlt):
    # upper bound
    du = actut(LB, UB)
    alpha_du_xl = (act(du) - act(LB))/(du - LB)
    alpha_u = alpha_du_xl
    beta_u = act(du) - alpha_du_xl * du
    # lower bound
    dl = actlt(LB, UB)
    alpha_dl_xu = (act(dl) - act(UB))/(dl - UB)
    alpha_l = alpha_dl_xu
    beta_l = act(dl) - alpha_dl_xu * dl
    
    return alpha_u, beta_u, alpha_l, beta_l

# Newise

@njit
def endpoint_first_case(UB, LB, act, actd):
    # upper bound 
    alpha = (act(UB) - act(LB))/(UB - LB)
    alpha_u = alpha
    beta_u = act(LB) - alpha * LB
    # lower bound
    alpha_l = actd(LB)
    beta_l = act(LB) - actd(LB) * LB

    return alpha_u, beta_u, alpha_l, beta_l

@njit
def endpoint_third_case(UB, LB, act, actd):
    # upper bound 
    alpha_u = actd(UB)
    beta_u = act(UB) - actd(UB) * UB
    # lower bound 
    alpha = (act(UB) - act(LB))/(UB - LB)
    alpha_l = alpha
    beta_l = act(LB) - alpha * LB

    return alpha_u, beta_u, alpha_l, beta_l

@njit
def endpoint_fifth_case(UB, LB, act, actd):
    # upper bound 
    alpha_u = actd(UB)
    beta_u = act(UB) - actd(UB) * UB
    # lower bound 
    alpha_l = actd(LB)
    beta_l = act(LB) - actd(LB) * LB

    return alpha_u, beta_u, alpha_l, beta_l

# robustVerifier

@njit
def robustVerifier_first_case(UB, LB, act, actd):
    # lower bound
    mid_x = (UB + LB)/2
    alpha_l = actd(mid_x)
    beta_l = act(mid_x) - actd(mid_x) * mid_x
    # upper bounds
    alpha = actd(mid_x)
    alpha_u = alpha
    beta_xl = act(LB) - alpha * LB
    beta_xu = act(UB) - alpha * UB
  
    dis_xl_min = min((alpha * LB + beta_xl - act(LB)), (alpha * UB + beta_xl - act(UB)))

    dis_xu_min = min((alpha * LB + beta_xu - act(LB)), (alpha * UB + beta_xu - act(UB))) 
    
    if (dis_xl_min >= 0) and (dis_xu_min < 0):
        beta_u = beta_xl
    elif (dis_xu_min >= 0) and (dis_xl_min < 0):
        beta_u = beta_xu
    else:
        beta_u = beta_xu if abs(beta_l - beta_xl) > abs(beta_l - beta_xu) else beta_xl
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def veriNet_third_case(UB, LB, act, actd):
    # upper bound
    mid_x = (UB + LB)/2
    alpha_u = actd(mid_x)
    beta_u = act(mid_x) - actd(mid_x) * mid_x
    # lower bounds
    alpha = actd(mid_x)
    alpha_l = alpha
    beta_xl = act(LB) - alpha * LB
    beta_xu = act(UB) - alpha * UB

    dis_xl_min = min((act(LB) - alpha * LB - beta_xl), (act(UB) - alpha * UB - beta_xl))

    dis_xu_min = min((act(LB) - alpha * LB - beta_xu), (act(UB) - alpha * UB - beta_xu)) 
    
    if (dis_xl_min >= 0) and (dis_xu_min < 0):
        beta_l = beta_xl
    elif (dis_xu_min >= 0) and (dis_xl_min < 0):
        beta_l = beta_xu
    else:
        beta_l = beta_xu if abs(beta_u - beta_xl) > abs(beta_u - beta_xu) else beta_xl
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def veriNet_fifth_case(UB, LB, act, actd, actut):
    # upper bound
    du = actut(LB, UB)
    alpha_du_xl = (act(du) - act(LB))/(du - LB)
    alpha_u = alpha_du_xl
    beta_u = act(du) - alpha_du_xl * du
    # lower bound
    alpha = alpha_du_xl
    alpha_l = alpha
    dl = sigmoidloww(LB, UB, alpha)
    beta_dl = act(dl) - alpha * dl
    beta_xu = act(UB) - alpha * UB

    dis_dl_min = min((act(dl) - alpha * dl - beta_dl), (act(UB) - alpha * UB - beta_dl))

    dis_xu_min = min((act(dl) - alpha * dl - beta_xu), (act(UB) - alpha * UB - beta_xu)) 
    
    if (dis_dl_min >= 0) and (dis_xu_min < 0):
        beta_l = beta_dl
    elif (dis_xu_min >= 0) and (dis_dl_min < 0):
        beta_l = beta_xu
    else:
        beta_l = beta_xu if abs(beta_u - beta_dl) > abs(beta_u - beta_xu) else beta_dl
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def robustVerifier_through_zero_case(UB, LB, act, actd):
    ub = UB
    lb = LB
    mid_x = (UB + LB) / 2
    
    if mid_x < 0:

        dl = mid_x
        du = -dl
        
        # interval not contain du 
        if ub < du:
            alpha_u, beta_u, alpha_l, beta_l = robustVerifier_first_case(UB, LB, act, actd)
        else: # interval contain du
            ## upper bound
            alpha_u = actd(dl)
            beta_lb = act(lb) - alpha_u * lb
            du_value = alpha_u * du + beta_lb
            if du_value >= act(du):
                beta_u = beta_lb
            else:
                beta_u = act(du) - alpha_u * du
                
            ## lower bound:
            alpha_l = actd(dl)
            beta_dl = act(dl) - alpha_l * dl
            ub_value = alpha_l * ub + beta_dl
            if ub_value >= act(ub):
                beta_l = act(ub) - alpha_l * ub
            else:
                beta_l = beta_dl
    
    elif mid_x > 0:

        du = mid_x
        dl = -du
        
        # interval not contain dl
        if dl < lb:
            alpha_u, beta_u, alpha_l, beta_l = veriNet_third_case(UB, LB, act, actd)
        else: # interval contain dl
            ## upper bound
            alpha_u = actd(du)
            beta_du = act(du) - alpha_u * du
            lb_value = alpha_u * lb + beta_du
            if lb_value <= act(lb):
                beta_u = act(lb) - alpha_u * lb
            else:
                beta_u = beta_du
                
            ## lower bound
            alpha_l = actd(du)
            beta_ub = act(ub) - alpha_l * ub
            dl_value = alpha_l * dl + beta_ub
            if dl_value >= act(dl):
                beta_l = act(dl) - alpha_l * dl
            else:
                beta_l = beta_ub
    
    else:
        ## uppwer bound
        alpha_u = actd(mid_x)
        beta_u = act(lb) - alpha_u * lb
        
        ## lower bound
        alpha_l = actd(mid_x)
        beta_l = act(ub) - alpha_l * ub
    
    return alpha_u, beta_u, alpha_l, beta_l

# @njit
# def guided_first_case(UB, LB, act, actd, under_LB, under_UB):
#     # upper bound [LB, UB] 两点连线
#     alpha = (act(UB) - act(LB))/(UB - LB)
#     alpha_u = alpha
#     beta_u = act(LB) - alpha * LB
    
#     # lower bound 
#     # 先判断 采样区间的中点是否大于0，若大于0，就不能取中点的切线，因为此时采样区间中点的切线不 sound，则只能取 (0,sigma(0))点切线
#     mid_x = (under_UB + under_LB)/2
#     if mid_x >= 0:
#         alpha_l = actd(0)
#         beta_l = act(0)
#     else:
#         alpha_l = actd(mid_x)
#         beta_l = act(mid_x) - actd(mid_x) * mid_x
    
#     return alpha_u, beta_u, alpha_l, beta_l

# @njit
# def guided_second_case(UB, LB, act, actd, under_LB, under_UB):
#     # upper bound
#     # 先判断 采样区间的中点是否小于0，若小于0，就不能取中点的切线，因为此时采样区间中点的切线不 sound，则只能取 (0,sigma(0))点切线
#     mid_x = (under_UB + under_LB)/2
#     if mid_x <= 0:
#         alpha_u = actd(0)
#         beta_u = act(0) 
#     else:
#         alpha_u = actd(mid_x)
#         beta_u = act(mid_x) - actd(mid_x) * mid_x
        
#     # lower bounds [LB, UB] 两点连线
#     alpha = (act(UB) - act(LB))/(UB - LB)
#     alpha_l = alpha
#     beta_l = act(LB) - alpha * LB
    
#     return alpha_u, beta_u, alpha_l, beta_l

# @njit
# def guided_third_case(UB, LB, act, actd, actut, actlt, under_LB, under_UB):
#     # 先根据 面积最小原则 + [LB, UB] 初始化 upper/lower bound
#     # upper bound
#     du = actut(LB, UB)
#     alpha_du_xl = (act(du) - act(LB))/(du - LB)
#     alpha_u = alpha_du_xl
#     beta_u = act(du) - alpha_du_xl * du
#     # lower bound
#     dl = actlt(LB, UB)
#     alpha_dl_xu = (act(dl) - act(UB))/(dl - UB)
#     alpha_l = alpha_dl_xu
#     beta_l = act(dl) - alpha_dl_xu * dl
    
#     # 再根据采样区间优化 upper/lower bound
#     if under_UB <= 0:
#         mid_x = (under_UB + under_LB)/2
#         alpha_l = actd(mid_x)
#         beta_l = act(mid_x) - actd(mid_x) * mid_x
    
#     if under_LB >= 0:
#         mid_x = (under_UB + under_LB)/2
#         alpha_u = actd(mid_x)
#         beta_u = act(mid_x) - actd(mid_x) * mid_x
        
#     return alpha_u, beta_u, alpha_l, beta_l
    
@njit
def guided_by_median_first_case(UB, LB, act, actd, actut, actlt, median_x):
    # upper bound [LB, UB] 两点连线
    alpha = (act(UB) - act(LB))/(UB - LB)
    alpha_u = alpha
    beta_u = act(LB) - alpha * LB
    
    # lower bound 
    # 先判断 采样区间中位数是否大于0，若大于0，就不能取中位数的切线，因为此时采样区间中点的切线不 sound，则只能取 (0,sigma(0))点切线
    if median_x >= 0:
        mid_x = (UB + LB)/2
        alpha_l = actd(mid_x)
        beta_l = act(mid_x) - actd(mid_x) * mid_x
    else:
        alpha_l = actd(median_x)
        beta_l = act(median_x) - actd(median_x) * median_x
        # if np.isnan(alpha_l):
        #     alpha_l = 0.0
        #     beta_l = 0.0
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def guided_by_median_second_case(UB, LB, act, actd, actut, actlt, median_x):
    # upper bound
    # 先判断 采样区间的中位数是否小于0，若小于0，就不能取中位数的切线，因为此时采样区间中位数的切线不 sound，则只能取 (0,sigma(0))点切线
    if median_x <= 0:
        mid_x = (UB + LB)/2
        alpha_u = actd(mid_x)
        beta_u = act(mid_x) - actd(mid_x) * mid_x
    else:
        alpha_u = actd(median_x)
        beta_u = act(median_x) - actd(median_x) * median_x
        # if np.isnan(alpha_u):
        #     alpha_l = 0.0
        #     beta_l = 1.0
        
    # lower bounds [LB, UB] 两点连线
    alpha = (act(UB) - act(LB))/(UB - LB)
    alpha_l = alpha
    beta_l = act(LB) - alpha * LB
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def guided_by_median_third_case(UB, LB, act, actd, actut, actlt, median_x):
    # 先根据 面积最小原则 + [LB, UB] 初始化 upper/lower bound
    # upper bound
    du = actut(LB, UB)
    alpha_du_xl = (act(du) - act(LB))/(du - LB)
    alpha_u = alpha_du_xl
    beta_u = act(du) - alpha_du_xl * du
    # lower bound
    dl = actlt(LB, UB)
    alpha_dl_xu = (act(dl) - act(UB))/(dl - UB)
    alpha_l = alpha_dl_xu
    beta_l = act(dl) - alpha_dl_xu * dl
    
    # 再根据采样区间优化 upper/lower bound
    if median_x <= dl:
        alpha_l = actd(median_x)
        beta_l = act(median_x) - actd(median_x) * median_x
    
    if median_x >= du:        
        alpha_u = actd(median_x)
        beta_u = act(median_x) - actd(median_x) * median_x
        
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def guided_by_endpoint_first_case(UB, LB, act, actd, actut, actlt, under_l, under_u):
    # upper bound [LB, UB] 两点连线
    alpha = (act(UB) - act(LB))/(UB - LB)
    alpha_u = alpha
    beta_u = act(LB) - alpha * LB
    
    # lower bound 
    if under_l >= 0:
        alpha_l = actd(0)
        beta_l = act(0)
    else:
        alpha_l = actd(under_l)
        beta_l = act(under_l) - actd(under_l) * under_l

    
    # alpha_l = actd(under_l)
    # beta_l = act(under_l) - actd(under_l) * under_l
    # if UB > 0:
    #     dl = actlt(LB, UB)
    #     if under_l > dl:
    #         alpha_dl_xu = (act(dl) - act(UB))/(dl - UB)
    #         alpha_l = alpha_dl_xu
    #         beta_l = act(dl) - alpha_dl_xu * dl
        
        # if np.isnan(alpha_l):
        #     alpha_l = 0.0
        #     beta_l = 0.0
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def guided_by_endpoint_second_case(UB, LB, act, actd, actut, actlt, under_l, under_u):
    # upper bound
    if under_u <= 0:
        alpha_u = actd(0)
        beta_u = act(0) 
    else:
        alpha_u = actd(under_u)
        beta_u = act(under_u) - actd(under_u) * under_u
        
    # alpha_u = actd(under_u)
    # beta_u = act(under_u) - actd(under_u) * under_u
    # if LB <= 0:
    #     du = actut(LB, UB)
    #     if under_u < du:
    #         alpha_du_xl = (act(du) - act(LB))/(du - LB)
    #         alpha_u = alpha_du_xl
    #         beta_u = act(du) - alpha_du_xl * du
        # if np.isnan(alpha_u):
        #     alpha_l = 0.0
        #     beta_l = 1.0
        
    # lower bounds [LB, UB] 两点连线
    alpha = (act(UB) - act(LB))/(UB - LB)
    alpha_l = alpha
    beta_l = act(LB) - alpha * LB
    
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def guided_by_endpoint_third_case(UB, LB, act, actd, actut, actlt, under_l, under_u):
    # 先根据 面积最小原则 + [LB, UB] 初始化 upper/lower bound
    # upper bound
    du = actut(LB, UB)
    alpha_du_xl = (act(du) - act(LB))/(du - LB)
    alpha_u = alpha_du_xl
    beta_u = act(du) - alpha_du_xl * du
    # lower bound
    dl = actlt(LB, UB)
    alpha_dl_xu = (act(dl) - act(UB))/(dl - UB)
    alpha_l = alpha_dl_xu
    beta_l = act(dl) - alpha_dl_xu * dl
    
    # 再根据采样区间优化 upper/lower bound
    if under_l <= dl:
        alpha_l = actd(under_l)
        beta_l = act(under_l) - actd(under_l) * under_l
    
    if under_u >= du:
        alpha_u = actd(under_u)
        beta_u = act(under_u) - actd(under_u) * under_u
        
    return alpha_u, beta_u, alpha_l, beta_l

@njit
def sigmoid_linear_bounds(LB, UB, strategy_map_LB, strategy_map_UB, method):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    
    # print("lower:",LB)
    # print("upper:",UB)
    # print("median:",strategy_map_LB)
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                act = sigmoid
                actd = sigmoidd
                actut = sigmoidut
                actlt = sigmoidlt
                actup = sigmoidup
                actlow = sigmoidlow
                
                if UB[i,j,k] == LB[i,j,k]:
                    alpha_u[i,j,k] = actd(UB[i,j,k])
                    alpha_l[i,j,k] = actd(LB[i,j,k])
                    beta_u[i,j,k] = act(UB[i,j,k])-actd(UB[i,j,k])*UB[i,j,k]
                    beta_l[i,j,k] = act(LB[i,j,k])-actd(LB[i,j,k])*LB[i,j,k]
                
                # sig'(l) > k > sig'(u)
                elif LB[i,j,k] >= 0:
                    if method == 'DeepCert':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_second_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                    elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_third_case(UB[i,j,k], LB[i,j,k], act, actd)
                    elif method == 'NeWise':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_third_case(UB[i,j,k], LB[i,j,k], act, actd)      
                    elif method == 'RobustVerifier':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = veriNet_third_case(UB[i,j,k], LB[i,j,k], act, actd)              
                    elif method == 'guided_by_median':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])
                    elif method == 'guided_by_endpoint':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                
                # sig'(l) < k < sig'(u)    
                elif UB[i,j,k] <= 0:
                    if method == 'DeepCert':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_first_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                    elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_first_case(UB[i,j,k], LB[i,j,k], act, actd)
                    elif method == 'NeWise':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd)   
                    elif method == 'RobustVerifier':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = robustVerifier_first_case(UB[i,j,k], LB[i,j,k], act, actd)                
                    elif method == 'guided_by_median':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])
                    elif method == 'guided_by_endpoint':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                    
                else:
                    
                    if method == 'RobustVerifier':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = robustVerifier_through_zero_case(UB[i,j,k], LB[i,j,k], act, actd)
                        continue
                    
                    alpha = (act(UB[i,j,k])-act(LB[i,j,k]))/(UB[i,j,k]-LB[i,j,k])
                    dU = actd(UB[i,j,k])
                    dL = actd(LB[i,j,k])
                    
                    # sig'(l) < k < sig'(u) 
                    if act(UB[i,j,k])-dU*(UB[i,j,k]-LB[i,j,k]) < act(LB[i,j,k]) and act(LB[i,j,k])+dL*(UB[i,j,k]-LB[i,j,k]) < act(UB[i,j,k]):
                        if method == 'DeepCert':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_first_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                        elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_first_case(UB[i,j,k], LB[i,j,k], act, actd)
                        elif method == 'NeWise':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd)      
                        elif method == 'guided_by_median':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])
                        elif method == 'guided_by_endpoint':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                       
                    # sig'(l) > k > sig'(u)
                    elif act(UB[i,j,k])-dU*(UB[i,j,k]-LB[i,j,k]) > act(LB[i,j,k]) and act(LB[i,j,k])+dL*(UB[i,j,k]-LB[i,j,k]) > act(UB[i,j,k]):
                        if method == 'DeepCert':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_second_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                        elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_third_case(UB[i,j,k], LB[i,j,k], act, actd)
                        elif method == 'NeWise':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_third_case(UB[i,j,k], LB[i,j,k], act, actd)    
                        elif method == 'guided_by_median':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])
                        elif method == 'guided_by_endpoint':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                        
                    # k > sig'(l) and k > sig'(u)
                    else:
                        if method == 'DeepCert':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_third_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt)
                        elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_fifth_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt)
                        elif method == 'NeWise':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_fifth_case(UB[i,j,k], LB[i,j,k], act, actd)      
                        elif method == 'guided_by_median':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_third_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])
                        elif method == 'guided_by_endpoint':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_third_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                            
    return alpha_u, alpha_l, beta_u, beta_l

@njit
def tanh(x):
    return np.tanh(x)

@njit
def tanhd(x):
    return 1.0/np.cosh(x)**2

@njit
def tanhid(x):
    return np.arccosh(1.0/np.sqrt(x))

@njit 
def tanhut(l, u):
    act = tanh
    actd = tanhd
    actid = tanhid
    upper = u
    lower = 0
    al = act(l)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (act(guess)-al)/(guess-l)
        if guesss >= guesst:
            upper = guess
        else:
            lower = guess
    return upper
    
@njit 
def tanhlt(l, u):
    act = tanh
    actd = tanhd
    actid = tanhid
    upper = 0
    lower = l
    au = act(u)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (au-act(guess))/(u-guess)
        if guesss >= guesst:
            lower = guess
        else:
            upper = guess
    return lower

@njit 
def tanhup(l, u, k):
    act = tanh
    actd = tanhd
    upper = u
    lower = max(l,0.0)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            upper = guess
        elif k < guesst:
            lower = guess
        else:
            upper = guess
            break
    return upper

@njit 
def tanhlow(l, u, k):
    act = tanh
    actd = tanhd
    upper = min(u, 0.0)
    lower = l
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            lower = guess
        elif k < guesst:
            upper = guess
        else:
            lower = guess
            break
    return lower

@njit
def tanh_linear_bounds(LB, UB, strategy_map_LB, strategy_map_UB, method):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                act = tanh
                actd = tanhd
                actut = tanhut
                actlt = tanhlt
                actup = tanhup
                actlow = tanhlow
                
                if UB[i,j,k] == LB[i,j,k]:
                    alpha_u[i,j,k] = actd(UB[i,j,k])
                    alpha_l[i,j,k] = actd(LB[i,j,k])
                    beta_u[i,j,k] = act(UB[i,j,k])-actd(UB[i,j,k])*UB[i,j,k]
                    beta_l[i,j,k] = act(LB[i,j,k])-actd(LB[i,j,k])*LB[i,j,k]
                
                elif LB[i,j,k] >= 0:
                    if method == 'DeepCert':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_second_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                    elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_third_case(UB[i,j,k], LB[i,j,k], act, actd)
                    elif method == 'NeWise':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_third_case(UB[i,j,k], LB[i,j,k], act, actd)      
                    elif method == 'RobustVerifier':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = veriNet_third_case(UB[i,j,k], LB[i,j,k], act, actd)  
                    elif method == 'guided_by_median':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])            
                    elif method == 'guided_by_endpoint':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                        
                elif UB[i,j,k] <= 0:
                    if method == 'DeepCert':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_first_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                    elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_first_case(UB[i,j,k], LB[i,j,k], act, actd)
                    elif method == 'NeWise':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd)   
                    elif method == 'RobustVerifier':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = robustVerifier_first_case(UB[i,j,k], LB[i,j,k], act, actd) 
                    elif method == 'guided_by_median':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])               
                    elif method == 'guided_by_endpoint':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                        
                else:
                    
                    if method == 'RobustVerifier':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = robustVerifier_through_zero_case(UB[i,j,k], LB[i,j,k], act, actd)
                        continue
                    
                    alpha = (act(UB[i,j,k])-act(LB[i,j,k]))/(UB[i,j,k]-LB[i,j,k])
                    dU = actd(UB[i,j,k])
                    dL = actd(LB[i,j,k])
                    
                    if act(UB[i,j,k])-dU*(UB[i,j,k]-LB[i,j,k]) < act(LB[i,j,k]) and act(LB[i,j,k])+dL*(UB[i,j,k]-LB[i,j,k]) < act(UB[i,j,k]):
                        if method == 'DeepCert':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_first_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                        elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_first_case(UB[i,j,k], LB[i,j,k], act, actd)
                        elif method == 'NeWise':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd)
                        elif method == 'guided_by_median':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])     
                        elif method == 'guided_by_endpoint':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                            
                    elif act(UB[i,j,k])-dU*(UB[i,j,k]-LB[i,j,k]) > act(LB[i,j,k]) and act(LB[i,j,k])+dL*(UB[i,j,k]-LB[i,j,k]) > act(UB[i,j,k]):
                        if method == 'DeepCert':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_second_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                        elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_third_case(UB[i,j,k], LB[i,j,k], act, actd)
                        elif method == 'NeWise':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_third_case(UB[i,j,k], LB[i,j,k], act, actd)    
                        elif method == 'guided_by_median':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])
                        elif method == 'guided_by_endpoint':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                            
                    else:
                        if method == 'DeepCert':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_third_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt)
                        elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_fifth_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt)
                        elif method == 'NeWise':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_fifth_case(UB[i,j,k], LB[i,j,k], act, actd)     
                        elif method == 'guided_by_median':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_third_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k]) 
                        elif method == 'guided_by_endpoint':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_third_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                            
    return alpha_u, alpha_l, beta_u, beta_l

@njit
def atan(x):
    return np.arctan(x)

@njit
def atand(x):
    return 1.0/(1.0+x**2)

@njit
def atanid(x):
    return np.sqrt(1.0/x-1.0)

@njit 
def atanut(l, u):
    act = atan
    actd = atand
    actid = atanid
    upper = u
    lower = 0
    al = act(l)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (act(guess)-al)/(guess-l)
        if guesss >= guesst:
            upper = guess
        else:
            lower = guess
    return upper
    
@njit 
def atanlt(l, u):
    act = atan
    actd = atand
    actid = atanid
    upper = 0
    lower = l
    au = act(u)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (au-act(guess))/(u-guess)
        if guesss >= guesst:
            lower = guess
        else:
            upper = guess
    return lower

@njit 
def atanup(l, u, k):
    act = atan
    actd = atand
    upper = u
    lower = max(l, 0.0)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            upper = guess
        elif k < guesst:
            lower = guess
        else:
            upper = guess
            break
    return upper

@njit 
def atanlow(l, u, k):
    act = atan
    actd = atand
    upper = u
    lower = min(u, 0.0)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            lower = guess
        elif k < guesst:
            upper = guess
        else:
            lower = guess
            break
    return lower
    


def atan_linear_bounds(LB, UB, strategy_map_LB, strategy_map_UB, method):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                act = atan
                actd = atand
                actut = atanut
                actlt = atanlt
                actup = atanup
                actlow = atanlow
                
                if UB[i,j,k] == LB[i,j,k]:
                    alpha_u[i,j,k] = actd(UB[i,j,k])
                    alpha_l[i,j,k] = actd(LB[i,j,k])
                    beta_u[i,j,k] = act(UB[i,j,k])-actd(UB[i,j,k])*UB[i,j,k]
                    beta_l[i,j,k] = act(LB[i,j,k])-actd(LB[i,j,k])*LB[i,j,k]
                    
                elif LB[i,j,k] >= 0:
                    if method == 'DeepCert':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_second_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                    elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_third_case(UB[i,j,k], LB[i,j,k], act, actd)
                    elif method == 'NeWise':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_third_case(UB[i,j,k], LB[i,j,k], act, actd)      
                    elif method == 'RobustVerifier':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = veriNet_third_case(UB[i,j,k], LB[i,j,k], act, actd)
                    elif method == 'guided_by_median':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])            
                    elif method == 'guided_by_endpoint':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                        
                elif UB[i,j,k] <= 0:
                    if method == 'DeepCert':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_first_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                    elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_first_case(UB[i,j,k], LB[i,j,k], act, actd)
                    elif method == 'NeWise':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd)   
                    elif method == 'RobustVerifier':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = robustVerifier_first_case(UB[i,j,k], LB[i,j,k], act, actd)    
                    elif method == 'guided_by_median':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])                           
                    elif method == 'guided_by_endpoint':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                        
                else:
                    
                    if method == 'RobustVerifier':
                        alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = robustVerifier_through_zero_case(UB[i,j,k], LB[i,j,k], act, actd)
                        continue
                    
                    alpha = (act(UB[i,j,k])-act(LB[i,j,k]))/(UB[i,j,k]-LB[i,j,k])
                    dU = actd(UB[i,j,k])
                    dL = actd(LB[i,j,k])
                    
                    if act(UB[i,j,k])-dU*(UB[i,j,k]-LB[i,j,k]) < act(LB[i,j,k]) and act(LB[i,j,k])+dL*(UB[i,j,k]-LB[i,j,k]) < act(UB[i,j,k]):
                        if method == 'DeepCert':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_first_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                        elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_first_case(UB[i,j,k], LB[i,j,k], act, actd)
                        elif method == 'NeWise':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd)  
                        elif method == 'guided_by_median':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_first_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])                   
                        elif method == 'guided_by_endpoint':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_first_case(UB[i,j,k], LB[i,j,k], act, actd, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                            
                    elif act(UB[i,j,k])-dU*(UB[i,j,k]-LB[i,j,k]) > act(LB[i,j,k]) and act(LB[i,j,k])+dL*(UB[i,j,k]-LB[i,j,k]) > act(UB[i,j,k]):
                        if method == 'DeepCert':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_second_case(UB[i,j,k], LB[i,j,k], act, actd, actup, actlow)
                        elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_third_case(UB[i,j,k], LB[i,j,k], act, actd)
                        elif method == 'NeWise':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_third_case(UB[i,j,k], LB[i,j,k], act, actd)  
                        elif method == 'guided_by_median':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])              
                        elif method == 'guided_by_endpoint':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_second_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                            
                    else:
                        if method == 'DeepCert':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = deepCert_third_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt)
                        elif method == 'VeriNet': # Optimal relaxation based on VeriNet
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = minimal_area_fifth_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt)
                        elif method == 'NeWise':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = endpoint_fifth_case(UB[i,j,k], LB[i,j,k], act, actd)    
                        elif method == 'guided_by_median':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_median_third_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k])   
                        elif method == 'guided_by_endpoint':
                            alpha_u[i,j,k], beta_u[i,j,k], alpha_l[i,j,k], beta_l[i,j,k] = guided_by_endpoint_third_case(UB[i,j,k], LB[i,j,k], act, actd, actut, actlt, strategy_map_LB[i,j,k], strategy_map_UB[i,j,k])
                            
    return alpha_u, alpha_l, beta_u, beta_l
