import numpy as np
import torch
from scipy.optimize import minimize
from system_model import SystemModel

class MPC_Solver:
    def __init__(self,
                 args,
                 d_C,
                 d_s,
                 d_a, 
                 loc,
                 Cm,
                 Cthre,
                 lmd_lbound,
                 lmd_ubound,
                 pred_h=20,
                 alpha=0.9,
                 beta=0.1,   
                 model_path=None,
                 root_dir=None):
        self._d_C = d_C
        self._d_s = d_s
        self._d_a = d_a
        self._loc = loc

        self._Cm = Cm
        self._Cthre = Cthre
        self._lmd_lbound = lmd_lbound
        self._lmd_ubound = lmd_ubound

        self._alpha = alpha
        self._beta = beta

        self._pred_h = pred_h
        self._init_guess = None

        self._model_path = model_path
        

        self._cpu_ranges = np.array([
            [args.cpu1_max, args.cpu1_min],
            [args.cpu2_max, args.cpu2_min] 
        ])
        

        self._minute_cycle = args.minute_cycle
        self._time_increment = 1.0 / (self._minute_cycle - 1)  
        self._time_column = args.time_column 

        self.nn_sys = SystemModel(input_shape=args.input_shape,
                                output_shape=args.output_shape,
                                hidden_units=args.hidden_units,
                                activation=args.activation,
                                output_activation=args.output_activation)
        self.real_sys = SystemModel(input_shape=args.input_shape,
                                  output_shape=args.output_shape,
                                  hidden_units=args.hidden_units,
                                  activation=args.activation,
                                  output_activation=args.output_activation)
        if self._model_path:
            self.nn_sys.load_state_dict(torch.load(self._model_path))
            self.real_sys.load_state_dict(torch.load(self._model_path)) 
        self.x = None
            
    
    def solve_sample(self, st, n=100):
        ls = np.random.uniform(0, 20, size=(n, self._pred_h*self._d_a))
        cost = np.inf
        res = None
        for i in range(n):
            c = self.loss_func(ls[i, :], st)
            if c < cost:
                cost = c
                res = ls[i, :]
        return res[:2]
    
    def loss_func(self, lambdas, st):
        Ct_vec, _ = self.pred_seq(lambdas, st)
        loss = 0
        tm = st[self._time_column]  
        for i in range(self._pred_h+1):
            temp_loss = 0
            if Ct_vec[:, i][0] < self._Cm:
                temp_loss += np.linalg.norm(Ct_vec[:, i] - self._Cm, ord=2) ** 2
            else:
                temp_loss += Ct_vec[0, i] ** 2

            if i > 0:
                if Ct_vec[:, i][0] > self._Cthre:
                    if (round(tm * (self._minute_cycle - 1)) + i) % self._minute_cycle == 1:
                        temp_loss += 2*self._beta * np.linalg.norm(Ct_vec[:, i] - Ct_vec[:, i-1], ord=2) ** 2
                    else:
                        temp_loss += self._beta * np.linalg.norm(Ct_vec[:, i] - Ct_vec[:, i-1], ord=2) ** 2
            loss += (self._alpha ** i) * temp_loss
        return loss
    
    def pred_seq(self, lambdas, st):
        Ct_vec = np.zeros((self._d_C, self._pred_h+1))
        st_vec = np.zeros((self._d_s, self._pred_h+1))

        Ct = st[self._loc:self._loc+self._d_C]
        Ct_vec[:, 0] = Ct[:] * (self._cpu_ranges[:, 0] - self._cpu_ranges[:, 1]) + self._cpu_ranges[:, 1]
        st_vec[:, 0] = st[:]

        for i in range(self._pred_h):
            temp_st = self.model_pred(st_vec[:, i], lambdas[i*self._d_a:(i+1)*self._d_a]).reshape(-1, )
            st_vec[:, i+1] = st_vec[:, i].copy()
            st_vec[self._loc:self._loc+self._d_C, i+1] = temp_st[:self._d_C]

            st_vec[self._time_column, i+1] = (st_vec[self._time_column, i] + self._time_increment) if st_vec[self._time_column, i] + self._time_increment <= 1 else 0
            Ct_vec[:, i+1] = temp_st[:self._d_C] * (self._cpu_ranges[:, 0] - self._cpu_ranges[:, 1]) + self._cpu_ranges[:, 1]
        
        return Ct_vec, st_vec
    
    def model_pred(self, st, lmd):
        with torch.no_grad():
            st = torch.tensor(st, dtype=torch.float32).unsqueeze(0)
            lmd = torch.tensor(lmd, dtype=torch.float32).unsqueeze(0)
            x = torch.cat([st, lmd], axis=1)
            x = x.unsqueeze(1)
            out = self.nn_sys(x)
        return out.numpy()
    
    def model_sml(self, st, lmd):
        peak = False
        if (round(st[self._time_column] * (self._minute_cycle - 1))) % self._minute_cycle == 1:
            peak = True
        with torch.no_grad():
            st = torch.tensor(st, dtype=torch.float32).unsqueeze(0)
            lmd = torch.tensor(lmd, dtype=torch.float32).unsqueeze(0)
            x = torch.cat([st, lmd], axis=1)
            x = x.unsqueeze(1)
            out = self.real_sys(x)
        if peak == False:
            return out.numpy() + np.random.randn(out.shape[0], out.shape[1]) * 1e-2
        else:
            return out.numpy() + np.random.randn(out.shape[0], out.shape[1]) * 1e-2 + np.random.uniform(0, 5, size=(out.shape[0], out.shape[1])) / 100.

    def solve_slsqp(self, st):
        if isinstance(self._init_guess, np.ndarray):
            init = self._init_guess
        else:            
            init = np.random.uniform(0, 1, size=(self._d_a*self._pred_h,))

        bounds = [(self._lmd_lbound, self._lmd_ubound) for _ in range(self._d_a * self._pred_h)]
        res = minimize(fun=self.loss_func,
                       x0=init,
                       method='SLSQP',
                       args=(st,),
                       )
        self._init_guess = res.x

        return res.x[:2]
    
    def finite_diff_gradient(self, lmds, st, eps=1e-6):
        grad = np.zeros_like(lmds)
        for i in range(lmds.shape[0]):
            x_forward = lmds.copy()
            x_backward = lmds.copy()
            x_forward[i] += eps
            x_backward[i] -= eps
            grad[i] = (self.loss_func(x_forward, st) - self.loss_func(x_backward, st)) / (2 * eps)
        
        return grad
    
    def solve_bfgs(self, st):
        if isinstance(self._init_guess, np.ndarray):
            init = self._init_guess
        else:
            init = np.random.uniform(0, 1, size=(self._d_a*self._pred_h,))

        bounds = [(self._lmd_lbound, self._lmd_ubound) for _ in range(self._d_a * self._pred_h)]
        res = minimize(fun=self.loss_func,
                       x0=init,
                       method='BFGS',
                       bounds=bounds,
                       args=(st,),
                       )
        self._init_guess = res.x

        return res.x[:2]
    
    def solve_GD(self, st, step=1e-4, max_iters=100, tol=1e-3):
        if not self._init_guess:
            init = np.ones(self._d_a*self._pred_h)
        else:
            init = self._init_guess
        
        x = init
        i = 0
        while i < max_iters:
            grad = self.finite_diff_gradient(x, st)
            x = x - step * grad
            x = np.clip(x, self._lmd_lbound, self._lmd_ubound)
            if np.linalg.norm(grad, ord=1) < tol:
                break
        
        self._init_guess = x
        
        return x[:2]
