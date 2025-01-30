import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
import copy
from multiprocessing import Pool
from os import cpu_count
from labellines import labelLine, labelLines
from tqdm import tqdm

from tools.graphing import *
from tools.KRAS_variant import *

class Results:
    """Saves t, y, and initial conditions (y0_original) for time course and calculates functions of on the fly y using indices. e.g Results['total'].
    """
    #TODO: modify so calculations can be called on the entire matrices or the last.

    def __init__(self, t, y, y0_original):

        self.t = t
        self.y = y
        self.y0_original = y0_original

    @property
    def y_ss(self):
        return self.y[:,-1]

    @property
    def total(self):
        return self.y[1,-1]+self.y[4,-1]+self.y[8,-1]+self.y[6,-1]
    
    @property
    def per_RAS_GTP_Tot(self):
        return (self.y[1, -1] + self.y[4, -1] + self.y[6, -1] + self.y[8, -1]) / (self.y0_original[0] + self.y0_original[5]) * 100

    @property
    def per_RAS_GTP_Eff(self):
        return (self.y[4, -1] + self.y[8, -1]) / self.y0_original[3] * 100

    @property
    def per_WT_RAS_GTP(self):
        return (self.y[1, -1] + self.y[4, -1]) / self.y0_original[0] if self.y0_original[0] > 0 else 0

    @property
    def per_WT_RAS_GTP_Eff(self):
        return self.y[4, -1] / self.y0_original[3]

    @property
    def per_Mut_RAS_GTP(self):
        return (self.y[6, -1] + self.y[8, -1]) / self.y0_original[5] * 100 if self.y0_original[5] > 0 else 0

    @property
    def per_Mut_RAS_GTP_Eff(self):
        return self.y[8, -1] / self.y0_original[3]

    @property
    def per_WT_RAS_GTP_Tot(self):
        return (self.y[1, -1] + self.y[4, -1]) / (self.y0_original[0] + self.y0_original[5])

    def __getitem__(self, key): # allows Results['key'] to use key to grab attr which triggers the @property.
        try:
            return self.__getattribute__(key)
        except:
            print(f"Error: attr function for {key} results key does not exist.")
            return None

class ODE_Simulation:
    """Function takes a model, two KRAS_Variants, and state_parameters (initial conditions) and performs ODE integration and other analyses on model.
    #TODO Make this into RAS agnostic class that gets overridden / extended by a RAS_Simulation class.
    """

    def __init__(self, model_fun, WT:KRAS_Variant, Mutant:KRAS_Variant, state_parameters:dict[float]):

        self.model_fun = model_fun
        self.WT = copy.deepcopy(WT)
        self.Mutant = copy.deepcopy(Mutant)
        self.state_parameters = state_parameters

        self.results:Results
        self.saved_Sis = []

        self.labels = []

    def get_modify_params(self, params:list[Parameter], modifications:list[float]) -> tuple[KRAS_Variant,KRAS_Variant,dict[float]]:
        """Copies WT, Mutant, and state_parameters, and makes mofications to them, returning new ones.

        Args:
            params (list[Parameter]): list of Paramter objects to modify. only uses param.name.
            modifications (list[float]): list of multipliers. Must have same order as params. #TODO this isn't robust, make it a list of tuples (Parameter, multiplier)!

        Returns:
            tuple[KRAS_Variant,KRAS_Variant,dict]: fresh parameters for simulation.
        """
        
        WT = copy.deepcopy(self.WT)
        Mutant = copy.deepcopy(self.Mutant)
        state_parameters = copy.deepcopy(self.state_parameters)

        for i,param in enumerate(params):

            if param.type == 'state_parameter': 
                state_parameters[param.name] *= modifications[i]

            if param.type == 'kinetic':
                Mutant.__setattr__(param.name, Mutant.__getattribute__(param.name) * modifications[i])

        return WT, Mutant, state_parameters

    def integrate_model(self, t_end:float, y0:list[float], WT:KRAS_Variant=None, Mutant:KRAS_Variant=None, state_parameters:dict[float]=None, plot_option:bool=False):
        
        if WT is None:
            WT = copy.deepcopy(self.WT)
        if Mutant is None:
            Mutant = copy.deepcopy(self.Mutant)
        if state_parameters is None:
            state_parameters = copy.deepcopy(self.state_parameters)

        # integrate
        sol = solve_ivp(self.model_fun, [0,t_end], y0, args = (WT,Mutant,state_parameters,), method='LSODA', rtol=1e-6, atol=1e-11)

        t = sol['t']
        y = sol['y']
        results = Results(t,y,y0)
        self.results = results

        if plot_option:
            plt.plot(t,np.transpose(y))
            plt.xlabel('t')
            plt.ylabel('y')

        return results
    
    def integrate_model_to_ss(self, y0:list[float], WT = None, Mutant = None, state_parameters = None, tol=1e-17):

        t_max = 10000
        dmet = [1]*len(y0)

        if WT is None:
            WT = copy.deepcopy(self.WT)

        if Mutant is None:
            Mutant = copy.deepcopy(self.Mutant)

        if state_parameters is None:
            state_parameters = copy.deepcopy(self.state_parameters)

        y0_original = copy.deepcopy(y0)

        while np.dot(dmet,dmet)>tol:
            #TODO: collect all t and y into one vector for Results. Would require specifying simulation to start at t_end of previous sim.
            y0_old = y0

            results = self.integrate_model(t_max,y0,WT,Mutant,state_parameters)

            y0=results['y_ss']
            dmet=(y0-y0_old)

        results = Results(results.t,results.y,y0_original)
        
        return results
           
    def response_line(self, y0:list[float], param:Parameter, n:int=50, out_option:str='total', plot_option:bool=True, progress_bar:bool=True) -> tuple[list[float],list[float]]:
        """Creates a graph with an x-axis with parameter multipliers, and a y-axis of model output responses.

        Args:
            y0 (list[float]): initial values.
            param (Parameter): Parameter to modify.
            n (int, optional): number of test points log spaced between Parameter.bounds. Defaults to 50.
            out_option (str, optional): the index of results used as a response to model. Defaults to 'total'.
            plot_option (bool, optional): Turn on plotting automatically. Defaults to True.

        Returns:
            tuple[list[float],list[float]]: param_multipliers, responses.
        """
        
        param_multipliers = np.logspace(param.bounds[0], param.bounds[1], n) #TODO: fix this to make range and plot based on param range type.

        responses = np.zeros(n)
        if progress_bar: tq = tqdm(range(n), desc="Running simulations...")
        for i,multiplier in enumerate(param_multipliers):

            WT, Mutant, state_parameters = self.get_modify_params([param],[multiplier])
            results = self.integrate_model_to_ss(y0, WT, Mutant, state_parameters)

            responses[i] = results[out_option]

            if progress_bar: tq.update()
        if progress_bar: tq.close()

        if plot_option:

            plt.plot(param_multipliers,responses)
            plt.xlabel(f"{param.label} multiplier")
            if out_option == 'signal':
                plt.ylabel('RAS-GTP signal [%]')
            elif out_option == 'total':
                plt.ylabel('total RAS-GTP [M]')
            else:
                plt.ylabel(out_option)
            plt.semilogx()

        return param_multipliers, responses
    
    def response_surface_2D(self, y0, param_1:Parameter, param_2:Parameter, n:int=50, out_option:str='total', plot_option:bool=True, progress_bar:bool=True):
        #TODO: document

        param_1_multipliers = np.logspace(param_1.bounds[0],param_1.bounds[1],n) #TODO: fix this to make range and plot based on range type.
        param_2_multipliers = np.logspace(param_2.bounds[0],param_2.bounds[1],n)

        responses = np.zeros([n,n])
        if progress_bar: tq = tqdm(range(n*n), desc="Running simulations...")
        for i1,mult1 in enumerate(param_1_multipliers):
            for i2,mult2 in enumerate(param_2_multipliers):

                WT, Mutant, state_parameters = self.get_modify_params([param_1,param_2],[mult1,mult2])
                results = self.integrate_model_to_ss(y0, WT, Mutant, state_parameters)

                responses[i1,i2] = results[out_option]

                if progress_bar: tq.update()
        if progress_bar: tq.close()

        if plot_option:
            plt.contourf(param_2_multipliers, param_1_multipliers, responses, levels=25)

            plt.xlabel(f"{param_2.label} multiplier")
            plt.ylabel(f"{param_1.label} multiplier")

            if out_option == 'signal':
                plt.colorbar(label='RAS-GTP signal [%]')
            elif out_option == 'total':
                plt.colorbar(label='total RAS-GTP [M]')

            plt.semilogx()
            plt.semilogy()

        return param_1_multipliers, param_2_multipliers, responses
        
    def sobol_analysis_parralell(self, y0:list[float], params_to_modify:list[Parameter], num_processors=cpu_count(), plot_bar=False, out_option='total') -> tuple[list[str], list[float]]:
        """
        #TODO: document
        """

        d = len(params_to_modify)
        problem = {
            'num_vars': d,
            'names': [param.label for param in params_to_modify],
            'bounds': [param.bounds for param in params_to_modify]
        }
        param_multipliers = saltelli.sample(problem, 1024)
        #print("performing {} {}muM drug simulations".format(1024*2*(d+1),round(drug_dose,4)))
        
        inputs = []
        for multipliers in param_multipliers:

            sim_temp = copy.deepcopy(self)
            params_to_modify_temp = copy.deepcopy(params_to_modify)
            
            WT, Mutant, state_parameters = self.get_modify_params(params_to_modify_temp,multipliers)

            inputs.append({'model':sim_temp,'out_option':out_option, 'y0':y0, 'WT':WT, 'Mutant':Mutant, 'state_parameters':state_parameters})

        pool=Pool(processes = num_processors)
        outputs = pool.map(sim_wrapper,inputs)
        pool.close()
        pool.join()

        Y = np.array(outputs)
        Si = sobol.analyze(problem, Y, print_to_console=False)

        if plot_bar:
            plt.bar(problem['names'],Si['ST'])
            plt.xticks(rotation=90)
            plt.title('ST')
            plt.show()
            plt.bar(problem['names'],Si['S1'])
            plt.xticks(rotation=90)
            plt.title('S1')
            plt.show()
            heatmap(Si['S2'], problem['names'], problem['names'])
            plt.title('S2')
            plt.show()

        return problem['names'],Si

    def random_parameters_parralell(self, y0:list[float], params_to_modify:list[Parameter], n:int=1000, num_processors:int=cpu_count(), out_option:str='total', param_multipliers:list[float]=None) -> list[float]:
        """Performs Monte Carlo sampling, varying each parameter in params_to_modify randomly then runs simulation with the modified parameters, returning list of out_option results. Performs this in paralell.

        Args:
            y0 (list[float]): initial values.
            params_to_modify (list[Parameter]): list of parameters to vary within their respective range_type.
            n (int, optional): number of samples to take. Defaults to 1000.
            num_processors (int, optional): number of paralell processors for Pool. Defaults to cpu_count().
            out_option (str, optional): which results to save. Defaults to 'total'.
            param_multipliers (list[float], optional): Can pass custom random multipliers. This is to perform this experiment on multiple contexts with the same random parameters. Defaults to None.

        Returns:
            list[floats]: list of output option results.
        """

        if param_multipliers is None:
            param_multipliers = get_param_multipliers(params_to_modify,n=n)

        inputs = []
        for multipliers in param_multipliers:

            sim_temp = copy.deepcopy(self)
            params_to_modify_temp = copy.deepcopy(params_to_modify)
            
            WT, Mutant, state_parameters = self.get_modify_params(params_to_modify_temp,multipliers)

            inputs.append({'model':sim_temp,'out_option':out_option, 'y0':y0, 'WT':WT, 'Mutant':Mutant, 'state_parameters':state_parameters})

        pool=Pool(processes = num_processors)
        outputs = pool.map(sim_wrapper,inputs)
        pool.close()
        pool.join()

        Y = np.array(outputs)
        return Y
    
    def spider_plot(self, y0:list[float], params_to_modify:list[Parameter], n:int=50, out_option:str='total'):
        #TODO: document

        # Run simulations
        Ys = []
        param_multipliers = []
        t = tqdm(range(len(params_to_modify)), desc='Running simulations...')
        for param in params_to_modify:
            
            multipliers, Y = self.response_line(y0, param, n=n, out_option=out_option, plot_option=False, progress_bar=False)
            param_multipliers.append(multipliers)
            Ys.append(Y)
            t.update()

        # Plot simulations
        for i,param in enumerate(params_to_modify):
            plt.semilogx(param_multipliers[i], Ys[i], label=param.name)
        plt.xlabel('parameter multiplier')
        plt.ylabel(out_option)
        labelLines()

def sim_wrapper(input:dict) -> Results:
    """Wrapper function to run sims using parallell processing.

    Args:
        input (dict): dict with structure: {'model':ODE_Simulation, 'y0':list[float], 'WT':KRAS_Variant, 'Mutant':KRAS_Variant, state_parameters:dict[float], 'out_option':str}

    Returns:
        Results: Results object with simulation results.
    """
    results = input['model'].integrate_model_to_ss(input['y0'], WT=input['WT'], Mutant=input['Mutant'], state_parameters=input['state_parameters'])
    
    if input['out_option'] == 'all_results':
        return results #type: ignore
    else:
        return results[input['out_option']] #type: ignore

def get_param_multipliers(params_to_modify:list[Parameter],n:int=1000) -> list[list[float]]:
    """Takes list of Parameters and n and returns list of n lists of multiplier flots which each coorespond to each Parameter in the same order as params_to_modify. TODO: make it return list of tuples of param and multiplier.
    Args:
        params_to_modify (list[Parameter]): list of Parameters to modify. Will follow each Parameter.range_type.
        n (int, optional): Number of random multiplier lists to generate. Defaults to 1000.

    Returns:
        list[list[float]]: _description_
    """
    
    param_multipliers = []
    for i in range(n):

        multipliers = []
        for param in params_to_modify:

            if param.range_type == 'log-uniform':
                multiplier = np.power(10,np.random.default_rng().uniform(param.bounds[0],param.bounds[1]))

            elif param.range_type == 'log-normal':
                multiplier = np.random.lognormal(param.bounds[0],param.bounds[1],1)

            elif param.range_type == 'uniform':
                multiplier = np.random.default_rng().uniform(param.bounds[0],param.bounds[1])

            elif param.range_type == 'normal':
                multiplier = np.random.default_rng().normal(param.bounds[0],param.bounds[1])

            else:
                multiplier = None

            multipliers.append(multiplier)

        param_multipliers.append(multipliers)

    return param_multipliers



