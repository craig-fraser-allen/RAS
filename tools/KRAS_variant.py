import numpy as np
import pandas as pd #type: ignore
import copy
from os.path import join

class Parameter:
    def __init__(self, name:str, bounds:tuple[float], range_type:str, value:float=None, type:str=None, label:str=None):
        """Dataclass to track data on a given parameter.

        Args:
            name (str): name of variable, ***MUST*** coorespond to the attr name in KRAS_Variant.
            bounds (tuple[float]): the bounds on the variable. Used in parameter sweeps.
            range_type (str): description of distribution of parameters used in parameter sweeps.
            value (float, optional): raw value or current value of parameter in simulation. Defaults to None.
            type (str, optional): "kinetic". or "state_parameter" Defaults to None.
            label (str, optional): _description_. Defaults to None.
        """

        self.name = name
        self.label = label
        self.bounds = bounds

        self.range_type = range_type
        if range_type not in ['log-uniform', 'log-normal', 'normal', 'uniform', 'none']: 
            print(f"Warning: Parameter.range_type can only be ['log-uniform', 'log-normal', 'normal', 'uniform'], not {range_type}")

        self.value = value

        self.type = type
        if type not in ['kinetic', 'state_parameter', 'none']: 
            print(f"Warning: Parameter.type can only be state_parameter or kinetic, not {type}.")

class KRAS_Variant:
    def __init__(self, name:str, color:str = 'grey', k_GTP_GEF:float=None):
        """Class used to define kinetic parameters for given RAS mutation. Used as k in ODE integration.

        Args:
            name (str): General string tag for mutation.
            color (str, optional): matplotlib compliant color string for plotting. Defaults to 'grey'.
            kT_GTP_GEF(float, optional): GEF rate of GTP catalysis. Determines dependent parameters. Defaults to None.
        """
        
        self.name = name
        self.color = color
        self.type = None

        # used to estimate 3D to 2D reaction rates.
        self.volscale = 250

        # state variables that are assumed constants.
        self.GAP = 6e-11
        self.GTP = 180e-6
        self.GDP = 18e-6
        self.GEF = 2e-10

        # intrinsic RAS reaction rates. TODO: units.
        self.k_hyd = (3.5e-4)
        self.k_d_GDP = (1.1e-4)
        self.k_d_GTP = (2.5e-4)
        self.k_a_GDP = (2.3e6)
        
        # GAP related kinetic parameters.
        self.k_cat_GAP = (5.4)
        self.K_m_GAP = (.23e-6)/self.volscale

        # GEF related kinetic parameters.
        self.k_GDP_GEF = 3.9
        self.K_m_GDP_GEF = (3.86e-4)/self.volscale
        self.K_m_GTP_GEF = (3e-4)/self.volscale

        # Effector related kinetic parameters.
        self.K_d_Eff=(80e-9)
        self.k_a_Eff=(4.5e7)
            
        # determines if kassT is dependent or kT depedent. WT should be dependent kT, mutants should be kassT dependent. TODO: Check with Ed for rationale.
        if k_GTP_GEF:
            self.type = 'dep_kassT'
            self.k_GTP_GEF = copy.deepcopy(k_GTP_GEF)
            self.k_a_GTP = self.k_GDP_GEF*self.K_m_GTP_GEF*((self.k_a_GDP*self.k_d_GTP)/(self.k_d_GDP*self.k_GTP_GEF))/self.K_m_GDP_GEF
            
        else:
            self.type = 'dep_kT'
            self.k_a_GTP = (2.2e6)
            
        # drug related kinetic parameters. #TODO wasteful to have all of them here, should simply import from csv with row = [attr_name_string, value, source_string]
        self.k_on_SOSi = (1.1e7)/self.volscale
        self.k_off_SOSi = self.k_on_SOSi*(470e-9) # from BI-3407 Hofman Paper, K_D = 470 nmol/L
        self.k_on_tricomplex = (1e7)/self.volscale
        self.K_D_2 = (115e-9)
        self.k_on_panKRASi = 1.6e7 # from https://doi.org/10.1038/s41586-023-06123-3
        self.k_off_panKRASi = 0.042 # from https://doi.org/10.1038/s41586-023-06123-3
        self.k_on_panBRAFi = 1e7/self.volscale
        self.k_off_panBRAFi = self.k_on_panBRAFi*(6.13e-9) #K_D from https://www.sciencedirect.com/science/article/pii/S0021925823002764?via%3Dihub
        self.Kd_KRAS_OFF = 3.7e-6 #from https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b02052
        self.Kd_KRAS_OFF_ON_D = 4e-6
        self.Kd_KRAS_OFF_ON_T = 4e-6

    # Dependent kinetic parameters.
    @property
    def k_GTP_GEF(self):
        Haldaneint=(self.k_a_GDP*self.k_d_GTP)/(self.k_d_GDP*self.k_a_GTP)
        return self.k_GDP_GEF*self.K_m_GTP_GEF*Haldaneint/self.K_m_GDP_GEF
    
    @k_GTP_GEF.setter # if k_GTP_GEF is set, automatically changes k_a_GTP
    def k_GTP_GEF(self, k_GTP_GEF):
        self.k_a_GTP = self.k_GDP_GEF*self.K_m_GTP_GEF*((self.k_a_GDP*self.k_d_GTP)/(self.k_d_GDP*k_GTP_GEF))/self.K_m_GDP_GEF

    @property
    def k_d_Eff(self):
        return self.k_a_Eff*self.K_d_Eff
    
    @property
    def k_off_tricomplex(self):
        return self.k_on_tricomplex*self.K_D_2

    def modify_params(self, params_to_modify:list[Parameter]):
        """ Cycles through list of Parameter objects and sets attrs to their values.

        Args:
            params_to_modify (list[Parameter]): list of Parameter objects. Must have defined value.
        """
        for param in params_to_modify:
            if param.value is None:
                print("Error: param.value = None.")
            if param.type == 'state_parameter':
                print("Error: passed state Parameter to be modified.")
            if param.name not in set([attr for attr in dir(self) if not attr.startswith('__')]):
                print(f"Error: class does not contain attr {param.name}. It will still set it, but it likely won't affect simulation.")
            self.__setattr__(param.name, param.value)
    
def make_KRAS_Variant_from_index(mutants_df:pd.DataFrame, mutant_index:str, color='grey', k_GTP_GEF=None) -> KRAS_Variant:
    """Take a dataframe of mutant parameter multipliers and create a KRAS_Variant using a given mutant index string.
    TODO: make this a function of KRAS_Variant that when itialized, can take these arguments and run this function.

    Args:
        mutants_df (pd.DataFrame): dataframe of mutant multipliers from RAS_ODE_model_kinetic_parameters type file.
        mutant_index (str): name of mutation. Should match row index in mutants_df.
        color (str, optional): Used for plotting purposes. Defaults to 'grey'.
        k_GTP_GEF (_type_, optional): Used to set mutant parameter dependencies. Should use WT.k_GTP_GEF normally.. Defaults to None.

    Returns:
        KRAS_Variant: Mutant parameter object.
    """

    variant = KRAS_Variant(mutant_index,color,k_GTP_GEF=k_GTP_GEF)

    param_multipliers = mutants_df.loc[mutant_index].to_dict()
    params_to_modify = [Parameter(param_ind,(np.nan,np.nan),'none',float(param_value)*copy.deepcopy(variant.__getattribute__(param_ind)), 'none') for param_ind, param_value in param_multipliers.items()]
    
    variant.modify_params(params_to_modify)

    return variant

# Make WT instance. Get dependent kT for use in mutants.
WT = KRAS_Variant('WT',color='forestgreen')
WT_k_GTP_GEF = WT.k_GTP_GEF

# ===== Make Mutant instances. By passing in kT, kassT will automatically be dependent to satisfy detailed balance. ============================================================

# "Mutant" KRAS_Variant that is really WT. Used as control in some simulations.
WT_Mut = KRAS_Variant('WT',color='forestgreen',k_GTP_GEF=WT_k_GTP_GEF)

# Make KRAS_Variants for mutants.
mutants_df = pd.read_excel(join('data','RAS_ODE_model_kinetic_parameters_v2.xlsx'),index_col=0,header=0)

A146T = make_KRAS_Variant_from_index(mutants_df,'A146T',k_GTP_GEF=WT_k_GTP_GEF) # passing in WT k_GTP_GEF to specify that kassT is dependent.
A146V = make_KRAS_Variant_from_index(mutants_df,'A146V',k_GTP_GEF=WT_k_GTP_GEF)
A59T = make_KRAS_Variant_from_index(mutants_df,'A59T',k_GTP_GEF=WT_k_GTP_GEF)
F28L = make_KRAS_Variant_from_index(mutants_df,'F28L',k_GTP_GEF=WT_k_GTP_GEF)
G12A = make_KRAS_Variant_from_index(mutants_df,'G12A',k_GTP_GEF=WT_k_GTP_GEF)
G12C = make_KRAS_Variant_from_index(mutants_df,'G12C',color='royalblue',k_GTP_GEF=WT_k_GTP_GEF)
G12D = make_KRAS_Variant_from_index(mutants_df,'G12D',color='cornflowerblue',k_GTP_GEF=WT_k_GTP_GEF)
G12E = make_KRAS_Variant_from_index(mutants_df,'G12E',k_GTP_GEF=WT_k_GTP_GEF)
G12P = make_KRAS_Variant_from_index(mutants_df,'G12P',k_GTP_GEF=WT_k_GTP_GEF)
G12R = make_KRAS_Variant_from_index(mutants_df,'G12R',k_GTP_GEF=WT_k_GTP_GEF)
G12S = make_KRAS_Variant_from_index(mutants_df,'G12S',k_GTP_GEF=WT_k_GTP_GEF)
G12V = make_KRAS_Variant_from_index(mutants_df,'G12V',color='lightskyblue',k_GTP_GEF=WT_k_GTP_GEF)
G13C = make_KRAS_Variant_from_index(mutants_df,'G13C',k_GTP_GEF=WT_k_GTP_GEF)
G13D = make_KRAS_Variant_from_index(mutants_df,'G13D',color='orchid',k_GTP_GEF=WT_k_GTP_GEF)
G13S = make_KRAS_Variant_from_index(mutants_df,'G13S',k_GTP_GEF=WT_k_GTP_GEF)
G13V = make_KRAS_Variant_from_index(mutants_df,'G13V',k_GTP_GEF=WT_k_GTP_GEF)
Q61H = make_KRAS_Variant_from_index(mutants_df,'Q61H',k_GTP_GEF=WT_k_GTP_GEF)
Q61K = make_KRAS_Variant_from_index(mutants_df,'Q61K',k_GTP_GEF=WT_k_GTP_GEF)
Q61L = make_KRAS_Variant_from_index(mutants_df,'Q61L',k_GTP_GEF=WT_k_GTP_GEF,color='crimson')
Q61P = make_KRAS_Variant_from_index(mutants_df,'Q61P',k_GTP_GEF=WT_k_GTP_GEF)
Q61R = make_KRAS_Variant_from_index(mutants_df,'Q61R',k_GTP_GEF=WT_k_GTP_GEF)
Q61W = make_KRAS_Variant_from_index(mutants_df,'Q61W',k_GTP_GEF=WT_k_GTP_GEF)

all_mutants = [WT_Mut,A146T,A146V,A59T,F28L,G12A,G12C,G12D,G12E,G12P,G12R,G12S,G12V,G13C,G13D,G13S,G13V,Q61H,Q61K,Q61L,Q61P,Q61R,Q61W]  #A59T #10gly11 or 10dupG

# Adjust any drug related parameters to mutation type. #TODO could add columns for these into master data excel file.

# K_D_2s (for KRAS) based on https://www.nature.com/articles/s41586-024-07205-6 extended table 2
G12V.K_D_2 = 84.8e-9
G12C.K_D_2 = 40.3e-9
G12D.K_D_2 = 317e-9
G12R.K_D_2 = 271e-9
G12A.K_D_2 = 128e-9
G13D.K_D_2 = 342e-9
G13C.K_D_2 = 64.5e-9
Q61H.K_D_2 = 87.2e-9

# K_D_2s (for NRAS) based on https://www.nature.com/articles/s41586-024-07205-6 extended table 3
Q61K.K_D_2 = 72e-9
Q61L.K_D_2 = 238e-9
Q61R.K_D_2 = 237e-9

