from tools.KRAS_variant import *

def Tricomplex_model(t:float, y:np.array, WT:KRAS_Variant, Mutant:KRAS_Variant, state_parameters:dict[float]) -> np.array:
    """_summary_

    Args:
        t (float): _description_
        y (np.array): _description_
        WT (KRAS_Variant): WT RAS parameters.
        Mutant (KRAS_Variant): Mutant RAS parameters
        state_parameters (dict[float]): State parameters.

    Returns:
        np.array: dydt, rates of change of each concentration in same order as y.
    """
    
    GD=y[0]
    GT=y[1]
    G0=y[2]
    Eff=y[3]
    GTEff=y[4]
    GDV=y[5]
    GTV=y[6]
    G0V=y[7]
    GTEffV=y[8]
    Drug=y[9]
    CYPA=y[10]
    Bicomplex=y[11]
    Tricomplex_WT=y[12]
    Tricomplex_Mutant=y[13]

    VmaxD=WT.k_GDP_GEF*state_parameters['GEF']
    VmaxT=WT.k_GTP_GEF*state_parameters['GEF']*state_parameters['GDP']/state_parameters['GTP']
    KmD=WT.K_m_GDP_GEF
    KmT=WT.K_m_GTP_GEF
    Vmax=WT.k_cat_GAP*state_parameters['GAP']
    Km=WT.K_m_GAP
    kint=WT.k_hyd
    kdissD=WT.k_d_GDP
    kdissT=WT.k_d_GTP
    kassDGDP=WT.k_a_GDP*state_parameters['GDP']
    kassTGTP=WT.k_a_GTP*state_parameters['GTP']
    kassEff=WT.k_a_Eff
    kdissEff=WT.k_d_Eff

    VmaxDV=Mutant.k_GDP_GEF*state_parameters['GEF']
    VmaxTV=Mutant.k_GTP_GEF*state_parameters['GEF']*state_parameters['GDP']/state_parameters['GTP']
    KmDV=Mutant.K_m_GDP_GEF
    KmTV=Mutant.K_m_GTP_GEF
    VmaxV=Mutant.k_cat_GAP*state_parameters['GAP']
    KmV=Mutant.K_m_GAP
    kintV=Mutant.k_hyd
    kdissDV=Mutant.k_d_GDP
    kdissTV=Mutant.k_d_GTP
    kassDGDPV=Mutant.k_a_GDP*state_parameters['GDP']
    kassTGTPV=Mutant.k_a_GTP*state_parameters['GTP']
    kassEffV=Mutant.k_a_Eff
    kdissEffV=Mutant.k_d_Eff

    # Rate Expressions TODO: documentation
    # WT expressions
    R1=(VmaxD*GD/KmD-VmaxT*GT/KmT)/(1+GD/KmD+GT/KmT+GDV/KmDV+GTV/KmTV)
    R2=Vmax*GT/(Km*(1+GTV/KmV)+GT)
    R3=kint*GT
    R4=kdissD*GD-kassDGDP*G0
    R5=kdissT*GT-kassTGTP*G0
    R6=kassEff*GT*Eff-kdissEff*GTEff
    R7=kint*GTEff

    # Mutant expressions
    R8=(VmaxDV*GDV/KmDV-VmaxTV*GTV/KmTV)/(1+GD/KmD+GT/KmT+GDV/KmDV+GTV/KmTV)
    R9=VmaxV*GTV/(KmV*(1+GT/Km)+GTV)
    R10=kintV*GTV
    R11=kdissDV*GDV-kassDGDPV*G0V
    R12=kdissTV*GTV-kassTGTPV*G0V
    R13=kassEffV*GTV*Eff-kdissEffV*GTEffV
    R14=kintV*GTEffV
    
    # Drug related expressions
    k_on_1 = 1e7
    k_off_1 = k_on_1*WT.K_D_1
    
    k_on_2_Mut = 1e7
    k_off_2_Mut = k_on_2_Mut*Mutant.K_D_2
    
    k_on_2_WT = 1e7
    k_off_2_WT = k_on_2_WT*WT.K_D_2
    
    R15=k_on_1*Drug*CYPA - k_off_1*Bicomplex
    R16=k_on_2_WT*GT*Bicomplex - k_off_2_WT*Tricomplex_WT
    R17=k_on_2_Mut*GTV*Bicomplex - k_off_2_Mut*Tricomplex_Mutant

    # System of ODEs TODO doucment
    dydt=[-R1+R2+R3-R4+R7,
        R1-R2-R3-R5-R6-R16,
        R4+R5,
        -R6+R7-R13+R14,
        (R6-R7),
        (-R8+R9+R10-R11+R14),
        (R8-R9-R10-R12-R13-R17),
        (R11+R12),
        (R13-R14),
        -R15,
        -R15,
        R15-R16-R17,
        R16,
        R17
        ]

    return dydt