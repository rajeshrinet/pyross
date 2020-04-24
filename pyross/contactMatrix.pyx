import  numpy as np
cimport numpy as np
cimport cython

DTYPE   = np.float
ctypedef np.float_t DTYPE_t
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR:
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray rp0, Ni, drpdt, CM, FM

    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')





def India():
    import pandas as pd
    my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_home_1.xlsx', sheet_name='India')
    CH = np.array(my_data)

    my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_work_1.xlsx', sheet_name='India',index_col=None)
    CW = np.array(my_data)
    
    my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_school_1.xlsx', sheet_name='India',index_col=None)
    CS = np.array(my_data)
    
    my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx', sheet_name='India',index_col=None)
    CO = np.array(my_data)
    return CH, CW, CS, CO


def UK():
    import pandas as pd
    my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_home_2.xlsx', sheet_name='United Kingdom of Great Britain')
    CH0 = np.array(my_data)

    my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_work_2.xlsx', sheet_name='United Kingdom of Great Britain',index_col=None)
    CW0 = np.array(my_data)
    
    my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_school_2.xlsx', sheet_name='United Kingdom of Great Britain',index_col=None)
    CS0 = np.array(my_data)
    
    my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_other_locations_2.xlsx', sheet_name='United Kingdom of Great Britain',index_col=None)
    CO0 = np.array(my_data)
    
    #hard coding the first row from the file as panda wont read them
    CH = np.zeros((16, 16))
    CH[0,:]= np.array((0.478812799633172, 0.55185413960287,0.334323605154544,0.132361228266194,0.138531587861408,0.281604887066586,0.406440258772792,0.493947983343078,0.113301080935514,0.0746826413664804,0.0419640342896305,0.0179831987029717,0.00553694264516568,0.00142187285266089,0,0.000505582193632659))
    for i in range(15):
        CH[i+1, :] = CH0[i, :]
    CW = np.zeros((16, 16))
    CW[0,:]= np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000820604524144799,0.0000120585150153575,0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000316436833811157))
    for i in range(15):
        CW[i+1, :] = CW0[i, :]
    CS = np.zeros((16, 16))
    CS[0,:]= np.array((0.974577996106766,0.151369805263473,0.00874880925953218,0.0262790907947637,0.0111281607429249,0.0891043051294382,0.125477587043249,0.0883182775274553,0.0371824197201174,0.0294092695284747,0.0000000000000000000000000000000000000510911446027435,0.0000000000000000000000000000000113982464440009,0.00758428705895781,0.00151636767747242,0.0000000000000000000000000000000000000000000000000123262013953524,0.000000000000000000000000000000000000000000000000000000000000000597486362181075))
    for i in range(15):
        CS[i+1, :] = CS0[i, :]
    CO = np.zeros((16, 16))
    CO[0,:]= np.array((0.257847576361162,0.100135168376607,0.0458036773638843,0.127084549151753,0.187303683093508,0.257979214509792,0.193228849121415,0.336594916946786,0.309223290169635,0.070538522966953,0.152218422246435,0.113554851510519,0.0615771477785246,0.040429874099682,0.0373564987094767,0.00669781557624776))
    for i in range(15):
        CO[i+1, :] = CO0[i, :]
    return CH, CW, CS, CO
