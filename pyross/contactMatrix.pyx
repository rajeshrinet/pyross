import  numpy as np
cimport numpy as np
cimport cython
import pandas as pd


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
        readonly str country
        np.ndarray CH, CW, CS, CO

    def __init__(self, country):
        self.country = country

        if self.country == 'India':
            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_home_1.xlsx', sheet_name='India')
            self.CH = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_work_1.xlsx', sheet_name='India',index_col=None)
            self.CW = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_school_1.xlsx', sheet_name='India',index_col=None)
            self.CS = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx', sheet_name='India',index_col=None)
            self.CO = np.array(my_data)
    
        elif self.country == 'Italy':
            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_home_1.xlsx', sheet_name='Italy')
            self.CH = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_work_1.xlsx', sheet_name='Italy',index_col=None)
            self.CW = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_school_1.xlsx', sheet_name='Italy',index_col=None)
            self.CS = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx', sheet_name='Italy',index_col=None)
            self.CO = np.array(my_data)
    
        elif self.country == 'China':
            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_home_1.xlsx', sheet_name='China')
            self.CH = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_work_1.xlsx', sheet_name='China',index_col=None)
            self.CW = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_school_1.xlsx', sheet_name='China',index_col=None)
            self.CS = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx', sheet_name='China',index_col=None)
            self.CO = np.array(my_data)
    
        elif self.country == 'Germany':
            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_home_1.xlsx', sheet_name='Germany')
            self.CH = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_work_1.xlsx', sheet_name='Germany',index_col=None)
            self.CW = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_school_1.xlsx', sheet_name='Germany',index_col=None)
            self.CS = np.array(my_data)

            my_data = pd.read_excel('../data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx', sheet_name='Germany',index_col=None)
            self.CO = np.array(my_data)
    
        elif self.country == 'UK':
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
            self.CH = CH
            self.CW = CW
            self.CS = CS
            self.CO = CO
        else:
            raise RuntimeError("Country {0} not implemented".format(self.country))

    cpdef get_individual_contactMatrices(self):
        return self.CH, self.CW, self.CS, self.CO

    def constant_contactMatrix(self):
        cdef:
            np.ndarray C
        C = self.CH + self.CW + self.CS + self.CO
        def C_func(t):
            return C
        return C_func

    def interventions_temporal(self,times,interventions):
        cdef:
            np.ndarray t_arr = np.array(times)
            np.ndarray prefac_arr = np.array(interventions)
            int index
            np.ndarray C
            np.ndarray CH = self.CH, CW = self.CW
            np.ndarray CS = self.CS, CO = self.CO
        # times: ordered array with temporal boundaries between the
        #        different interventions
        # interventions: ordered matrix with prefactors of CW, CS, CO matrices
        #                during the different time intervals
        # note that len(interventions) = len(times) + 1
        def C_func(t):
            index = np.argmin( t_arr < t)
            if index == 0:
                if t >= t_arr[len(t_arr)-1]:
                    index = -1
            #print("t = {0},\tprefac_arr = {1}".format(t,prefac_arr[index]))
            return CH + prefac_arr[index,0]*CW \
                      + prefac_arr[index,1]*CS \
                      + prefac_arr[index,2]*CO
        return C_func


    def interventions_threshold(self,thresholds,interventions):
        cdef:
            np.ndarray thresholds_ = np.array(thresholds)
            np.ndarray prefac_arr = np.array(interventions)
            int index
            np.ndarray C
            np.ndarray CH = self.CH, CW = self.CW
            np.ndarray CS = self.CS, CO = self.CO
        # thresholds: array of shape [K*M,3] with K*M population numbers (S,Ia,Is)
        # interventions: array of shape [K+1,3] with prefactors during different
        #                phases of intervention
        # The current state of the intervention is defined by the
        # largest integer "index" such that state[j] >= thresholds[index,j] for all j.
        #
        def C_func(t,S,Ia,Is):
            state = np.concatenate((S, Ia, Is))
            index = np.argmin((thresholds_ <= state ).all(axis=1))
            if index == 0:
                N = len(thresholds_)
                if (thresholds_[N-1] <= state ).all():
                    index = N
            return CH + prefac_arr[index,0]*CW \
                      + prefac_arr[index,1]*CS \
                      + prefac_arr[index,2]*CO
        return C_func



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
